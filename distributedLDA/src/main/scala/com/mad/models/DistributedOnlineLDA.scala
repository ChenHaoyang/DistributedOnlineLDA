package com.mad.models

import java.io._
import java.util.{ Random }
import breeze.linalg.{ DenseMatrix => BDM, DenseVector => BDV, all, normalize, sum }
import breeze.numerics._
import breeze.stats.distributions.{ Gamma, RandBasis }
import breeze.stats.mean
import com.mad.util.Utils
import com.mad.io._
import org.apache.spark.SparkContext
import org.apache.spark.storage.{ StorageLevel }
import org.apache.spark.mllib.linalg.{ DenseVector, Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.{ Broadcast }
import org.apache.hadoop.hbase.spark.HBaseContext
import org.apache.hadoop.hbase.{ TableName }
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.client.{ Scan, Put, Get, Result }
import org.apache.hadoop.hbase.spark.HBaseRDDFunctions._
import org.apache.hadoop.fs.Path
import scala.collection.mutable.{ ArrayBuffer }
import scala.util.Try

class DistributedOnlineLDA(params: OnlineLDAParams)(implicit sc: SparkContext) extends OnlineLDA with Serializable {

  override type BowMinibatch = RDD[Document]
  override type MinibatchSStats = (TopicUpdatesMap, BDM[Double], Int)
  override type LdaModel = ModelSStats[RDD[LambdaRow]]
  override type Minibatch = RDD[String]

  type MatrixRow = Array[Double]

  //private var gammaShape: Double = 100
  private var lambdaSumbd: Broadcast[BDV[Double]] = null
  private var alphabd: Broadcast[BDM[Double]] = null
  private var wordTotalbd: Broadcast[Array[Long]] = null
  //private var topicUpdates: Broadcast[TopicUpdatesMap] = null
  private var permanentParamsbd: Broadcast[PermanentParams] = null
  private var oldWordsLambdaSum: BDV[Double] = null
  //  private var rho: Double = 0
  //  private var mbSize: Int = 0
  /**
   *
   * @param mb
   * @param model
   * @return the joined document RDD
   */
  private def createJoinedRDD(mb: BowMinibatch, model: LdaModel): RDD[Array[(Long, Long, Array[Double])]] = {
    //get the distinct word size of the current mini-batch
    val lambdaRows = model.lambda
    val wordTotalbd = this.wordTotalbd

    val wordIdDocIdCount = mb
      .zipWithIndex()
      .flatMap {
        case (docBOW, docId) =>
          docBOW.wordIds.zip(docBOW.wordCts)
            .map { case (wordId, wordCount) => (wordId, (wordCount, docId)) }
      }

    //Join with lambda to get RDD of (wordId, ((wordCount, docId), lambda(wordId),::)) tuples
    val wordIdDocIdCountRow = wordIdDocIdCount
      .join(lambdaRows.filter { row => wordTotalbd.value.contains(row.index) } //filter the lambda matrix before join
        .map(row => (row.index, row.vector.toArray)))

    //Now group by docID in order to recover documents
    val wordIdCountRow = wordIdDocIdCountRow
      .map { case (wordId, ((wordCount, docId), wordRow)) => (docId, List((wordId, wordCount, wordRow))) }

    //    val removedDocId = wordIdCountRow
    //      .groupByKey()
    //      .map { case (docId, wdIdCtRow) => wdIdCtRow.toArray }

    val removedDocId = wordIdCountRow.reduceByKey((w1, w2) => w1 ::: w2)
      .map(kv => kv._2.toArray)

    //println("mb partition: " + removedDocId.partitions.size)
    removedDocId
  }

  /**
   * Perform E-Step on minibatch.
   *
   * @param mb
   * @param model
   * @return The sufficient statistics for this minibatch
   */
  override def eStep(
    mb: BowMinibatch,
    model: LdaModel
  ): MinibatchSStats = {

    val mbSize = mb.count().toInt
    val removedDocId = createJoinedRDD(mb, model)
    val permanentParamsbd = this.permanentParamsbd
    val alphabd = this.alphabd
    val lambdaSumbd = this.lambdaSumbd

    //Perform e-step on documents as a map, then reduce results by wordId.
    val eStepResult = removedDocId
      .map(wIdCtRow => {
        val localParams = permanentParamsbd.value
        val currentWordsWeight = wIdCtRow.map(_._3.toArray)
        val currentTopicsMatrix = new BDM( //V * T
          currentWordsWeight.size,
          localParams.numTopics,
          currentWordsWeight.flatten,
          0,
          localParams.numTopics,
          true
        )
        val ELogBetaDoc = Utils.dirichletExpectation(currentTopicsMatrix, lambdaSumbd.value) //V * T
        DistributedOnlineLDA.oneDocEStep(
          Document(wIdCtRow.map(_._1), wIdCtRow.map(_._2)),
          localParams.numTopics,
          localParams.convergenceThreshold,
          alphabd.value,
          ELogBetaDoc
        )
      }).persist(StorageLevel.DISK_ONLY)
      .setName("eStepResult")
    //   D * T
    val gammaMT = BDM.vertcat(eStepResult.map(x => x.topicProportions.t).collect(): _*)

    val topicUpdates = eStepResult
      .flatMap(updates => updates.topicUpdates)
      .reduceByKey(Utils.arraySum)

    val topicUpdatesMap = TopicUpdatesMap(scala.collection.mutable.Map(topicUpdates.collect().toMap.toSeq: _*))

    eStepResult.unpersist()

    (topicUpdatesMap, gammaMT, mbSize)
  }

  /**
   * Perform m-step by updating the current model with the minibatch sufficient statistics.
   *
   * @param model
   * @param mSStats
   * @return Updated LDA model
   */
  override def mStep(
    model: LdaModel,
    topicUpdatesMap: TopicUpdatesMap,
    gammaMT: BDM[Double],
    mbSize: Int
  ): (LdaModel, BDV[Double]) = {

    val permanentParamsbd = this.permanentParamsbd
    val rho = math.pow(params.decay + model.numUpdates, -params.learningRate)
    val temporaryParamsbd = model.lambda.sparkContext.broadcast(TemporaryParams(
      rho = rho,
      mbSize = mbSize.toDouble
    ))
    val topicUpdatesbd = model.lambda.sparkContext.broadcast(topicUpdatesMap)

    oldWordsLambdaSum = model.lambda.filter { row =>
      topicUpdatesbd.value.topicUpdatesMap.contains(row.index)
    }.treeAggregate(BDV.zeros[Double](params.numTopics))(
      seqOp = (c, v) => {
      c + BDV(v.vector.toArray)
    },
      combOp = (c1, c2) => {
      c1 + c2
    }
    )

    //println("oldWordsLambdaSum: " + oldWordsLambdaSum(0))
    model.lambda
      .foreach(r => {
        val localTopicUpdates = topicUpdatesbd.value
        if (localTopicUpdates.topicUpdatesMap.contains(r.index)) {
          val localTParams = temporaryParamsbd.value
          val localPParams = permanentParamsbd.value
          val newArray = DistributedOnlineLDA.oneWordMStep(
            r.vector.toArray,
            localTopicUpdates.topicUpdatesMap.get(r.index).get,
            localPParams.totalDocs,
            localPParams.eta,
            localTParams.rho,
            localTParams.mbSize
          )
          r.vector = Vectors.dense(newArray)
        }
      })

    val newWordsLambdaSum = model.lambda.filter { row =>
      topicUpdatesbd.value.topicUpdatesMap.contains(row.index)
    }.treeAggregate(BDV.zeros[Double](params.numTopics))(
      seqOp = (c, v) => {
      c + BDV(v.vector.toArray)
    },
      combOp = (c1, c2) => {
      c1 + c2
    }
    )

    //println("newWordsLambdaSum: " + newWordsLambdaSum(0))

    val lambdaSumUpdate = newWordsLambdaSum - oldWordsLambdaSum

    //topicUpdates.destroy()
    //model.lambda.rows.unpersist()
    //model.lambda = newTopics
    temporaryParamsbd.unpersist(true)
    temporaryParamsbd.destroy()
    topicUpdatesbd.unpersist(true)
    topicUpdatesbd.destroy()
    topicUpdatesMap.topicUpdatesMap.clear()
    topicUpdatesMap.topicUpdatesMap = null

    if (params.optimizeDocConcentration) DistributedOnlineLDA.updateAlpha(model, gammaMT, rho)

    (model, lambdaSumUpdate)
  }

  /**
   *
   */
  private def initLambda(sc: SparkContext, topicNum: Int) = {

    val topics = sc.parallelize(Array.range(0, params.vocabSize - 1), params.partitions)
    val hbaseContext = Utils.getHBaseContext(sc)
    //initialize lambda matrix in HBase
    topics.hbaseBulkPut(hbaseContext, TableName.valueOf("lambda"),
      (wordID) => {
        val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
          new Random(new Random().nextLong()).nextLong()
        ))
        val values = Gamma(100.0, 1.0 / 100.0)(randBasis).sample(topicNum).toArray
        val family = Bytes.toBytes("topics")
        val put = new Put(Bytes.toBytes(wordID.toString()))
        for (i <- 1 to topicNum) {
          put.addColumn(family, Bytes.toBytes(i.toString()), Bytes.toBytes(values(i - 1)))
        }
        put
      })
  }

  /**
   *
   */
  private def loadLambda(sc: SparkContext): RDD[LambdaRow] = {
    val hbaseContext = Utils.getHBaseContext(sc)
    val permanentParamsbd = this.permanentParamsbd
    val scan = new Scan()
      .addFamily(Bytes.toBytes("topics"))
      .setCaching(100)
    val matrixRows = hbaseContext.hbaseRDD(TableName.valueOf("lambda"), scan)
      .map(pair => {
        val localParams = permanentParamsbd.value
        val arrayBuf = new ArrayBuffer[Double]()
        val family = Bytes.toBytes("topics")
        for (i <- 1 to localParams.numTopics) {
          arrayBuf += Bytes.toDouble(pair._2.getValue(family, Bytes.toBytes(i.toString())))
        }
        val row = LambdaRow(
          Bytes.toString(pair._2.getRow).toLong,
          Vectors.dense(arrayBuf.toArray)
        )
        arrayBuf.clear()
        row
      }).repartition(params.partitions)
      .persist(StorageLevel.MEMORY_ONLY)
      .setName("lambda")

    matrixRows
  }

  /**
   *
   */
  private def sumLambda = (lambda: RDD[LambdaRow]) => {
    lambda.treeAggregate(BDV.zeros[Double](params.numTopics))(
      seqOp = (c, v) => {
      (c + new BDV(v.vector.toArray))
    },
      combOp = (c1, c2) => {
      c1 + c2
    }
    )
  }

  /**
   * 潜在トピックの推定を行う
   *
   * @param loader Corpusを読み込み用ローダ
   * @return 訓練されたモデル
   */
  def inference(loader: LoadRDD)(implicit sc: SparkContext): LdaModel = {
    //val testRDD = sc.parallelize(1 to 5, 5).map(x => LambdaRow(x, Vectors.dense(Array(1.0)))).persist(StorageLevel.MEMORY_ONLY)
    //testRDD.foreach { x => x.vector = Vectors.dense(Array(2.0)) }
    //val re = testRDD.collect().foreach(x => print(x.vector(0)))
    //var ttlStart = System.currentTimeMillis()
    permanentParamsbd = sc.broadcast(PermanentParams(
      eta = params.eta,
      convergenceThreshold = params.convergenceThreshold,
      numTopics = params.numTopics,
      totalDocs = params.totalDocs
    ))

    //初期化 lambda matrix V * T
    println("===============================================initializing lambda===============================================")
    var start = System.currentTimeMillis()
    //initLambda(sc)
    val lambda = loadLambda(sc)
    println("process time: " + (System.currentTimeMillis() - start) / 1000 + "s")
    println("===============================================lambda initialized===============================================")

    //corpusとheldOutの初期化
    var corpus = loader.load(sc, params.partitions)
      .get
      .setName("corpus")
    var heldOut = sc.objectFile[Document]("/home/charles/Data/input/heldout")
      .repartition(params.partitions)
      .persist(StorageLevel.MEMORY_AND_DISK)
      .setName("heldOut")

    /** alias for docConcentration */
    def alpha = if (params.alpha.size == 1) {
      if (params.alpha(0) == -1) new BDM(params.numTopics, 1, Array.fill(params.numTopics)(1.0 / params.numTopics))
      else {
        require(
          params.alpha(0) >= 0,
          s"all entries in alpha must be >=0, got: $params.alpha(0)"
        )
        new BDM(params.numTopics, 1, Array.fill(params.numTopics)(params.alpha(0)))
      }
    } else {
      require(
        params.alpha.size == params.numTopics,
        s"alpha must have length k, got: $params.alpha.size"
      )
      params.alpha.toArray.foreach {
        case (x) =>
          require(x >= 0, s"all entries in alpha must be >= 0, got: $x")
      }
      new BDM(params.numTopics, 1, params.alpha.toArray)
    }

    var curModel = ModelSStats[RDD[LambdaRow]](lambda, alpha, params.eta, 0)

    var mbProcessed = 0

    //lambda行列の行和を計算
    println("===============================================calculating lambdaSum===============================================")
    start = System.currentTimeMillis()
    val lambdaSum = sumLambda(curModel.lambda)
    println("LambdaSum time: " + (System.currentTimeMillis() - start) / 1000 + "s")
    println("===============================================lambdaSum calculated===============================================")

    val bufferedWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("/home/charles/Data/output/perplxity.txt", true)))

    lambdaSumbd = sc.broadcast(lambdaSum)
    alphabd = sc.broadcast(curModel.alpha)

    /*Iterationに入る*/
    while (mbProcessed <= params.maxOutterIter) {
      start = System.currentTimeMillis()
      mbProcessed += 1

      //corpusとheldOutの再初期化(for cleaner reason)
      //      if ((System.currentTimeMillis() - ttlStart) / 1000 > 3400) {
      //        ttlStart = System.currentTimeMillis()
      //        //TTLのため、permanentParamsbdを再送する
      //        permanentParamsbd.unpersist(true)
      //        permanentParamsbd.destroy()
      //        permanentParamsbd = sc.broadcast(PermanentParams(
      //          eta = params.eta,
      //          convergenceThreshold = params.convergenceThreshold,
      //          numTopics = params.numTopics,
      //          totalDocs = params.totalDocs
      //        ))
      //
      //        corpus.unpersist()
      //        corpus = loader.load(sc).get
      //        heldOut.unpersist()
      //        heldOut = sc.objectFile("/home/charles/Data/input/heldout")
      //        heldOut.persist(StorageLevel.MEMORY_AND_DISK)
      //      }

      println("===============================================Iteration: " + mbProcessed + "===============================================")
      val mb = corpus.sample(true, params.miniBatchFraction)
        .persist(StorageLevel.MEMORY_AND_DISK)
        .setName("mini_batch")

      wordTotalbd = sc.broadcast(mb.flatMap(docBOW => docBOW.wordIds).distinct().collect())

      val mbSStats = eStep(
        mb,
        curModel
      )

      curModel.numUpdates = mbProcessed

      val mResult = mStep(
        curModel,
        mbSStats._1,
        mbSStats._2,
        mbSStats._3
      )
      curModel = mResult._1
      lambdaSum += mResult._2

      lambdaSumbd.unpersist(true)
      alphabd.unpersist(true)
      lambdaSumbd.destroy()
      alphabd.destroy()

      lambdaSumbd = sc.broadcast(lambdaSum)
      alphabd = sc.broadcast(curModel.alpha)

      if (params.perplexity && (mbProcessed % params.calFreq == 1)) {
        //start = System.currentTimeMillis()
        val logPerplex = logPerplexity(heldOut, curModel)
        println("logPerplexity: " + logPerplex)
        //println("logPerplexity time: " + (System.currentTimeMillis() - start) / 1000 + "s")
        bufferedWriter.write(mbProcessed + " " + logPerplex + "\n")
        bufferedWriter.flush()
        //System.gc()
      }
      val wordNum = wordTotalbd.value.size
      wordTotalbd.unpersist(true)
      wordTotalbd.destroy()

      mb.unpersist()

      println("mbSize: " + mbSStats._3 + ",   wordsNum: " + wordNum)
      println("Iteration time: " + (System.currentTimeMillis() - start) / 1000 + "s")
    }
    bufferedWriter.close()
    curModel
  }

  def logPerplexity(
    documents: BowMinibatch,
    model: LdaModel
  ): Double = {

    val tokensCount = documents.map(doc => doc.wordCts.sum).sum
    -logLikelihoodBound(documents, model) / tokensCount
  }

  private def logLikelihoodBound(
    documents: BowMinibatch,
    model: LdaModel
  ): Double = {

    val extendedDocRdd = createJoinedRDD(documents, model)
    val lambdaSumbd = this.lambdaSumbd
    val permanentParamsbd = this.permanentParamsbd
    //    val numTopics = params.numTopics
    //    val eta = params.eta
    //    val convergenceThreshold = params.convergenceThreshold
    val alphabd = this.alphabd
    //val alpha = model.alpha
    //val paramsBD = extendedDocRdd.sparkContext.broadcast(params)

    //calculate L13 + L14 -L22
    val corpusPart =
      extendedDocRdd.map(wIdCtRow => {
        val localAlpha = alphabd.value
        val localParams = permanentParamsbd.value
        var docBound = 0.0D
        val currentWordsWeight = wIdCtRow.map(_._3)
        val currentTopicsMatrix = new BDM( //V * T
          currentWordsWeight.size,
          localParams.numTopics,
          currentWordsWeight.flatten,
          0,
          localParams.numTopics,
          true
        )
        val ELogBetaDoc = Utils.dirichletExpectation(currentTopicsMatrix, lambdaSumbd.value) //V * T
        val gammad = DistributedOnlineLDA.oneDocEStep(
          Document(wIdCtRow.map(_._1), wIdCtRow.map(_._2)),
          localParams.numTopics,
          localParams.convergenceThreshold,
          localAlpha,
          ELogBetaDoc
        ).topicProportions

        val ElogThetad = Utils.dirichletExpectation(gammad)
        wIdCtRow.map(_._2).zipWithIndex.foreach {
          case (count, idx) =>
            {
              docBound += count * Utils.logSumExp(ELogBetaDoc(idx, ::).t + ElogThetad.toDenseVector)
            }
        }

        //E[log p(theta | alpha) - log q(theta | gamma)]
        //L11 - L21
        //                    T * 1
        docBound += sum((localAlpha - gammad) :* ElogThetad)
        docBound += sum(lgamma(gammad) - lgamma(localAlpha))
        docBound += lgamma(sum(localAlpha)) - lgamma(sum(gammad))

        docBound
      }).sum

    //Bound component for prob(topic-term distributions):
    //E[log p(beta | eta) - log q(beta | lambda)]
    //L12 - L23
    val sumEta = model.eta * params.vocabSize

    //sum((eta - lambda) :* Elogbeta)
    val part1 = model.lambda.map(row => {
      val localParams = permanentParamsbd.value
      val rowMat = new BDM[Double](1, localParams.numTopics, row.vector.toArray)
      val rowElogBeta = Utils.dirichletExpectation(rowMat, lambdaSumbd.value)
      sum((localParams.eta - rowMat) :* rowElogBeta)
    }).sum

    //sum(lgamma(lambda) - lgamma(eta))
    val part2 = model.lambda.map(row => {
      val localParams = permanentParamsbd.value
      val rowMat = new BDM[Double](1, localParams.numTopics, row.vector.toArray)
      sum(lgamma(rowMat) - lgamma(localParams.eta))
    }).sum
    val topicsPart = part1 + part2 + sum(lgamma(sumEta) - lgamma(lambdaSumbd.value))

    corpusPart + topicsPart
  }

  /**
   * Save a learned topic model. Uses Java object serialization.
   */
  def saveModel(model: LdaModel, saveLocation: File)(implicit sc: SparkContext): Try[Unit] =
    Try {
      val lambda = model.lambda
      val hbaseContext = Utils.getHBaseContext(sc)
      lambda.hbaseBulkPut(hbaseContext, TableName.valueOf("lambda"),
        (r) => {
          val values = r.vector.toArray
          val family = Bytes.toBytes("topics")
          val put = new Put(Bytes.toBytes(r.index.toString()))
          val topicNum = values.length
          for (i <- 1 to topicNum) {
            put.addColumn(family, Bytes.toBytes(i.toString()), Bytes.toBytes(values(i - 1)))
          }
          put
        })

      val oos = new ObjectOutputStream(new FileOutputStream(saveLocation))
      oos.writeObject(model.copy(lambda = null))
      oos.close()
    }

  /**
   * Load a saved topic model from save location.
   * Uses Java object deserialization.
   */
  def loadModel(saveLocation: File)(implicit sc: SparkContext): Try[LdaModel] = {
    Try {
      val model = new ObjectInputStream(new FileInputStream(saveLocation))
        .readObject()
        .asInstanceOf[LdaModel]
      model.copy(lambda = loadLambda(sc))
    }
  }
}

object DistributedOnlineLDA {

  private def getGamma(numTopics: Int): BDM[Double] = {
    new BDM[Double](
      numTopics,
      1,
      Gamma(100.0, 1.0 / 100.0).sample(numTopics).toArray
    )
  }

  /**
   * Perform the E-Step on one document (to be performed in parallel via a map)
   *
   * @param doc document from corpus.
   * @param model the current LdaModel
   * @param gamma the corresponding gamma matrix(T * 1)
   * @param ELogBetaDoc ELogBeta matrix of the current document
   * @return Sufficient statistics for this document.
   */
  private def oneDocEStep(
    doc: Document,
    numTopics: Int,
    convergenceThreshold: Double,
    alpha: BDM[Double],
    ELogBetaDoc: BDM[Double]
  ): MbSStats[Array[(Long, Array[Double])], BDM[Double]] = {

    val wordIds = doc.wordIds
    val wordCts = new BDM(doc.wordCts.size, 1, doc.wordCts.map(_.toDouble).toArray)

    var gammaDoc = getGamma(numTopics)

    var expELogThetaDoc = exp(Utils.dirichletExpectation(gammaDoc)) //T * 1
    var expELogBetaDoc = exp(ELogBetaDoc) //V * T                                                       
    var phiNorm = expELogBetaDoc * expELogThetaDoc + 1e-100 //V * 1

    var convergence = false
    var iter = 0

    //begin update iteration.  Stop if convergence has occurred or we reach the max number of iterations.
    while (!convergence) {

      val lastGammaD = gammaDoc.copy
      //                       T * 1               T * V                V * 1                       
      val gammaPreComp = expELogThetaDoc :* (expELogBetaDoc.t * (wordCts :/ phiNorm))
      gammaDoc = gammaPreComp :+ alpha
      expELogThetaDoc = exp(Utils.dirichletExpectation(gammaDoc))
      phiNorm = expELogBetaDoc * expELogThetaDoc + 1e-100

      if (mean(abs(gammaDoc - lastGammaD)) < convergenceThreshold) convergence = true

      iter += 1
    }
    //                                               V * 1                 1 * T                                          
    val lambdaUpdatePreCompute: BDM[Double] = (wordCts :/ phiNorm) * (expELogThetaDoc.t)
    //println("=====================================" + lambdaUpdatePreCompute.rows + "=====================================")

    //Compute lambda row updates and zip with rowIds (note: rowIds = wordIds)
    //                          V * T                  V * T   
    val lambdaUpdate = (lambdaUpdatePreCompute :* expELogBetaDoc).t // T * V
      .toArray
      .grouped(numTopics)
      .toArray
      .zip(wordIds)
      .map(x => (x._2, x._1))

    MbSStats(lambdaUpdate, gammaDoc)
  }

  /**
   * Merge the rows of the overall topic matrix and the minibatch topic matrix
   *
   * @param lambdaRow row from overall topic matrix
   * @param updateRow row from minibatch topic matrix
   * @param mbSize number of documents in the minibatch
   * @return merged row.
   */
  private def oneWordMStep(
    lambdaRow: Array[Double],
    updateRow: Array[Double],
    totalDocs: Long,
    eta: Double,
    rho: Double,
    mbSize: Double
  ): Array[Double] = {

    //val rho = math.pow(params.decay + numUpdates, -params.learningRate)
    val updatedLambda1 = lambdaRow.map(_ * (1.0 - rho))
    val updatedLambda2 = updateRow.map(x => (x * (totalDocs.toDouble / mbSize) + eta) * rho)

    Utils.arraySum(updatedLambda1, updatedLambda2)
  }

  /**
   * Update alpha based on `gammat`, the inferred topic distributions for documents in the
   * current mini-batch. Uses Newton-Rhapson method.
   * @see Section 3.3, Huang: Maximum Likelihood Estimation of Dirichlet Distribution Parameters
   *      (http://jonathan-huang.org/research/dirichlet/dirichlet.pdf)
   */
  private def updateAlpha(model: ModelSStats[RDD[LambdaRow]], gammat: BDM[Double], weight: Double): Unit = {
    //val weight = this.temporaryParamsbd.value.rho
    val N = gammat.rows.toDouble
    //                                                         D * T
    val logphat: BDM[Double] = sum(Utils.dirichletExpectation(gammat)(::, breeze.linalg.*)).t / N //T * 1
    val gradf = N * (-Utils.dirichletExpectation(model.alpha) + logphat)

    val c = N * trigamma(sum(model.alpha))
    val q = -N * trigamma(model.alpha)
    val b = sum(gradf / q) / (1D / c + sum(1D / q))

    val dalpha = -(gradf - b) / q

    if (all((weight * dalpha + model.alpha) :> 0D)) {
      model.alpha :+= weight * dalpha
    }
  }
}