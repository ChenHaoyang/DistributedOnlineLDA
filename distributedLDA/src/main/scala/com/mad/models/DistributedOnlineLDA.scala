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
import org.apache.spark.mllib.linalg.distributed.{ IndexedRow, IndexedRowMatrix }
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
  override type MinibatchSStats = (RDD[(Long, Array[Double])], BDM[Double], Int)
  override type LdaModel = ModelSStats[IndexedRowMatrix]
  override type Lambda = IndexedRowMatrix
  override type Minibatch = RDD[String]

  type MatrixRow = Array[Double]

  //private var gammaShape: Double = 100
  private var lambdaSumbd: Broadcast[BDV[Double]] = null
  private var alphabd: Broadcast[BDM[Double]] = null
  private var paramsbd = sc.broadcast(params)
  private var topicUpdates: Broadcast[TopicUpdatesMap] = null
  private var rhobd: Broadcast[Double] = null
  private var mbSizebd: Broadcast[Int] = null
  /**
   *
   * @param mb
   * @param model
   * @return the joined document RDD
   */
  private def createJoinedRDD(mb: BowMinibatch, model: LdaModel): RDD[Array[(Long, Long, Array[Double])]] = {
    //get the distinct word size of the current mini-batch
    val wordTotal = mb.flatMap(docBOW => docBOW.wordIds).distinct().collect();

    val wordIdDocIdCount = mb
      .zipWithIndex()
      .flatMap {
        case (docBOW, docId) =>
          docBOW.wordIds.zip(docBOW.wordCts)
            .map { case (wordId, wordCount) => (wordId, (wordCount, docId)) }
      }

    //Join with lambda to get RDD of (wordId, ((wordCount, docId), lambda(wordId),::)) tuples
    val wordIdDocIdCountRow = wordIdDocIdCount
      .join(model.lambda.rows.filter { row => wordTotal.contains(row.index) } //filter the lambda matrix before join
        .map(row => (row.index, row.vector.toArray)))

    //Now group by docID in order to recover documents
    val wordIdCountRow = wordIdDocIdCountRow
      .map { case (wordId, ((wordCount, docId), wordRow)) => (docId, (wordId, wordCount, wordRow)) }

    val removedDocId = wordIdCountRow
      .groupByKey()
      .map { case (docId, wdIdCtRow) => wdIdCtRow.toArray }

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
    //val lambdaSumbd = mb.sparkContext.broadcast(lambdaSum)
    val removedDocId = createJoinedRDD(mb, model)
    //val paramsBD = mb.sparkContext.broadcast(params)

    //Perform e-step on documents as a map, then reduce results by wordId.
    val eStepResult = removedDocId
      .map(wIdCtRow => {
        val localParams = paramsbd.value
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
        oneDocEStep(
          Document(wIdCtRow.map(_._1), wIdCtRow.map(_._2)),
          localParams,
          alphabd.value,
          ELogBetaDoc
        )
      })
    //   D * T
    val gammaMT = BDM.vertcat(eStepResult.map(x => x.topicProportions.t).collect(): _*)

    val topicUpdates = eStepResult
      .flatMap(updates => updates.topicUpdates)
      .reduceByKey(Utils.arraySum)

    (topicUpdates, gammaMT, mbSize)
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
  private def oneDocEStep = (
    doc: Document,
    params: OnlineLDAParams,
    alpha: BDM[Double],
    ELogBetaDoc: BDM[Double]
  ) => {

    val wordIds = doc.wordIds
    val wordCts = new BDM(doc.wordCts.size, 1, doc.wordCts.map(_.toDouble).toArray)

    var gammaDoc = getGamma

    var expELogThetaDoc = exp(Utils.dirichletExpectation(gammaDoc)) //T * 1
    var expELogBetaDoc = exp(ELogBetaDoc) //V * T                                                       
    var phiNorm = expELogBetaDoc * expELogThetaDoc + 1e-100 //V * 1

    var convergence = false
    var iter = 0

    //begin update iteration.  Stop if convergence has occurred or we reach the max number of iterations.
    while (iter < params.maxInnerIter && !convergence) {

      val lastGammaD = gammaDoc.t
      //                       T * 1               T * V                V * 1                       
      val gammaPreComp = expELogThetaDoc :* (expELogBetaDoc.t * (wordCts :/ phiNorm))
      gammaDoc = gammaPreComp :+ alpha
      expELogThetaDoc = exp(Utils.dirichletExpectation(gammaDoc))
      phiNorm = expELogBetaDoc * expELogThetaDoc + 1e-100

      if (mean(abs(gammaDoc.t - lastGammaD)) < params.convergenceThreshold) convergence = true

      iter += 1
    }
    //                                               V * 1                 1 * T                                          
    val lambdaUpdatePreCompute: BDM[Double] = (wordCts :/ phiNorm) * (expELogThetaDoc.t)
    //println("=====================================" + lambdaUpdatePreCompute.rows + "=====================================")

    //Compute lambda row updates and zip with rowIds (note: rowIds = wordIds)
    //                          V * T                  V * T   
    val lambdaUpdate = (lambdaUpdatePreCompute :* expELogBetaDoc).t // T * V
      .toArray
      .grouped(params.numTopics)
      .toArray
      .zip(wordIds)
      .map(x => (x._2, x._1))

    MbSStats(lambdaUpdate, gammaDoc)
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
    gammaMT: BDM[Double]
  ): (LdaModel, BDV[Double]) = {

    //mini batch size
    //    val mbSize = mSStats._3
    //    val gammaMT = mSStats._2
    //val topicUpdates = mSStats._1.sparkContext.broadcast(mSStats._1.collect().toMap)
    //val wordTotal = topicUpdates.map(_._1).collect()
    //val topicUpdatesMap = topicUpdates.collect().toMap
    //val paramsBD = model.lambda.rows.sparkContext.broadcast(params)
    //val rho = math.pow(params.decay + model.numUpdates, -params.learningRate)

    val newLambdaRows = model
      .lambda
      .rows
      //      .map(r => (r.index, r))
      //      .leftOuterJoin(topicUpdates)
      //      .map {
      //        case (rowID, (lambdaRow, rowUpdate)) =>
      //          if (!rowUpdate.isEmpty) {
      //            IndexedRow(
      //              rowID,
      //              Vectors.dense(
      //                oneWordMStep(
      //                  lambdaRow.vector.toArray,
      //                  rowUpdate.get, //OrElse(Array.fill(params.numTopics)(0.0)),
      //                  rho,
      //                  mbSize
      //                )
      //              )
      //            )
      //          } else
      //            lambdaRow
      //      }.persist(StorageLevel.MEMORY_AND_DISK)
      .map(r => {
        val localTopicUpdates = topicUpdates.value
        if (localTopicUpdates.topicUpdatesMap.contains(r.index)) {
          val newArray = oneWordMStep(
            r.vector.toArray,
            localTopicUpdates.topicUpdatesMap.get(r.index).get
          )
          IndexedRow(r.index, Vectors.dense(newArray)) //, BDV(newArray) - BDV(r.vector.toArray))
        } else
          r
      })

    val newTopics = new IndexedRowMatrix(newLambdaRows.persist(StorageLevel.MEMORY_AND_DISK))

    //mark for checkpoint
    if (model.numUpdates % params.checkPointFreq == 0) {
      if (model.numUpdates == params.checkPointFreq)
        Utils.cleanCheckPoint(true)
      else
        Utils.cleanCheckPoint(false)
      newLambdaRows.checkpoint()
    }

    val lambdaSunUpdate = model.lambda.rows.filter { row =>
      topicUpdates.value.topicUpdatesMap.contains(row.index)
    }.zip(newTopics.rows.filter { row =>
      //val vec = row.vector.toArray
      topicUpdates.value.topicUpdatesMap.contains(row.index)
    })
      .treeAggregate(BDV.zeros[Double](params.numTopics))(
        seqOp = (c, v) => {
        c + BDV(v._2.vector.toArray) - BDV(v._1.vector.toArray)
      },
        combOp = (c1, c2) => {
        c1 + c2
      }
      )

    //topicUpdates.destroy()
    model.lambda.rows.unpersist()
    model.lambda = newTopics

    if (params.optimizeDocConcentration) updateAlpha(model, gammaMT)

    (model, lambdaSunUpdate)
  }

  /**
   * Merge the rows of the overall topic matrix and the minibatch topic matrix
   *
   * @param lambdaRow row from overall topic matrix
   * @param updateRow row from minibatch topic matrix
   * @param mbSize number of documents in the minibatch
   * @return merged row.
   */
  private def oneWordMStep = (
    lambdaRow: MatrixRow,
    updateRow: MatrixRow
  ) => {

    //val rho = math.pow(params.decay + numUpdates, -params.learningRate)
    val localParams = paramsbd.value
    val updatedLambda1 = lambdaRow.map(_ * (1 - rhobd.value))
    val updatedLambda2 = updateRow.map(x => (x * (localParams.totalDocs.toDouble / mbSizebd.value) + localParams.eta) * rhobd.value)

    Utils.arraySum(updatedLambda1, updatedLambda2)
  }

  /**
   * Update alpha based on `gammat`, the inferred topic distributions for documents in the
   * current mini-batch. Uses Newton-Rhapson method.
   * @see Section 3.3, Huang: Maximum Likelihood Estimation of Dirichlet Distribution Parameters
   *      (http://jonathan-huang.org/research/dirichlet/dirichlet.pdf)
   */
  private def updateAlpha(model: LdaModel, gammat: BDM[Double]): Unit = {
    val weight = rhobd.value
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

  /**
   *
   */
  private def initLambda(sc: SparkContext, topicNum: Int) = {
    val hbaseContext = Utils.getHBaseContext(sc)
    val topics = sc.parallelize(Array.range(0, params.vocabSize - 1), params.partitions)

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
  private def loadLambda(sc: SparkContext): IndexedRowMatrix = {
    val hbaseContext = Utils.getHBaseContext(sc)
    val scan = new Scan()
      .addFamily(Bytes.toBytes("topics"))
      .setCaching(100)
    val matrixRows = hbaseContext.hbaseRDD(TableName.valueOf("lambda"), scan)
      .map(pair => {
        val arrayBuf = new ArrayBuffer[Double]()
        val family = Bytes.toBytes("topics")
        for (i <- 1 to paramsbd.value.numTopics) {
          arrayBuf += Bytes.toDouble(pair._2.getValue(family, Bytes.toBytes(i.toString())))
        }
        val row = new IndexedRow(
          Bytes.toString(pair._2.getRow).toLong,
          Vectors.dense(arrayBuf.toArray)
        )
        arrayBuf.clear()
        row
      }).repartition(paramsbd.value.partitions)
      .persist(StorageLevel.MEMORY_AND_DISK)

    matrixRows.count()

    new IndexedRowMatrix(matrixRows)
  }

  /**
   *
   */
  private def sumLambda = (lambda: IndexedRowMatrix) => lambda.rows.treeAggregate(BDV.zeros[Double](paramsbd.value.numTopics))(
    seqOp = (c, v) => {
    (c + new BDV(v.vector.toArray))
  },
    combOp = (c1, c2) => {
    c1 + c2
  }
  )

  private def getGamma = {
    val localParams = paramsbd.value
    new BDM[Double](
      localParams.numTopics,
      1,
      Gamma(100.0, 1.0 / 100.0).sample(localParams.numTopics).toArray
    )
  }
  /**
   * 潜在トピックの推定を行う
   *
   * @param loader Corpusを読み込み用ローダ
   * @return 訓練されたモデル
   */
  def inference(loader: LoadRDD)(implicit sc: SparkContext): LdaModel = {
    val broadCastedVals = new ArrayBuffer[Broadcast[TopicUpdatesMap]]()
    val broadCastedDouble = new ArrayBuffer[Broadcast[Double]]()
    val broadCastedInt = new ArrayBuffer[Broadcast[Int]]()

    //初期化 lambda matrix V * T
    println("===============================================initializing lambda===============================================")
    var start = System.currentTimeMillis()
    //initLambda(sc)
    paramsbd = sc.broadcast(params)
    val lambda = loadLambda(sc)
    println("process time: " + (System.currentTimeMillis() - start) / 1000 + "s")
    println("===============================================lambda initialized===============================================")

    //corpusとheldOutの初期化
    println("===============================================initializing corpus===============================================")
    start = System.currentTimeMillis()
    var corpusLifeStart = System.currentTimeMillis()
    var corpus = loader.load().get
    var heldOut = corpus.sample(true, 100.0 / params.totalDocs.toDouble)
    heldOut.persist(StorageLevel.MEMORY_AND_DISK)
    val heldOutArray = heldOut.collect()
    val corpusPartitions = corpus.partitions.length
    println("process time: " + (System.currentTimeMillis() - start) / 1000 + "s")
    println("===============================================corpus initialized===============================================")

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

    var curModel = ModelSStats[IndexedRowMatrix](lambda, alpha, params.eta, 0)

    var mbProcessed = 0

    //lambda行列の行和を計算
    println("===============================================calculating lambdaSum===============================================")
    start = System.currentTimeMillis()
    val lambdaSum = sumLambda(curModel.lambda)
    println("LambdaSum time: " + (System.currentTimeMillis() - start) / 1000 + "s")
    println("===============================================lambdaSum calculated===============================================")

    val bufferedWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("/home/charles/Data/output/perplxity.txt", true)))

    paramsbd.unpersist(true)
    paramsbd.destroy()
    paramsbd = sc.broadcast(params)
    /*Iterationに入る*/
    while (mbProcessed <= params.maxOutterIter) {
      start = System.currentTimeMillis()
      mbProcessed += 1

      //corpusとheldOutの再初期化(for cleaner reason)
      if ((System.currentTimeMillis() - corpusLifeStart) / 1000 > 650) {
        corpusLifeStart = System.currentTimeMillis()
        corpus.unpersist()
        heldOut.unpersist()
        corpus = loader.load().get
        heldOut = corpus.sparkContext.parallelize(heldOutArray, corpusPartitions)
        heldOut.persist(StorageLevel.MEMORY_AND_DISK)
      }

      println("===============================================Iteration: " + mbProcessed + "===============================================")
      //      val gamma = new BDM[Double](
      //        params.numTopics,
      //        1,
      //        Gamma(100.0, 1.0 / 100.0).sample(params.numTopics).toArray
      //      )

      lambdaSumbd = sc.broadcast(lambdaSum)
      alphabd = sc.broadcast(curModel.alpha)
      rhobd = sc.broadcast(math.pow(params.decay + mbProcessed, -params.learningRate))

      val mbSStats = eStep(
        corpus.sample(true, params.miniBatchFraction),
        curModel
      )
      //println("Estep time: " + (System.currentTimeMillis() - start) / 1000 + "s")

      //start = System.currentTimeMillis()
      val topicUpdatesMap = TopicUpdatesMap(scala.collection.mutable.Map(mbSStats._1.collect().toMap.toSeq: _*))
      topicUpdates = sc.broadcast(topicUpdatesMap)
      mbSizebd = sc.broadcast(mbSStats._3)
      curModel.numUpdates = mbProcessed
      val mResult = mStep(
        curModel,
        mbSStats._2
      )
      curModel = mResult._1
      //println("Mstep time: " + (System.currentTimeMillis() - start) / 1000 + "s")

      //start = System.currentTimeMillis()
      lambdaSum += mResult._2

      if (params.perplexity && mbProcessed % (500) == 1) {
        val newLambdaSumbd = corpus.sparkContext.broadcast(lambdaSum.copy)
        //start = System.currentTimeMillis()
        val logPerplex = logPerplexity(heldOut, curModel, newLambdaSumbd)
        println("logPerplexity: " + logPerplex)
        //println("logPerplexity time: " + (System.currentTimeMillis() - start) / 1000 + "s")
        bufferedWriter.write(mbProcessed + " " + logPerplex + "\n")
        bufferedWriter.flush()
        newLambdaSumbd.unpersist()
        newLambdaSumbd.destroy()
        //System.gc()
      }

      if (mbProcessed % params.checkPointFreq == 0) {
        //broadCastedVals.foreach { b => b.destroy() }
        for (i <- 0 to params.checkPointFreq - 2) {
          broadCastedVals(i).unpersist(true)
          broadCastedDouble(i).unpersist(true)
          broadCastedInt(i).unpersist(true)
          broadCastedVals(i).destroy()
          broadCastedDouble(i).destroy()
          broadCastedInt(i).destroy()
        }
        broadCastedVals.clear()
        broadCastedDouble.clear()
        broadCastedInt.clear()
        paramsbd.unpersist(true)
        paramsbd.destroy()
        paramsbd = sc.broadcast(params)
      }

      lambdaSumbd.unpersist(true)
      alphabd.unpersist(true)
      lambdaSumbd.destroy()
      alphabd.destroy()

      topicUpdates.unpersist(true)
      topicUpdatesMap.topicUpdatesMap.clear()
      topicUpdatesMap.topicUpdatesMap = null
      broadCastedVals += topicUpdates
      rhobd.unpersist(true)
      broadCastedDouble += rhobd
      mbSizebd.unpersist(true)
      broadCastedInt += mbSizebd

      println("Iteration time: " + (System.currentTimeMillis() - start) / 1000 + "s")
    }
    bufferedWriter.close()
    curModel
  }

  def logPerplexity(
    documents: BowMinibatch,
    model: LdaModel,
    lambdaSumbd: Broadcast[BDV[Double]]
  ): Double = {

    val tokensCount = documents.map(doc => doc.wordCts.sum).sum
    -logLikelihoodBound(documents, model, lambdaSumbd) / tokensCount
  }

  private def logLikelihoodBound(
    documents: BowMinibatch,
    model: LdaModel,
    lambdaSumbd: Broadcast[BDV[Double]]
  ): Double = {

    val extendedDocRdd = createJoinedRDD(documents, model)
    //val lambdaSumbd = documents.sparkContext.broadcast(lambdaSum)
    //val alpha = model.alpha
    //val paramsBD = extendedDocRdd.sparkContext.broadcast(params)

    //calculate L13 + L14 -L22
    val corpusPart =
      extendedDocRdd.map(wIdCtRow => {
        val params = paramsbd.value
        var docBound = 0.0D
        val currentWordsWeight = wIdCtRow.map(_._3)
        val currentTopicsMatrix = new BDM( //V * T
          currentWordsWeight.size,
          params.numTopics,
          currentWordsWeight.flatten,
          0,
          params.numTopics,
          true
        )
        val ELogBetaDoc = Utils.dirichletExpectation(currentTopicsMatrix, lambdaSumbd.value) //V * T
        val gammad = oneDocEStep(
          Document(wIdCtRow.map(_._1), wIdCtRow.map(_._2)),
          params,
          alphabd.value,
          ELogBetaDoc
        ).topicProportions

        val ElogThetad = Utils.dirichletExpectation(gammad)
        wIdCtRow.map(_._2).zipWithIndex.foreach {
          case (count, idx) =>
            {
              docBound += count * Utils.logSumExp(ELogBetaDoc(idx, ::).t + ElogThetad.toDenseVector)
            }
        }
        val localAlpha = alphabd.value
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
    val part1 = model.lambda.rows.map(row => {
      val params = paramsbd.value
      val rowMat = new BDM[Double](1, params.numTopics, row.vector.toArray)
      val rowElogBeta = Utils.dirichletExpectation(rowMat, lambdaSumbd.value)
      sum((params.eta - rowMat) :* rowElogBeta)
    }).sum

    //sum(lgamma(lambda) - lgamma(eta))
    val part2 = model.lambda.rows.map(row => {
      val params = paramsbd.value
      val rowMat = new BDM[Double](1, params.numTopics, row.vector.toArray)
      sum(lgamma(rowMat) - lgamma(params.eta))
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
      lambda.rows.hbaseBulkPut(hbaseContext, TableName.valueOf("lambda"),
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