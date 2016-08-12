package com.mad.models

import java.io._
import java.util.{ Random }
import breeze.linalg.{ DenseMatrix => BDM, DenseVector => BDV, all, normalize, sum }
import breeze.numerics._
import breeze.stats.distributions.{ Gamma, RandBasis }
import breeze.stats.mean
import com.mad.util.Utils
import org.apache.spark.SparkContext
import org.apache.spark.storage.{ StorageLevel }
import org.apache.spark.mllib.linalg.{ DenseVector, Vector, Vectors }
import org.apache.spark.mllib.linalg.distributed.{ IndexedRow, IndexedRowMatrix }
import org.apache.spark.rdd.RDD
import org.apache.hadoop.hbase.spark.HBaseContext
import org.apache.hadoop.hbase.{ TableName }
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.client.{ Scan, Put, Get, Result }
import org.apache.hadoop.hbase.spark.HBaseRDDFunctions._
import org.apache.hadoop.fs.Path
import scala.collection.mutable.{ ArrayBuffer }
import scala.util.Try

class DistributedOnlineLDA(params: OnlineLDAParams) extends OnlineLDA with Serializable {

  override type BowMinibatch = RDD[Document]
  override type MinibatchSStats = (RDD[(Long, Array[Double])], BDM[Double], Int)
  override type LdaModel = ModelSStats[IndexedRowMatrix]
  override type Lambda = IndexedRowMatrix
  override type Minibatch = RDD[String]

  type MatrixRow = Array[Double]

  private var gammaShape: Double = 100

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
   * @param lambdaSum
   * @param gamma
   * @return The sufficient statistics for this minibatch
   */
  override def eStep(
    mb: BowMinibatch,
    model: LdaModel,
    lambdaSum: BDV[Double],
    gamma: Gamma
  ): MinibatchSStats = {

    val mbSize = mb.count().toInt
    val lambdaSumbd = mb.sparkContext.broadcast(lambdaSum)
    val removedDocId = createJoinedRDD(mb, model)

    //Perform e-step on documents as a map, then reduce results by wordId.
    val eStepResult = removedDocId
      .map(wIdCtRow => {
        val currentWordsWeight = wIdCtRow.map(_._3.toArray)
        val currentTopicsMatrix = new BDM( //V * T
          currentWordsWeight.size,
          params.numTopics,
          currentWordsWeight.flatten,
          0,
          params.numTopics,
          true
        )
        val ELogBetaDoc = Utils.dirichletExpectation(currentTopicsMatrix, lambdaSumbd.value) //V * T
        oneDocEStep(
          Document(wIdCtRow.map(_._1), wIdCtRow.map(_._2)),
          model,
          gamma,
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
  def oneDocEStep(
    doc: Document,
    model: LdaModel,
    gamma: BDM[Double],
    ELogBetaDoc: BDM[Double]
  ): MbSStats[Array[(Long, Array[Double])], BDM[Double]] = {

    val wordIds = doc.wordIds
    val wordCts = new BDM(doc.wordCts.size, 1, doc.wordCts.map(_.toDouble).toArray)

    var gammaDoc = gamma

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
      gammaDoc = gammaPreComp :+ model.alpha
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
    mSStats: MinibatchSStats
  ): LdaModel = {

    //mini batch size
    val mbSize = mSStats._3
    val gammaMT = mSStats._2
    val topicUpdates = mSStats._1
    //val wordTotal = topicUpdates.map(_._1).collect()

    val rho = math.pow(params.decay + model.numUpdates, -params.learningRate)

    val newLambdaRows = model
      .lambda
      .rows
      .map(r => (r.index, r.vector.toArray))
      .leftOuterJoin(topicUpdates)
      .map {
        case (rowID, (lambdaRow, rowUpdate)) =>
          if (!rowUpdate.isEmpty) {
            IndexedRow(
              rowID,
              Vectors.dense(
                oneWordMStep(
                  lambdaRow,
                  rowUpdate.get, //OrElse(Array.fill(params.numTopics)(0.0)),
                  rho,
                  mbSize
                )
              )
            )
          } else
            IndexedRow(rowID, Vectors.dense(lambdaRow))
      }
    model.lambda.rows.unpersist()
    newLambdaRows.persist(StorageLevel.MEMORY_AND_DISK)
    val newTopics = new IndexedRowMatrix(newLambdaRows)
    if (params.optimizeDocConcentration) updateAlpha(model, gammaMT, rho)
    model.copy(lambda = newTopics)
  }

  /**
   * Merge the rows of the overall topic matrix and the minibatch topic matrix
   *
   * @param lambdaRow row from overall topic matrix
   * @param updateRow row from minibatch topic matrix
   * @param rho current learning rate
   * @param mbSize number of documents in the minibatch
   * @return merged row.
   */
  def oneWordMStep(
    lambdaRow: MatrixRow,
    updateRow: MatrixRow,
    rho: Double,
    mbSize: Double
  ): MatrixRow = {

    //val rho = math.pow(params.decay + numUpdates, -params.learningRate)
    val updatedLambda1 = lambdaRow.map(_ * (1 - rho))
    val updatedLambda2 = updateRow.map(x => (x * (params.totalDocs.toDouble / mbSize) + params.eta) * rho)

    Utils.arraySum(updatedLambda1, updatedLambda2)
  }

  /**
   * Update alpha based on `gammat`, the inferred topic distributions for documents in the
   * current mini-batch. Uses Newton-Rhapson method.
   * @see Section 3.3, Huang: Maximum Likelihood Estimation of Dirichlet Distribution Parameters
   *      (http://jonathan-huang.org/research/dirichlet/dirichlet.pdf)
   */
  private def updateAlpha(model: LdaModel, gammat: BDM[Double], rho: Double): Unit = {
    val weight = rho
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

  private def loadLambda = (hbaseContext: HBaseContext) => {
    val scan = new Scan()
      .addFamily(Bytes.toBytes("topics"))
      .setCaching(100)
    val matrixRows = hbaseContext.hbaseRDD(TableName.valueOf("lambda"), scan)
      .map(pair => {
        val arrayBuf = new ArrayBuffer[Double]()
        val family = Bytes.toBytes("topics")
        for (i <- 1 to params.numTopics) {
          arrayBuf += Bytes.toDouble(pair._2.getValue(family, Bytes.toBytes(i.toString())))
        }
        val row = new IndexedRow(
          Bytes.toString(pair._2.getRow).toLong,
          Vectors.dense(arrayBuf.toArray)
        )
        arrayBuf.clear()
        row
      })
    matrixRows.persist(StorageLevel.MEMORY_AND_DISK)
    new IndexedRowMatrix(matrixRows)
  }

  /**
   *
   */
  private def initLambda(implicit sc: SparkContext): IndexedRowMatrix = {
    val topics = sc.parallelize(Array.range(0, params.vocabSize - 1), 28)
    val hbaseContext = Utils.getHBaseContext(sc)

    //initialize lambada matrix in HBase
        topics.hbaseBulkPut(hbaseContext, TableName.valueOf("lambda"),
          (wordID) => {
            val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
              new Random(new Random().nextLong()).nextLong()
            ))
            val values = Gamma(gammaShape, 1.0 / gammaShape)(randBasis).sample(params.numTopics).toArray
            val family = Bytes.toBytes("topics")
            val put = new Put(Bytes.toBytes(wordID.toString()))
            for (i <- 1 to params.numTopics) {
              put.addColumn(family, Bytes.toBytes(i.toString()), Bytes.toBytes(values(i - 1)))
            }
            put
          })

    //load lambda from HBase
    loadLambda(hbaseContext)
  }

  private def sumLambda = (lambda: IndexedRowMatrix) => lambda.rows.treeAggregate(BDV.zeros[Double](params.numTopics))(
    seqOp = (c, v) => {
    (c + new BDV(v.vector.toArray))
  },
    combOp = (c1, c2) => {
    c1 + c2
  }
  )

  /**
   * Perform inference to learn the LDA model.
   *
   * @param minibatchIterator
   * @param sc
   * @return A trained LDA model.
   */
  def inference(corpus: RDD[Document]): LdaModel = {

    //initialize lambda matrix V * T
    println("--------------------initializing lambda--------------------")
    val start = System.currentTimeMillis()
    val lambda = initLambda(corpus.sparkContext)
    println("--------------------lambda initialized--------------------")
    println("process time: " + (System.currentTimeMillis() - start) / 1000 + "s")

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

    val heldOut = corpus.sample(true, 100.0 / params.totalDocs.toDouble)

    println("--------------------start iteration--------------------")
    while (mbProcessed <= params.maxOutterIter) {
      val start = System.currentTimeMillis()
      mbProcessed += 1

      //the sum vector(across each row) of lambda matrix 
      val lambdaSum = sumLambda(curModel.lambda)
      println("lambdaSum calculated")

      val gamma = new BDM[Double](
        params.numTopics,
        1,
        Gamma(100.0, 1.0 / 100.0).sample(params.numTopics).toArray
      )
      println("gamma initialized")

      println("==eStep start==")
      val mbSStats = eStep(corpus.sample(true, params.miniBatchFraction), curModel, lambdaSum, gamma)
      println("==eStep end==")

      println("**mStep start**")
      curModel = mStep(curModel.copy(numUpdates = mbProcessed), mbSStats)
      println("**mStep end**")

      println("iteration: " + mbProcessed + ", process time: " + (System.currentTimeMillis() - start) / 1000 + "s")

      if (params.perplexity && mbProcessed % 100 == 0)
        println("logPerplexity: " + logPerplexity(heldOut, curModel, lambdaSum, gamma))

    }
    curModel
  }

  def logPerplexity(
    documents: BowMinibatch,
    model: LdaModel,
    lambdaSum: BDV[Double],
    gamma: Gamma
  ): Double = {

    val tokensCount = documents.map(doc => doc.wordCts.sum).sum
    -logLikelihoodBound(documents, model, lambdaSum, gamma) / tokensCount
  }

  private def logLikelihoodBound(
    documents: BowMinibatch,
    model: LdaModel,
    lambdaSum: BDV[Double],
    gamma: Gamma
  ): Double = {

    val extendedDocRdd = createJoinedRDD(documents, model)
    val lambdaSumbd = documents.sparkContext.broadcast(lambdaSum)

    //calculate L13 + L14 -L22
    val corpusPart =
      extendedDocRdd.map(wIdCtRow => {
        var docBound = 0.0D
        val currentWordsWeight = wIdCtRow.map(_._3.toArray)
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
          model,
          gamma,
          ELogBetaDoc
        ).topicProportions

        val ElogThetad = Utils.dirichletExpectation(gammad)
        wIdCtRow.map(_._2).foreach { count =>
          {
            docBound += count * Utils.logSumExp(ELogBetaDoc(breeze.linalg.*, ::) + ElogThetad.toDenseVector)
          }
        }
        //E[log p(theta | alpha) - log q(theta | gamma)]
        //L11 - L21
        //                    T * 1
        docBound += sum((model.alpha - gammad) :* ElogThetad)
        docBound += sum(lgamma(gammad) - lgamma(model.alpha))
        docBound += lgamma(sum(model.alpha)) - lgamma(sum(gammad))
        docBound
      }).sum

    //Bound component for prob(topic-term distributions):
    //E[log p(beta | eta) - log q(beta | lambda)]
    //L12 - L23
    val sumEta = model.eta * params.vocabSize

    //sum((eta - lambda) :* Elogbeta)
    val part1 = model.lambda.rows.map(row => {
      val rowMat = new BDM[Double](1, params.numTopics, row.vector.toArray)
      val rowElogBeta = Utils.dirichletExpectation(rowMat, lambdaSumbd.value)
      sum((params.eta - rowMat) :* rowElogBeta)
    }).sum

    //sum(lgamma(lambda) - lgamma(eta))
    val part2 = model.lambda.rows.map(row => {
      val rowMat = new BDM[Double](1, params.numTopics, row.vector.toArray)
      sum(lgamma(rowMat) - lgamma(params.eta))
    }).sum
    val topicsPart = part1 + part2
    +sum(lgamma(sumEta) - lgamma(lambdaSum))

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
          for (i <- 1 to params.numTopics) {
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
      model.copy(lambda = loadLambda(Utils.getHBaseContext(sc)))
    }
  }
}