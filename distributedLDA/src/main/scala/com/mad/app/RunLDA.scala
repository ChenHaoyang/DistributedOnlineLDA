package com.mad.app

import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.linalg.{ Vectors }
import com.mad.models._
import com.mad.io._
import com.mad.util._

import java.io.File

object RunLDA {
  class Counter extends Serializable { var i: Int = 0; def inc: Unit = i += 1 }
  def main(args: Array[String]) {
    val topic_num = args(0).toInt
    val totalDocs = args(1).toLong
    val totalVcab = args(2).toInt
    val miniBatch = args(3).toInt
    val partitions = args(4).toInt
    val learningRate = args(5).toDouble
    val decay = args(6).toDouble
    val eta = args(7).toDouble
    val checkPointFreq = args(8).toInt
    val perplexityFreq = args(9).toInt
    val iteration = args(10).toInt
    val isContinue = args(11).toBoolean
    val initLambda = args(12).toBoolean
    val alpha = args(13).toDouble

    val conf = new SparkConf().setAppName("DistributedLDA")
    conf.registerKryoClasses(Array(
      classOf[TopicUpdatesMap],
      classOf[PermanentParams],
      classOf[TemporaryParams],
      classOf[LambdaRow]
    ))
    implicit val sc = new SparkContext(
      conf
    )
    //val cachedRdd = sc.parallelize(1 to 5, 5).map(x => new Counter()).cache
    //
    //
    //// trying to apply modification
    //
    //
    //cachedRdd.foreachPartition { p => p.map(x => x.inc) }
    //
    //
    //// modification worked: all values are ones
    //
    //
    //cachedRdd.collect.foreach(x => println(x.i))
    //チェックポイントのパスを設定
    //sc.setCheckpointDir(Utils.checkPointPath)

    val lda = new DistributedOnlineLDA(
      OnlineLDAParams(
        vocabSize = totalVcab,
        alpha = Vectors.dense(alpha),
        eta = eta,
        learningRate = learningRate,
        decay = decay,
        maxOutterIter = iteration,
        numTopics = topic_num,
        totalDocs = totalDocs,
        miniBatchFraction = miniBatch.toDouble / totalDocs.toDouble,
        partitions = partitions,
        checkPointFreq = checkPointFreq,
        perplexityFreq = perplexityFreq,
        initLambda = initLambda
      )
    )
    val rddLoader = new LoadRDDFromHBase(
      "url_info",
      "corpus",
      "doc"
    )
    val model = {
      if (isContinue) {
        println("continue")
        lda.continueTraining(rddLoader)
      } else {
        println("new")
        lda.inference(rddLoader)
      }
    }

    lda.saveModel(model)
    //Utils.cleanCheckPoint()

    sc.stop()
  }
}