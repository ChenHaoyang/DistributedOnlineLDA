package com.mad.app

import org.apache.spark.{ SparkConf, SparkContext }
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
    val checkPointFreq = args(6).toInt
    val iteration = args(7).toInt
    val isContinue = args(8).toBoolean

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
        eta = 1.0 / topic_num.toDouble,
        learningRate = learningRate,
        maxOutterIter = iteration,
        numTopics = topic_num,
        totalDocs = totalDocs,
        miniBatchFraction = miniBatch.toDouble / totalDocs.toDouble,
        partitions = partitions,
        checkPointFreq = checkPointFreq
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