package com.mad.app

import org.apache.spark.{ SparkConf, SparkContext }
import com.mad.models._
import com.mad.io._

import java.io.File

object RunLDA {
  def main(args: Array[String]) {
    val topic_num = args(0).toInt
    val totalDocs = args(1).toLong
    val totalVcab = args(2).toInt
    val miniBatch = args(3).toInt
    val partitions = args(4).toInt
    val learningRate = args(5).toDouble
    val checkPoint = args(6).toInt

    implicit val sc = new SparkContext(
      new SparkConf().setAppName("DistributedLDA")
    )

    val lda = new DistributedOnlineLDA(
      OnlineLDAParams(
        vocabSize = totalVcab,
        eta = 1.0 / totalVcab.toDouble,
        learningRate = learningRate,
        totalDocs = totalDocs,
        miniBatchFraction = miniBatch.toDouble / totalDocs.toDouble,
        partitions = partitions,
        checkPointFreq = checkPoint
      )
    )
    //
    val model = lda.inference(new LoadRDDFromHBase(
      "url_info",
      "corpus",
      "doc"
    ).load().get)

    lda.saveModel(model, new File("/home/charles/Data/output/lda"))

    sc.stop()
  }
}