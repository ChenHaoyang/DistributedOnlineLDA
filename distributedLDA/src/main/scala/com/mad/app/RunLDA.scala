package com.mad.app

import org.apache.spark.{ SparkConf, SparkContext }
import com.mad.models._
import com.mad.io._

import java.io.File

object RunLDA {
  def main(args: Array[String]) {
    val topic_num = args(0).toInt
    val totalDocs = args(1).toLong
    val miniBatch = args(2).toInt
    implicit val sc = new SparkContext(
      new SparkConf().setAppName("DistributedLDA")
    )

    val lda = new DistributedOnlineLDA(
      OnlineLDAParams(
        eta = 1.0 / topic_num.toDouble,
        totalDocs = totalDocs,
        miniBatchFraction = miniBatch.toDouble / totalDocs.toDouble
      )
    )

    val model = lda.inference(new LoadRDDFromHBase(
      "url_info",
      "corpus",
      "doc"
    ).load().get)

    lda.saveModel(model, new File("/home/charles/Data/output/lda"))

    sc.stop()
  }
}