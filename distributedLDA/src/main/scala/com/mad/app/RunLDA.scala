package com.mad.app

import org.apache.spark.{ SparkConf, SparkContext }
import com.mad.models._
import com.mad.io._
import com.mad.util._

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
    val iteration = args(7).toInt

    implicit val sc = new SparkContext(
      new SparkConf().setAppName("DistributedLDA")
    )
    //チェックポイントのパスを設定
    sc.setCheckpointDir(Utils.checkPointPath)

    val lda = new DistributedOnlineLDA(
      OnlineLDAParams(
        vocabSize = totalVcab,
        eta = 1.0 / totalVcab.toDouble,
        learningRate = learningRate,
        maxOutterIter = iteration,
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
    ))

    lda.saveModel(model, new File("/home/charles/Data/output/lda"))

    sc.stop()
  }
}