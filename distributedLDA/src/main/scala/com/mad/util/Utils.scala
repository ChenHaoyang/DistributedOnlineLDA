package com.mad.util

import breeze.linalg._
import breeze.numerics._
import com.mad.models.{ Document, ModelSStats }
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.hadoop.hbase.spark.HBaseContext
import org.apache.hadoop.hbase.{ HBaseConfiguration }
import org.apache.hadoop.fs.{ FileSystem, Path }
import org.apache.hadoop.conf.Configuration
import java.io.{ File }
import org.apache.commons.io.FileUtils

object Utils {

  /**
   *
   */
  def getHBaseContext(implicit sc: SparkContext): HBaseContext = {
    val hbaseConf = getHBaseConfig

    new HBaseContext(sc, hbaseConf)
  }
  
  def getHBaseConfig(): Configuration = {
    val hbaseConf = HBaseConfiguration.create()
    hbaseConf.addResource(new Path("/usr/local/hadoop-2.5.0-cdh5.3.9/etc/hadoop/core-site.xml"))
    hbaseConf.addResource(new Path("/usr/local/hadoop-2.5.0-cdh5.3.9/etc/hadoop/hbase-site.xml"))
    
    hbaseConf
  }

  def median[T](s: Seq[T])(implicit n: Fractional[T]) = {
    import n._
    val (lower, upper) = s.sortWith(_ < _).splitAt(s.size / 2)
    if (s.size % 2 == 0) (lower.last + upper.head) / fromInt(2) else upper.head
  }

  def mapVocabId(vocab: Seq[String]): Map[String, Int] =
    vocab
      .distinct
      .zipWithIndex
      .toMap

  def arraySum(a1: Array[Double], a2: Array[Double]): Array[Double] = {

    a1.zip(a2).map { case (a1El1, a2El1) => a1El1 + a2El1 }

  }

  def optionArrayMultiply(a1: Array[Double], a2: Option[Array[Double]]): Array[Double] = {

    a2 match {
      case Some(update) => a1.zip(update).map(x => x._1 * x._2)
      case None => a1
    }
  }

  def dirichletExpectation(srcMatrix: DenseMatrix[Double], sumVector: DenseVector[Double] = null): DenseMatrix[Double] =
    srcMatrix match {

      case x if (x.rows == 1 || x.cols == 1) =>
        if (sumVector == null)
          //digamma(srcMatrix) - digamma(sum(srcMatrix))
          srcMatrix.map { x => if (x < 0) println("vector has minus:" + x); digamma(x) } - digamma(sum(srcMatrix))
        else
          srcMatrix.map { x => if (x < 0) println("vector has minus:" + x); digamma(x) } - digamma(sumVector.toDenseMatrix)
      //digamma(srcMatrix) - digamma(sumVector.toDenseMatrix)

      case _ =>
        val test = sum(srcMatrix, Axis._0)
        val first_term = srcMatrix.map { x => if (x < 0) println("Matrix has minus: " + x); digamma(x) } //digamma(srcMatrix)
        if (sumVector == null)
          //first_term(*, ::) - digamma(sum(srcMatrix, Axis._0)).toDenseVector
          first_term(*, ::) - sum(srcMatrix, Axis._0).map { x => if (x < 0) println("Sum has minus: " + x); digamma(x) }.toDenseVector
        else
          //first_term(*, ::) - digamma(sumVector)
          first_term(*, ::) - sumVector.map { x => if (x < 0) println("Sum has minus: " + x); digamma(x) }
    }

  def getUniqueWords(documents: Seq[Document]): Map[Double, Int] = {

    var uniqueWords: Map[Double, Int] = Map.empty

    documents.foreach { document =>
      document.wordIds.foreach { word =>
        if (!uniqueWords.contains(word)) {
          uniqueWords += (word.toDouble -> uniqueWords.size)
        }
      }
    }

    uniqueWords
  }

  def euclideanDistance(v1: Array[Double], v2: Array[Double]): Double = {

    val sumOfSquareDifference = v1
      .zip(v2)
      .foldLeft(0.0) { (a, i) =>

        val dif = i._1 - i._2

        a + (dif * dif)
      }

    math.sqrt(sumOfSquareDifference)
  }

  def logSumExp(x: DenseVector[Double]): Double = {
    val a = max(x)
    a + log(sum(exp(x :- a)))
  }

  def cleanCheckPoint() {
    val conf = new Configuration()
    //conf.addResource(new Path("/usr/local/hadoop-2.5.0-cdh5.3.9/etc/hadoop/core-site.xml"))
    //conf.addResource(new Path("/usr/local/hadoop-2.5.0-cdh5.3.9/etc/hadoop/hdfs-site.xml"))

    System.setProperty("HADOOP_USER_NAME", "hdfs")
    val fs = FileSystem.get(conf)
    val checkPointPath = new Path(Constants.checkPointPath)
    if (fs.exists(checkPointPath)) {
      fs.delete(checkPointPath, true)
    }
    //fs.mkdirs(checkPointPath)
    //    var passFirstFolder = false
    //
    //    val status = fs.listStatus(new Path(checkPointPath))
    //
    //    status.foreach { x =>
    //      {
    //        if (x.isDirectory()) {
    //          val folder = fs.listStatus(x.getPath)
    //          folder.foreach { f =>
    //            {
    //              val files = fs.listFiles(f.getPath, true)
    //              while (files.hasNext())
    //                fs.delete(files.next.getPath, false)
    //              fs.delete(f.getPath, true)
    //            }
    //          }
    //          if (cleanRoot)
    //            fs.delete(x.getPath, true)
    //        }
    //      }
    //    }
    fs.close()
  }

  def cleanLocalDirectory() = {
    val file = new File(Constants.savePath)
    file.deleteOnExit()
  }
}