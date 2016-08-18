package com.mad.util

import breeze.linalg._
import breeze.numerics._
import com.mad.models.{ Document, ModelSStats }
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{ IndexedRowMatrix, IndexedRow }
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.hadoop.hbase.spark.HBaseContext
import org.apache.hadoop.hbase.{ HBaseConfiguration }
import org.apache.hadoop.fs.Path
import org.apache.hadoop.fs.{ FileSystem, Path }
import org.apache.hadoop.conf.Configuration

object Utils {

  val checkPointPath = "hdfs://devel-eng01.microad.jp:8020/user/spark/checkpoint"
  /**
   *
   */
  def getHBaseContext(implicit sc: SparkContext): HBaseContext = {
    val hbaseConf = HBaseConfiguration.create()
    hbaseConf.addResource(new Path("/usr/local/hadoop-2.5.0-cdh5.3.9/etc/hadoop/core-site.xml"))
    hbaseConf.addResource(new Path("/usr/local/hadoop-2.5.0-cdh5.3.9/etc/hadoop/hbase-site.xml"))

    new HBaseContext(sc, hbaseConf)
  }

  def mapVocabId(vocab: Seq[String]): Map[String, Int] =
    vocab
      .distinct
      .zipWithIndex
      .toMap

  def denseMatrix2IndexedRows(dm: DenseMatrix[Double]): Array[IndexedRow] = {

    dm(*, ::)
      .map(_.toArray)
      .toArray
      .zipWithIndex
      .map { case (row, index) => IndexedRow(index, Vectors.dense(row)) }

  }

  def rdd2DM(rddRows: RDD[IndexedRow]): DenseMatrix[Double] = {

    val localRows = rddRows
      .collect()
      .sortBy(_.index)
      .map(_.vector.toArray)

    DenseMatrix(localRows: _*)
  }

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
          digamma(srcMatrix) - digamma(sum(srcMatrix))
        else
          digamma(srcMatrix) - digamma(sumVector.toDenseMatrix)

      case _ =>

        val first_term = digamma(srcMatrix)
        if (sumVector == null)
          first_term(*, ::) - digamma(sum(srcMatrix, Axis._0)).toDenseVector
        else
          first_term(*, ::) - digamma(sumVector)
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

  def cleanCheckPoint(cleanRoot: Boolean) {
    val conf = new Configuration()
    //conf.addResource(new Path("/usr/local/hadoop-2.5.0-cdh5.3.9/etc/hadoop/core-site.xml"))
    //conf.addResource(new Path("/usr/local/hadoop-2.5.0-cdh5.3.9/etc/hadoop/hdfs-site.xml"))

    System.setProperty("HADOOP_USER_NAME", "hdfs")
    val fs = FileSystem.get(conf)
    var passFirstFolder = false

    val status = fs.listStatus(new Path(checkPointPath))

    status.foreach { x =>
      {
        if (x.isDirectory()) {
          val folder = fs.listStatus(x.getPath)
          folder.foreach { f =>
            {
              val files = fs.listFiles(f.getPath, true)
              while (files.hasNext())
                fs.delete(files.next.getPath, false)
              fs.delete(f.getPath, true)
            }
          }
          if (cleanRoot)
            fs.delete(x.getPath, true)
        }
      }
    }
    fs.close()
  }

}