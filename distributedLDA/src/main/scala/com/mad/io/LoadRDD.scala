package com.mad.io

import scala.util.Try
import org.apache.spark.rdd.RDD
import com.mad.models.Document
import org.apache.spark.SparkContext

trait LoadRDD {
  def load(implicit sc: SparkContext): Try[RDD[Document]]
}