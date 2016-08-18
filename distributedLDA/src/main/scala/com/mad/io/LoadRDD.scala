package com.mad.io

import scala.util.Try
import org.apache.spark.rdd.RDD
import com.mad.models.Document

trait LoadRDD {
  def load(): Try[RDD[Document]]
}