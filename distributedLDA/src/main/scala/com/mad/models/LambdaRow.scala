package com.mad.models

import org.apache.spark.mllib.linalg.{ Vector }

case class LambdaRow(
  val index: Long,
  var vector: Array[Double]
) extends Serializable
