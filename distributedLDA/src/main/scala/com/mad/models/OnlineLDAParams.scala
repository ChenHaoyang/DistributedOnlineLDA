package com.mad.models

import org.apache.spark.mllib.linalg.{Vector, Vectors}

case class OnlineLDAParams(
  vocabSize: Int,
  alpha: Vector = Vectors.dense(0),
  eta: Double,
  decay: Double,
  learningRate: Double,
  maxOutterIter: Int,
  maxInnerIter: Int,
  convergenceThreshold: Double,
  numTopics: Int,
  totalDocs: Int,
  miniBatchFraction: Double,
  optimizeDocConcentration: Boolean = true,
  perplexity: Boolean = false
)