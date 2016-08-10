package com.mad.models

import org.apache.spark.mllib.linalg.{Vector, Vectors}

case class OnlineLDAParams(
  vocabSize: Int = 171769,
  alpha: Vector = Vectors.dense(-1),
  eta: Double,
  decay: Double,
  learningRate: Double,
  maxOutterIter: Int,
  maxInnerIter: Int,
  convergenceThreshold: Double,
  numTopics: Int,
  totalDocs: Long,
  miniBatchFraction: Double,
  optimizeDocConcentration: Boolean = true,
  perplexity: Boolean = false
)