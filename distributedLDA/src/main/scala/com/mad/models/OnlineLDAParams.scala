package com.mad.models

import org.apache.spark.mllib.linalg.{ Vector, Vectors }

case class OnlineLDAParams(
  vocabSize: Int = 171769,
  alpha: Vector = Vectors.dense(-1),
  eta: Double,
  decay: Double = 1024d,
  learningRate: Double = 0.51,
  maxOutterIter: Int = 50000,
  maxInnerIter: Int = 100,
  convergenceThreshold: Double = 0.001,
  numTopics: Int = 3000,
  totalDocs: Long = 2193062L,
  miniBatchFraction: Double,
  optimizeDocConcentration: Boolean = true,
  perplexity: Boolean = true
)