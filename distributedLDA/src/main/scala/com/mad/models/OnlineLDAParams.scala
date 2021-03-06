package com.mad.models

import org.apache.spark.mllib.linalg.{ Vector, Vectors }

case class OnlineLDAParams(
  vocabSize: Int = 171769,
  alpha: Vector = Vectors.dense(-1),
  eta: Double,
  decay: Double = 1024d,
  learningRate: Double = 0.7,
  maxOutterIter: Int = 100000,
  maxInnerIter: Int = 100,
  convergenceThreshold: Double = 0.001,
  numTopics: Int = 3000,
  totalDocs: Long = 2193062L,
  miniBatchFraction: Double,
  partitions: Int = 100,
  optimizeDocConcentration: Boolean = false,
  perplexity: Boolean = true,
  initLambda: Boolean = true,
  perplexityFreq: Int = 200,
  checkPointFreq: Int = 5000,
  tfRankingMin: Int = 0
) extends Serializable

case class TopicUpdatesMap(
  var topicUpdatesMap: scala.collection.mutable.Map[Long, Array[Double]]
) extends Serializable

case class PermanentParams(
  eta: Double,
  convergenceThreshold: Double,
  numTopics: Int,
  totalDocs: Long
) extends Serializable

case class TemporaryParams(
  rho: Double,
  mbSize: Double
) extends Serializable