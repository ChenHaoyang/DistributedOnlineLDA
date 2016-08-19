package com.mad.models

import breeze.linalg.{ DenseMatrix => BDM, DenseVector => BDV }
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.{ RDD }

trait OnlineLDA {

  type BowMinibatch
  type MinibatchSStats
  type LdaModel
  type Lambda
  type Minibatch

  type Gamma = BDM[Double]

  def eStep(mb: BowMinibatch, model: LdaModel): MinibatchSStats

  def mStep(model: LdaModel, gammaMT: BDM[Double]): (LdaModel, BDV[Double])

}