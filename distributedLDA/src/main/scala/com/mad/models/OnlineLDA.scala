package com.mad.models

import breeze.linalg.{ DenseMatrix, DenseVector => BDV }

trait OnlineLDA {

  type BowMinibatch
  type MinibatchSStats
  type LdaModel
  type Lambda
  type Minibatch

  type Gamma = DenseMatrix[Double]

  def eStep(mb: BowMinibatch, model: LdaModel, lambdaSum: BDV[Double], gamma: Gamma): MinibatchSStats

  def mStep(model: LdaModel, mbSStats: MinibatchSStats): LdaModel

}