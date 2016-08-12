package com.mad.models

import breeze.linalg.{ DenseMatrix => BDM }

@SerialVersionUID(7334410294989955983L)
case class ModelSStats[T](lambda: T, alpha: BDM[Double], eta: Double, numUpdates: Int) extends Serializable