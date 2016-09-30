package com.mad.models

import breeze.linalg.{ DenseMatrix => BDM }

@SerialVersionUID(7334410294989955983L)
case class ModelSStats[T](var lambda: T, var alpha: BDM[Double], var eta: Double, var numUpdates: Int) extends Serializable
