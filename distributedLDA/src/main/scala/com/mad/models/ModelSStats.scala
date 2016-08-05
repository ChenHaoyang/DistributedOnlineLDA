package com.mad.models

@SerialVersionUID(7334410294989955983L)
case class ModelSStats[T](lambda: T, numUpdates: Int) extends Serializable