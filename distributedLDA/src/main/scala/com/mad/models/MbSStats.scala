package com.mad.models

case class MbSStats[T, U](topicUpdates: T, topicProportions: U, mbSize: Int = 0)