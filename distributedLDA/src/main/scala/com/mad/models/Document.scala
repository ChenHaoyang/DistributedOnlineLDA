package com.mad.models

case class Document(
  wordIds: IndexedSeq[Long],
  wordCts: IndexedSeq[Double]
)