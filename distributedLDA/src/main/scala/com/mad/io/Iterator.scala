package com.mad.io

import scala.util.Try

trait Iterator[U] {
  def next(): Try[U]
}