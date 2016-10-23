package com.mad.io

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.hadoop.hbase.client._
import org.apache.hadoop.hbase.{TableName}
import org.apache.hadoop.hbase.util.Bytes
import scala.util.Try
import com.mad.util.Utils
import com.mad.models.Document

import scala.collection.mutable.ArrayBuffer

class DocRDDIterator(
    tableName: String,
    familyName: String,
    colName: String,
    miniBatchSize: Int
    )
(implicit sc : SparkContext) extends Iterator[RDD[(Long, Document)]]{
  
  val conf = Utils.getHBaseConfig()
  val conn = ConnectionFactory.createConnection(conf)
  val table = conn.getTable(TableName.valueOf(tableName))
  val scan = new Scan()
  .addColumn(Bytes.toBytes(familyName), Bytes.toBytes(colName))
  .setCaching(100)
  val rs = table.getScanner(scan)
  
  override def next() = {
    Try{
      val rsArray = rs.next(miniBatchSize)
      if(rsArray.size == 0) {
        rs.close()
        table.close()
        conn.close()
        null}
      else{
        val docArray = rsArray.map { x => {
          val url_id = Bytes.toString(x.getRow).toLong
          val doc = Bytes.toString(x.value).split(",")
          val id = doc(0).split(":").map { x => x.toLong }
          val cnt = doc(1).split(":").map { x => x.toDouble }
          (url_id, Document(id, cnt))
        } }
        sc.parallelize(docArray.toSeq, 24)
      }
    }
  }
}