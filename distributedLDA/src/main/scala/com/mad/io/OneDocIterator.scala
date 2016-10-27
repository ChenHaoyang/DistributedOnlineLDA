package com.mad.io

import org.apache.hadoop.hbase.client._
import org.apache.hadoop.hbase.TableName
import org.apache.hadoop.hbase.util.Bytes
import scala.collection.immutable.{Map}
import scala.util.Try
import com.mad.util.Utils

class OneDocIterator(
    tableName: String,
    familyName: String,
    colName: String
    ) extends Iterator[(Long, Map[Int, Double])]{
  
  val conf = Utils.getHBaseConfig()
  val conn = ConnectionFactory.createConnection(conf)
  val table = conn.getTable(TableName.valueOf(tableName))
  val scan = new Scan()
  .addColumn(Bytes.toBytes(familyName), Bytes.toBytes(colName))
  .setCaching(100)
  val rs = table.getScanner(scan)
  
  override def next() = {
    Try{
      val re = rs.next()
      if(re == null){
        rs.close
        table.close()
        conn.close
        null
      }
      val docID = Bytes.toString(re.getRow).toLong
      val values = Bytes.toString(re.value()).split(":")
      val idx = values(0).split(",").map { x => x.toInt }
      val prob = values(1).split(",").map { x => x.toDouble }
      val kv = idx.zip(prob).toMap
      (docID, kv)
    }
  }
}