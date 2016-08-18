package com.mad.io

import com.mad.models.Document
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.{ StorageLevel }
import com.mad.util._
import org.apache.hadoop.hbase.client.{ Scan, Result }
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.{ TableName }
import scala.util.Try

/**
 *
 */
class LoadRDDFromHBase(
    tableName: String,
    familyName: String,
    colName: String
)(implicit sc: SparkContext) extends LoadRDD {

  /**
   *
   */
  override def load(): Try[RDD[Document]] = {
    Try {
      val hbaseContext = Utils.getHBaseContext(sc)
      val scan = new Scan()
        .addColumn(Bytes.toBytes(familyName), Bytes.toBytes(colName))
        .setCaching(100)
      val rdd = hbaseContext.hbaseRDD(TableName.valueOf(tableName), scan)
        .map(pair => {
          val doc = Bytes.toString(pair._2.value()).split(",")
          val id = doc(0).split(":").map { x => x.toLong }
          val cnt = doc(1).split(":").map { x => x.toLong }
          Document(id, cnt)
        })
      rdd.setName("corpus")
      rdd.persist(StorageLevel.MEMORY_AND_DISK)

      //Utils.cleanCheckPoint(true)
      //rdd.checkpoint()

      rdd
    }
  }
}