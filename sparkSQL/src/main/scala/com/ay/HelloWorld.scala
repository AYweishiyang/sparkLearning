package com.ay

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object HelloWorld {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("HelloWorld").setMaster("local[*]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val df = spark.read.json("C:\\Users\\AY\\Downloads\\test.json")
    df.show()
    df.createTempView("test")
    spark.sql("select * from test").show()
    spark.close()
  }
}
