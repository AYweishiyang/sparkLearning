package com.ay

import org.apache.spark.{SparkConf, SparkContext}

object WordCount {
  def main(args: Array[String]): Unit = {
    //配置
    val conf = new SparkConf().setAppName("wc").setMaster("local[*]")
    //创建sparkcontext
    val sc = new SparkContext(conf)


    val textFile = sc.textFile("D:\\Word.txt")
    val word = textFile.flatMap(_.split(" "))
    val k2v = word.map((_, 1)).reduceByKey(_+_, 1).saveAsTextFile("D:\\out")


    sc.stop()

  }
}
