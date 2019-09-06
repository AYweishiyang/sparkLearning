package com.ay
import org.apache.spark
import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.SparkSession

object MyKeans {
  val conf = new SparkConf().setAppName("HelloWorld").setMaster("local[*]")
  //    val conf = new SparkConf().setAppName("HelloWorld")
  val spark = SparkSession.builder().config(conf).getOrCreate()

  // Loads data.
  val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

  // Trains a k-means model.
  val kmeans = new KMeans().setK(2).setSeed(1L)
  val model = kmeans.fit(dataset)

  // Make predictions
  val predictions = model.transform(dataset)

  // Evaluate clustering by computing Silhouette score
  val evaluator = new ClusteringEvaluator()

  val silhouette = evaluator.evaluate(predictions)
  println(s"Silhouette with squared euclidean distance = $silhouette")

  // Shows the result.
  println("Cluster Centers: ")
  model.clusterCenters.foreach(println)
}
