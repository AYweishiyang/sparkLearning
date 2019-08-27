package com.ay


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.feature.VectorAssembler

//case class Features(timestamp: String, load:Double, weathertime:String, temp:Int, dew : Int, humi:Int, wind:String, windspeed:Int, windgust:Int, precip:Double, condition:String, dow:Int, doy:Int, day:Int, month:Int, hour:Int, minute:Int, t_m24:Double, t_m48:Double, tdif:Double)

object HelloWorld {

  def main(args: Array[String]): Unit = {

//    val conf = new SparkConf().setAppName("HelloWorld").setMaster("local[*]")
    val conf = new SparkConf().setAppName("HelloWorld")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val df = spark
      .read
      .format("csv")
      .option("header","true")
      .option("multiLine", true)
      .option("inferSchema", true)
      .load("/full_features.csv")
//      .load("file:///D:\\jupyter\\my_kaggle-master\\origin\\lecture05\\energy_forecasting_notebooks\\energy_forecasting_notebooks\\full_features.csv")

//    import spark.implicits._
//    val ds = df.as[Features]
//    ds.describe("temp" , "load").show()
//    df.filter("timestamp > '2019-07-31 00:00:00'").show(1000)
//    val features = ds("temp")

//    ds.
//    val df = spark.read.json("C:\\Users\\AY\\Downloads\\test.json")
//    df.show()
//    df.createTempView("test")
//
    val assembler = new VectorAssembler()
      .setInputCols(Array("temp", "dew", "humi", "windspeed", "precip", "dow", "doy", "month", "hour","minute", "windgust", "t_m24", "t_m48"))
      .setOutputCol("features")

    val output = assembler.transform(df)
//    val df1 = spark.sql("select * from test where timestamp > '2019-07-31 00:00:00'").show()
//    val df1 = spark.sql("select load from test")
//    df1.show()

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(output)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = output.randomSplit(Array(0.7, 0.3))

    // Train a GBT model.
    val gbt = new GBTRegressor()
      .setLabelCol("load")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    // Chain indexer and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, gbt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "load", "features").show(50)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("load")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val evaluator1 = new RegressionEvaluator()
      .setLabelCol("load")
      .setPredictionCol("prediction")
      .setMetricName("mae")
    val rmse = evaluator.evaluate(predictions)
    val mae = evaluator1.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    printf("map = %f",mae)
    val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
    println(s"Learned regression GBT model:\n ${gbtModel.toDebugString}")
    spark.close()
  }
}
