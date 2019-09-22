package com.ay

import com.microsoft.ml.spark.lightgbm.{LightGBMClassifier, LightGBMRegressor}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import java.util.Date

object MyMML {
  def main(args: Array[String]): Unit = {
    val start_time = new Date().getTime
//    val conf = new SparkConf().setAppName("HelloWorld").setMaster("local[*]")
    val conf = new SparkConf().setAppName("MML")
    val spark = SparkSession.builder().config(conf).getOrCreate()//config("spark.executor.cores",4)
    val df = spark
      .read
      .format("csv")
      .option("header", "true")
      .option("multiLine", true)
      .option("inferSchema", true)
      .load("/full_features_shift.csv")
//      .load("file:///H:\\jupyter\\energy_forecasting_notebooks\\full_features_shift.csv")

    val assembler = new VectorAssembler()
      .setInputCols(Array("temp", "dew", "humi", "windspeed", "precip", "dow", "doy", "month", "hour", "minute", "windgust", "t_m24", "t_m48"))
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
    val lgbm = new LightGBMRegressor()
      .setLabelCol("load")
      .setFeaturesCol("indexedFeatures")


    //    val gbt = new GBTRegressor()
    //      .setLabelCol("load")
    //      .setFeaturesCol("indexedFeatures")
    //      .setMaxIter(200)

    // Chain indexer and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, lgbm))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lgbm.maxDepth, Array(3, 4,5))
      .addGrid(lgbm.numLeaves, Array(16,20, 31,40,60))
      .build()

    val evaluator = new RegressionEvaluator()
      .setLabelCol("load")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val evaluator1 = new RegressionEvaluator()
      .setLabelCol("load")
      .setPredictionCol("prediction")
      .setMetricName("mae")
    // Train model. This also runs the indexer.
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEvaluator(evaluator1)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5) // Use 3+ in practice
      .setParallelism(10) // Evaluate up to 2 parameter settings in parallel
    val start_train_time = new Date().getTime

    val cvModel = cv.fit(trainingData)
    val end_train_time = new Date().getTime
    //
    println("best--params" + cvModel.bestModel.params)
    //      val model = pipeline.fit(trainingData)
    val predictions = cvModel.transform(testData)
    // Make predictions.
    //    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "load", "features").show(50)

    // Select (prediction, true label) and compute test error.
    //cvModel.bestModel.params.mkString
    //    val evaluator1 = new RegressionEvaluator()
    //      .setLabelCol("load")
    //      .setPredictionCol("prediction")
    //      .setMetricName("mae")
    val rmse = evaluator.evaluate(predictions)
    val mae = evaluator1.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    printf("map = %f\n", mae)
    //    val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
    //    println(s"Learned regression GBT model:\n ${gbtModel.toDebugString}")
    //    Thread.sleep(1000000)
    val end_time = new Date().getTime
    println("totle time : " + (end_time - start_time)) //单位毫秒 }}
    println("train_time : " + (end_train_time - start_train_time)) //单位毫秒 }}
    println(cvModel.bestModel.params.toString)
  }

}
