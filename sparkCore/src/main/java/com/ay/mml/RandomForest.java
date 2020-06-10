package com.ay.mml;

import com.microsoft.ml.spark.lightgbm.LightGBMRegressor;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * @author ay
 * @create 2020-06-08 8:39
 */
public class RandomForest {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("RF")
//                .config("spark.executor.cores",4)
//                .config("spark.executor.instances", 3)
//                .config("spark.executor.memory", "3g")
//                .config("spark.default.parallelism", 100)
                .config("spark.master", "local[*]")
                .config("spark.eventLog.enabled", "false")
                .getOrCreate();
        Dataset<Row> df = spark
                .read()
                .format("csv")
                .option("header", "true")
                .option("multiLine", true)
                .option("inferSchema", true)
//                .load("/full_features_shift.csv");
                .load("file:///E:\\mi\\jupyter\\energy_forecasting_notebooks\\final-data.csv");


        VectorAssembler vectorAssembler = new VectorAssembler()
//                .setInputCols(new String[]{"temp", "dew", "humi", "windspeed", "precip", "dow", "doy", "month", "hour", "minute", "windgust", "t_m24", "t_m48"})
                .setInputCols(new String[]{"temp", "dew", "humi", "windspeed", "precip", "dow", "doy", "month", "hour", "minute", "windgust", "t_m24", "t_m48"})
//                .setInputCols(new String[]{"temp", "t_m48"})
                .setOutputCol("features");
        Dataset<Row> output = vectorAssembler.transform(df);


//        Dataset<Row>[] splits = output.randomSplit(new double[]{0.7, 0.3});
//        Dataset<Row> trainingData = splits[0];
//        Dataset<Row> testData = splits[1];

        output.createOrReplaceTempView("all");
        Dataset<Row> trainingData = spark.sql("select * from all where (timestamp < '2019-06-01 00:00:00')");
        Dataset<Row> testData = spark.sql("select * from all where (timestamp >= '2019-06-01 00:00:00') and (timestamp < '2019-07-01 00:00:00')");

        output.printSchema();


        //Dataset<Row> actual = testData.filter(testData.col("timestamp"));
//        LightGBMRegressor lightGBMRegressor = new LightGBMRegressor()
//                .setLabelCol("load")
//                .setFeaturesCol("features");
        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("load")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{rf});


        ParamMap[] paramGridBuilder = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[]{60,80})
//                .addGrid(rf.maxDepth(), new int[]{6,8,10,15})
                .build();


        RegressionEvaluator regressionEvaluator1 = new RegressionEvaluator()
                .setLabelCol("load")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        RegressionEvaluator regressionEvaluator2 = new RegressionEvaluator()
                .setLabelCol("load")
                .setPredictionCol("prediction")
                .setMetricName("mse");
        RegressionEvaluator regressionEvaluator3 = new RegressionEvaluator()
                .setLabelCol("load")
                .setPredictionCol("prediction")
                .setMetricName("r2");
        RegressionEvaluator regressionEvaluator4 = new RegressionEvaluator()
                .setLabelCol("load")
                .setPredictionCol("prediction")
                .setMetricName("mae");


        // Train model. This also runs the indexer.

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(regressionEvaluator1)
//                .setEvaluator(regressionEvaluator2)
//                .setEvaluator(regressionEvaluator3)
//                .setEvaluator(regressionEvaluator4)
                .setEstimatorParamMaps(paramGridBuilder)
                .setNumFolds(3) // Use 3+ in practice
                .setParallelism(12);// Evaluate up to 2 parameter settings in parallel
        long start = System.currentTimeMillis();
        CrossValidatorModel crossValidatorModel = crossValidator.fit(trainingData);
        long end = System.currentTimeMillis();
        Dataset<Row> prediction = crossValidatorModel.transform(testData);

        prediction.select( "prediction", "load", "features").show(1000);


        double rmse = regressionEvaluator1.evaluate(prediction);
        double mse = regressionEvaluator2.evaluate(prediction);
        double r2 = regressionEvaluator3.evaluate(prediction);
        double mae = regressionEvaluator4.evaluate(prediction);
        System.out.println("(rmse) on test data = " + rmse);
        System.out.println("(mse) on test data = " + mse);
        System.out.println("(r2) on test data = " + r2);
        System.out.println("(mae) on test data = " + mae);


        PipelineModel bestModel = (PipelineModel) crossValidatorModel.bestModel();
        Transformer[] stages = ((PipelineModel) crossValidatorModel.bestModel()).stages();

        System.out.println("crossValidatorModel.params().toString()==" + crossValidatorModel.params().toString());

        System.out.println("bestModel.stages().length=====" + bestModel.stages()[0].extractParamMap().toString());


        System.out.println("train-time= " + (end - start) / 1000.0 + " s");
//        try {
//            Thread.sleep(1000000);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
        spark.stop();

    }
}
