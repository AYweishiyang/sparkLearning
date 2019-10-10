package com.ay.mml;

/**
 * @author ay
 * @create 2019-10-05 11:29
 */

// $example on$
import com.microsoft.ml.spark.lightgbm.LightGBMRegressor;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
// $example off$

public class LoadForecasting {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("javaMML")
//                .config("spark.master","local[*]")
//                .config("spark.eventLog.enabled", "false")
                .getOrCreate();
        Dataset<Row> df = spark
                .read()
                .format("csv")
                .option("header", "true")
                .option("multiLine", true)
                .option("inferSchema", true)
                .load("/full_features_shift.csv");
//                .load("file:///H:\\jupyter\\energy_forecasting_notebooks\\full_features_shift.csv");

        // $example on$
        // Load and parse the data file, converting it to a DataFrame.
//        Dataset<Row> data = spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");

//        df.show(10);
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"temp", "dew", "humi", "windspeed", "precip", "dow", "doy", "month", "hour", "minute", "windgust", "t_m24", "t_m48"})
//                .setInputCols(new String[]{"temp", "t_m48"})
                .setOutputCol("features");
        Dataset<Row> output = vectorAssembler.transform(df);
        output.show(10);

        Dataset<Row>[] splits = output.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        LightGBMRegressor lightGBMRegressor = new LightGBMRegressor()
                .setLabelCol("load")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{lightGBMRegressor});


        ParamMap[] paramGridBuilder = new ParamGridBuilder()
                .addGrid(lightGBMRegressor.maxDepth(), new int[]{3, 4, 5,6,7,8})
                .addGrid(lightGBMRegressor.numLeaves(), new int[]{16, 20, 31, 40, 60})
                .addGrid(lightGBMRegressor.learningRate(),new double[]{0.05,0.1})
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
                .setEvaluator(regressionEvaluator2)
                .setEvaluator(regressionEvaluator3)
                .setEvaluator(regressionEvaluator4)
                .setEstimatorParamMaps(paramGridBuilder)
                .setNumFolds(5) // Use 3+ in practice
                .setParallelism(10);// Evaluate up to 2 parameter settings in parallel
        long start = System.currentTimeMillis();
        CrossValidatorModel crossValidatorModel = crossValidator.fit(trainingData);
        long end = System.currentTimeMillis();


        Dataset<Row> prediction = crossValidatorModel.transform(testData);

        prediction.select("prediction", "load", "features").show(5);

        ParamMap paramMap = crossValidatorModel.paramMap();


        double rmse = regressionEvaluator1.evaluate(prediction);
        double mse = regressionEvaluator2.evaluate(prediction);
        double r2 = regressionEvaluator3.evaluate(prediction);
        double mae = regressionEvaluator4.evaluate(prediction);
        System.out.println("(rmse) on test data = " + rmse);
        System.out.println("(mse) on test data = " + mse);
        System.out.println("(r2) on test data = " + r2);
        System.out.println("(mae) on test data = " + mae);

        System.out.println("paramMap.toString()======" + paramMap.toString());
        System.out.println("train-time= " + (end-start)/1000.0 + " s");
        spark.stop();


    }
}
