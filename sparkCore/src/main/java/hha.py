
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from mmlspark.lightgbm import LightGBMRegressor
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline


from mmlspark.lightgbm import LightGBMRegressionModel
from mmlspark.train import ComputeModelStatistics
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from mmlspark.lightgbm import LightGBMRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
spark = SparkSession.builder.appName("load-forecasting").getOrCreate()

triazines = spark.read.format("csv").option("header","true").option("inferSchema", "true").load("/full_features_shift.csv")


assembler = VectorAssembler(inputCols=["temp", "dew", "humi", "windspeed", "precip", "dow", "doy", "month", "hour","minute", "windgust", "t_m24", "t_m48","value_shift","condition","wind"],outputCol="features")

data = assembler.transform(triazines)

# print("records read: " + str(triazines.count()))
# print("Schema: ")
# triazines.printSchema()
# triazines.limit(10).toPandas()



train, test = data.randomSplit([0.85, 0.15], seed=1)




lgbm = LightGBMRegressor().setLabelCol("load").setFeaturesCol("features")
lgbm.g
pipeline = Pipeline(stages=[lgbm])

paramGrid = ParamGridBuilder() \
    .addGrid(lgbm.numLeaves, [20, 31, 50]) \
    .addGrid(lgbm.maxDepth, [3, 4]) \
    .addGrid(lgbm.learningRate, [0.1, 0.05]) \
    .build()

evaluator = RegressionEvaluator(
    labelCol="load", predictionCol="prediction", metricName="rmse")


crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)  # use 3+ folds in practice
cvModel = crossval.fit(train)
prediction = cvModel.transform(test)
selected = prediction.select("timestamp","load", "prediction")
print(selected)
rmse = evaluator.evaluate(prediction)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# print(model.getFeatureImportances())
# scoredData = model.transform(test)
# scoredData.limit(10).toPandas()
# metrics = ComputeModelStatistics(evaluationMetric='regression',labelCol='load',scoresCol='prediction').transform(scoredData)
# metrics.toPandas().show()

