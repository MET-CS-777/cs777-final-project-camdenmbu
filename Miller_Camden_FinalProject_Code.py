# Camden Miller
# CS 777
# Final Project

# Import Libraries
from __future__ import print_function
import re
import sys
import numpy as np
import time
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import ChiSqSelector
from pyspark.sql.functions import when, col

# Metrics function
def m_metrics_l(ml_model,test_data):
    predictions = ml_model.transform(test_data).cache()
    predictionAndLabels = predictions.select("label","prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
    
    # Print some predictions vs labels
    # print(predictionAndLabels.take(10))
    metrics = MulticlassMetrics(predictionAndLabels)
    
    # Overall statistics
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    print(f"Precision = {precision:.4f} Recall = {recall:.4f} F1 Score = {f1Score:.4f}")
    print("Confusion matrix \n", metrics.confusionMatrix().toArray().astype(int))

# I used the same script and models to evaluate both datasets
if __name__ == "__main__":

    # Initialize Spark session
    spark = SparkSession.builder.appName("Final Project").getOrCreate()

    # Read in data and extract labels + features before preparing training and test data splits
    data = spark.read.csv(sys.argv[1], header=True, inferSchema=True)
    # Change "Backcourt" to "PG" to evaluate second dataset
    data = data.withColumn("label", when(col("label") == "PG", 1.0).otherwise(0.0))
    feature_columns = data.columns
    feature_columns.remove("label")
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    data_transformed = assembler.transform(data).select("features", "label").cache()
    train_data, test_data = data_transformed.randomSplit([0.8, 0.2])

    ### Logistic Regression
    classifier_logReg = LogisticRegression(maxIter=100, regParam=0.0001, featuresCol="features", labelCol="label")
    pipeline_logReg = Pipeline(stages=[classifier_logReg])

    start = time.time()
    model_logReg = pipeline_logReg.fit(train_data)
    print(f"Logistic Regression Model created in {time.time() - start:.2f}s.")
    print("\nPerformance Metrics: Logistic Regression")
    m_metrics_l(model_logReg, test_data)
    print(f"Total time {time.time() - start:.2f}s.")

    ### SVM
    classifier_SVM = LinearSVC(maxIter=100, regParam=0.0001, featuresCol="features", labelCol="label")
    pipeline_SVM = Pipeline(stages=[classifier_SVM])

    start = time.time()
    model_SVM = pipeline_SVM.fit(train_data)
    print(f"SVM Model created in {time.time() - start:.2f}s.")
    print("\nPerformance Metrics: SVM")
    m_metrics_l(model_SVM, test_data)
    print(f"Total time {time.time() - start:.2f}s.")

    spark.stop()