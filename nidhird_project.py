from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, to_date, unix_timestamp, split, explode, hour, dayofweek, when, mean, stddev, avg, count, dayofweek, month, datediff, expr
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DoubleType, IntegerType, LongType, FloatType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os
import sys
os.environ["SPARK_DRIVER_EXTRA_OPTS"] = "--illegal-access=warn"
os.environ["SPARK_EXECUTOR_EXTRA_OPTS"] = "--illegal-access=warn"

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("FlightPriceAnalysis") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

input_path = sys.argv[1]
output_path = sys.argv[2]
df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

"""### Preprocessing"""

# Select only the required columns
columns_to_keep = [
    "searchDate", "flightDate", "startingAirport", "destinationAirport",
    "isBasicEconomy", "isRefundable", "isNonStop", "baseFare", "totalFare", "seatsRemaining",
    "totalTravelDistance", "segmentsDepartureTimeEpochSeconds", "segmentsArrivalTimeEpochSeconds",
    "segmentsAirlineName", "segmentsCabinCode","segmentsDistance"
]
df = df.select(*columns_to_keep)

# Convert date columns to date format
df = df.withColumn("searchDate", to_date(col("searchDate"), "yyyy-MM-dd")) \
       .withColumn("flightDate", to_date(col("flightDate"), "yyyy-MM-dd"))

# Split and extract the first value from departure and arrival time columns
df = df.withColumn("segmentsDepartureTimeEpochSeconds", split(col("segmentsDepartureTimeEpochSeconds"), "\|\|")) \
       .withColumn("departure_time", col("segmentsDepartureTimeEpochSeconds").getItem(0)) \
       .withColumn("departure_time", from_unixtime(col("departure_time").cast("long")).cast("timestamp"))

df = df.withColumn("segmentsArrivalTimeEpochSeconds", split(col("segmentsArrivalTimeEpochSeconds"), "\|\|")) \
       .withColumn("arrival_time", col("segmentsArrivalTimeEpochSeconds").getItem(0)) \
       .withColumn("arrival_time", from_unixtime(col("arrival_time").cast("long")).cast("timestamp"))

df = df.drop("segmentsDepartureTimeEpochSeconds").drop("segmentsArrivalTimeEpochSeconds")

# Keep only the first value from the 'segmentsCabinCode' and 'segmentsAirlineName' columns
df = df.withColumn("cabinCode", split(col("segmentsCabinCode"), "\|\|").getItem(0))
df = df.withColumn("airlineName", split(col("segmentsAirlineName"), "\|\|").getItem(0))

# Drop original columns with multiple values
df = df.drop("segmentsCabinCode").drop("segmentsAirlineName")

from pyspark.sql.functions import col, sum as spark_sum

# Count the number of missing values in each column
missing_value_counts = df.select([(spark_sum(col(c).isNull().cast("int")).alias(c)) for c in df.columns])
missing_value_counts.show()

# Count the number of instances in the DataFrame
num_instances = df.count()
print(f"Number of instances in DataFrame: {num_instances}")

# Drop rows with missing values and fill remaining with mean
mean_values = df.agg(
    mean(col("totalTravelDistance")).alias("mean_totalTravelDistance"),
    mean(col("segmentsDistance")).alias("mean_segmentsDistance")
).collect()[0]

df = df.fillna({
    'totalTravelDistance': mean_values['mean_totalTravelDistance'],
    'segmentsDistance': mean_values['mean_segmentsDistance'],
}).dropna()

# Outlier detection and removal
stats = df.select(mean("totalFare").alias("mean"), stddev("totalFare").alias("stddev")).collect()[0]
mean_fare, stddev_fare = stats["mean"], stats["stddev"]
df = df.filter(~((col("totalFare") > mean_fare + 3 * stddev_fare) | (col("totalFare") < mean_fare - 3 * stddev_fare)))

# Count the number of instances in the DataFrame
num_instances = df.count()
print(f"Number of instances after outlier removal in DataFrame: {num_instances}")

from pyspark.sql.functions import col, count

# Count the number of rows before removing duplicates
original_count = df.count()
# Drop duplicate rows
df = df.dropDuplicates()
deduplicated_count = df.count()
duplicates = original_count - deduplicated_count
print(f"Number of duplicate rows: {duplicates}")
print(f"Number of instances after removing duplicates: {deduplicated_count}")

"""### EDA"""

from pyspark.sql.functions import expr

# Descriptive Statistics
df.describe().show()

from pyspark.sql.functions import col
import pandas as pd

# Identify numerical columns
numerical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, (DoubleType, IntegerType, LongType, FloatType))]

# Calculate Pearson's correlation coefficient with totalFare
correlation_results = {}
for feature in numerical_columns:
    correlation = df.stat.corr("totalFare", feature)
    correlation_results[feature] = correlation

correlation_df = pd.DataFrame(list(correlation_results.items()), columns=["Feature", "Correlation with totalFare"])
print(correlation_df)

# Trend Analysis of Average totalFare over months and weekdays to identify trends.
monthly_fare = df.groupBy(month("flightDate").alias("Month")).agg(avg("totalFare").alias("AvgFare")).orderBy("Month")
monthly_fare.show()

weekly_fare = df.groupBy(dayofweek("flightDate").alias("DayOfWeek")).agg(avg("totalFare").alias("AvgFare")).orderBy("DayOfWeek")
weekly_fare.show()

# Average totalFare by each Airline
airline_fare_distribution = df.groupBy("airlineName").agg(
    avg("totalFare").alias("AvgFare"),
    count("totalFare").alias("Count")
).orderBy("AvgFare", ascending=False)
airline_fare_distribution.show()

# Average totalFare by CabinCode
cabin_fare_distribution = df.groupBy("cabinCode").agg(
    avg("totalFare").alias("AvgFare"),
    count("totalFare").alias("Count")
).orderBy("AvgFare", ascending=False)
cabin_fare_distribution.show()

# Average fare for refundable vs. non-refundable flights.
fare_refundability = df.groupBy("isRefundable").agg(avg("totalFare").alias("AvgFare")).orderBy("isRefundable")
fare_refundability.show()

# Compare average fares for non-stop vs. layover flights.
fare_nonstop = df.groupBy("isNonStop").agg(avg("totalFare").alias("AvgFare")).orderBy("isNonStop")
fare_nonstop.show()

"""### Predicton Model"""

# Assemble features into a feature vector
feature_columns = ["isBasicEconomy", "isNonStop", "seatsRemaining", "totalTravelDistance"]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="feature_col")
df = assembler.transform(df)

# Split data into training and test sets
train_data, test_data = df.randomSplit([0.80, 0.20], seed=42)

# Define and train the Random Forest model
rf = RandomForestRegressor(featuresCol="feature_col", labelCol="totalFare")
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [10,20,30])
             .addGrid(rf.maxDepth, [5,10])
             .build())

evaluator = RegressionEvaluator(labelCol="totalFare", predictionCol="prediction", metricName="rmse")

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3,
                          parallelism=4)

cv_model = crossval.fit(train_data)

# Evaluate on test data
best_rf_model = cv_model.bestModel
test_predictions = best_rf_model.transform(test_data)
test_rmse = evaluator.evaluate(test_predictions)
print(f"Best Random Forest RMSE on test data: {test_rmse}")

evaluator_mae = RegressionEvaluator(labelCol="totalFare", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="totalFare", predictionCol="prediction", metricName="r2")

# Calculate evaluation metrics on test data
test_mae = evaluator_mae.evaluate(test_predictions)
test_r2 = evaluator_r2.evaluate(test_predictions)

print(f"Best Random Forest MAE on test data: {test_mae}")
print(f"Best Random Forest R² on test data: {test_r2}")

# Display comparison of actual vs. predicted fares
comparison_df = test_predictions.select("totalFare", "prediction")
comparison_df.show()

import pandas as pd

# Get feature importances
importances = best_rf_model.featureImportances
feature_names = feature_columns

# Create a DataFrame for feature importance
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances.toArray()
}).sort_values(by='Importance', ascending=False)

print(importances_df)

from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Convert categorical columns to numerical format
categorical_columns = ["airlineName"]
for col_name in categorical_columns:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name+"_index")
    df = indexer.fit(df).transform(df)
    encoder = OneHotEncoder(inputCols=[col_name+"_index"], outputCols=[col_name+"_onehot"])
    df = encoder.fit(df).transform(df)

# Assemble features into a feature vector
feature_columns = ["isBasicEconomy", "isRefundable", "isNonStop", "totalFare"] + [col_name+"_onehot" for col_name in categorical_columns]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

# Normalize the features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Train K-Means model
kmeans = KMeans(featuresCol="scaledFeatures", k=5, seed=42)
model = kmeans.fit(df)

# Make predictions
predictions = model.transform(df)

# Evaluate the clustering
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette score: {silhouette}")

# Analyze the clusters
predictions.groupBy("prediction").count().show()

# Show cluster centers
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

spark.stop()