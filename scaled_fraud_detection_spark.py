# scaled_fraud_detection_spark.py

"""
This script demonstrates a scalable fraud detection pipeline using PySpark and distributed XGBoost.
It covers:
1.  Simulated data generation (for demonstration purposes).
2.  Data ingestion into Spark DataFrames.
3.  Feature engineering and preprocessing using Spark MLlib.
4.  Distributed training of an XGBoost model.
5.  Model evaluation.

This pipeline is designed to be scalable to handle large datasets (billions of records)
by leveraging the distributed computing capabilities of Apache Spark.
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, hour, dayofweek, lit, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Import SparkXGBClassifier if available (requires xgboost>=1.6.0 with spark module)
# For older versions, you might use xgboost4j-spark or other wrappers.
# We will assume a recent enough xgboost version that includes spark integration.
from xgboost.spark import SparkXGBClassifier

def create_spark_session():
    """Creates and returns a Spark session."""
    spark = (
        SparkSession.builder.appName("ScalableFraudDetection")
        .config("spark.driver.memory", "2g") # Adjust as per sandbox limits
        .config("spark.executor.memory", "2g") # Adjust as per sandbox limits
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") # For better performance with pandas conversion if used
        .getOrCreate()
    )
    return spark

def generate_simulated_data(spark, num_rows=1000000, output_path="/home/ubuntu/ml_capstone_scaling/simulated_transactions.parquet"):
    """Generates simulated transaction data and saves it as Parquet."""
    print(f"Generating {num_rows} simulated transaction records...")

    # Define schema for the DataFrame
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("transaction_amount", DoubleType(), True),
        StructField("merchant_id", StringType(), True),
        StructField("location_country", StringType(), True),
        StructField("device_type", StringType(), True),
    ])

    # Create an empty DataFrame with the defined schema to ensure correct types
    # This is a bit of a workaround for direct data generation with specific types in PySpark
    # A more robust way for very large data would be to generate partitions in parallel
    
    # For demonstration, we'll create a smaller DataFrame and then scale up conceptually
    # Generating truly massive data in-memory first is not scalable, so we simulate by writing directly if possible
    # or generate in pandas then convert, but for Spark native, it's better to use Spark APIs

    # Let's use a simpler approach for generation that scales better within Spark itself
    df = spark.range(num_rows).withColumnRenamed("id", "_id")

    df = df.withColumn("transaction_id", (rand() * 1000000000).cast("int").cast("string")) \
           .withColumn("user_id", (rand() * 100000).cast("int").cast("string")) \
           .withColumn("timestamp", (lit(time.time()) - rand() * 3600*24*30).cast("timestamp")) \
           .withColumn("transaction_amount", (rand() * 1000 + 5).cast("double")) \
           .withColumn("merchant_id", (rand() * 1000).cast("int").cast("string")) \
           .withColumn("location_country", when(rand() < 0.3, "US").when(rand() < 0.6, "GB").otherwise("CA")) \
           .withColumn("device_type", when(rand() < 0.5, "mobile").when(rand() < 0.8, "desktop").otherwise("tablet"))

    # Simulate fraud label (highly imbalanced)
    # Approx 0.5% fraud for demonstration
    fraud_percentage = 0.005 
    df = df.withColumn("is_fraud", when(rand() < fraud_percentage, 1).otherwise(0).cast("integer"))
    
    df = df.drop("_id") # drop the temporary id column

    print("Generated data schema:")
    df.printSchema()
    print("Sample generated data:")
    df.show(5, truncate=False)
    print("Fraud distribution:")
    df.groupBy("is_fraud").count().show()

    df.write.mode("overwrite").parquet(output_path)
    print(f"Simulated data saved to {output_path}")
    return output_path

def main():
    spark = create_spark_session()
    
    # Create a directory for the project if it doesn't exist
    # This will be handled by shell command later

    # --- 1. Data Generation (Simulated) ---
    # In a real scenario, data would be in HDFS, S3, etc.
    # For this capstone, we generate a manageable size to demonstrate the pipeline.
    # The design scales to billions of records by Spark's nature.
    # Reducing num_rows for faster sandbox execution, but designed for millions/billions.
    data_path = generate_simulated_data(spark, num_rows=100000, output_path="/home/ubuntu/ml_capstone_scaling/simulated_transactions.parquet") 

    # --- 2. Data Ingestion ---
    print("\n--- Ingesting Data ---")
    raw_df = spark.read.parquet(data_path)
    raw_df.printSchema()
    raw_df.show(5, truncate=False)

    # --- 3. Feature Engineering and Preprocessing Pipeline ---
    print("\n--- Feature Engineering & Preprocessing ---")

    # Extract temporal features
    df_features = raw_df.withColumn("tx_hour", hour(col("timestamp"))) \
                        .withColumn("tx_dayofweek", dayofweek(col("timestamp")))

    # Define categorical and numerical columns
    # transaction_id, user_id, merchant_id are high cardinality; for a real system, more advanced embedding or hashing might be used.
    # For this demo, we'll select a few for one-hot encoding.
    categorical_cols = ["location_country", "device_type"]
    # merchant_id could be added if cardinality is managed or hashing is used.
    numerical_cols = ["transaction_amount", "tx_hour", "tx_dayofweek"]
    label_col = "is_fraud"

    # Create stages for the pipeline
    stages = []

    # StringIndexer and OneHotEncoder for categorical features
    for cat_col in categorical_cols:
        string_indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_index", handleInvalid="keep")
        one_hot_encoder = OneHotEncoder(inputCols=[string_indexer.getOutputCol()], outputCols=[cat_col + "_ohe"])
        stages += [string_indexer, one_hot_encoder]
    
    # StandardScaler for numerical features
    # Numerical features need to be assembled into a vector first for StandardScaler
    numerical_assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features_vec")
    stages.append(numerical_assembler)
    scaler = StandardScaler(inputCol="numerical_features_vec", outputCol="scaled_numerical_features")
    stages.append(scaler)

    # Assemble all processed features into a single feature vector
    feature_cols_for_assembly = [cat_col + "_ohe" for cat_col in categorical_cols] + ["scaled_numerical_features"]
    vector_assembler = VectorAssembler(inputCols=feature_cols_for_assembly, outputCol="features")
    stages.append(vector_assembler)

    # Create the preprocessing pipeline
    preprocessing_pipeline = Pipeline(stages=stages)
    
    # Fit the preprocessing pipeline and transform the data
    print("Fitting preprocessing pipeline...")
    preprocessing_model = preprocessing_pipeline.fit(df_features)
    processed_df = preprocessing_model.transform(df_features)

    print("Schema after preprocessing:")
    processed_df.printSchema()
    print("Sample of processed data (selected columns):")
    processed_df.select("features", label_col).show(5, truncate=False)

    # --- 4. Data Splitting ---
    print("\n--- Splitting Data ---")
    train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training data count: {train_df.count()}")
    print(f"Test data count: {test_df.count()}")

    # --- 5. Model Training (Distributed XGBoost) ---
    print("\n--- Training XGBoost Model ---")
    
    # Calculate scale_pos_weight for imbalanced data
    num_positives = train_df.filter(col(label_col) == 1).count()
    num_negatives = train_df.filter(col(label_col) == 0).count()
    scale_pos_weight_val = float(num_negatives) / num_positives if num_positives > 0 else 1.0
    print(f"Calculated scale_pos_weight: {scale_pos_weight_val}")

    xgb_classifier = SparkXGBClassifier(
        featuresCol="features",
        labelCol=label_col,
        eval_metric="aucpr", # Area Under Precision-Recall Curve, good for imbalanced data
        scale_pos_weight=scale_pos_weight_val,
        use_gpu=False, # Set to True if GPUs are available and configured in Spark
        # num_workers=2, # Example: set based on cluster resources. XGBoost Spark will try to infer if not set.
        seed=42
    )

    # Create the full pipeline including the XGBoost classifier
    # This is useful if you want to save/load the entire model including preprocessing
    # model_pipeline = Pipeline(stages=[preprocessing_model, xgb_classifier]) # This is incorrect, preprocessing_model is already fitted.
    # The correct way is to use the fitted preprocessing_model to transform, then train, or put unfitted stages in one pipeline.
    # For simplicity here, we use the already processed_df for training.

    print("Starting XGBoost training...")
    start_time = time.time()
    xgb_model = xgb_classifier.fit(train_df) # Train on preprocessed data
    end_time = time.time()
    print(f"XGBoost training completed in {end_time - start_time:.2f} seconds.")

    # --- 6. Model Evaluation ---
    print("\n--- Evaluating Model ---")
    predictions_df = xgb_model.transform(test_df)

    print("Sample predictions:")
    predictions_df.select("features", label_col, "rawPrediction", "probability", "prediction").show(5, truncate=False)

    evaluator_auprc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="probability", metricName="areaUnderPR")
    auprc = evaluator_auprc.evaluate(predictions_df.select(label_col, col("probability")))
    print(f"Area Under Precision-Recall Curve (AUPRC) on Test Data: {auprc:.4f}")

    evaluator_auroc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="probability", metricName="areaUnderROC")
    auroc = evaluator_auroc.evaluate(predictions_df.select(label_col, col("probability")))
    print(f"Area Under ROC Curve (AUROC) on Test Data: {auroc:.4f}")

    # For F1, Precision, Recall, we can use a confusion matrix approach or MulticlassClassificationEvaluator
    # For simplicity, focusing on AUPRC and AUROC here.
    # To get F1, Precision, Recall for the positive class (fraud=1):
    tp = predictions_df.filter((col(label_col) == 1) & (col("prediction") == 1.0)).count()
    fp = predictions_df.filter((col(label_col) == 0) & (col("prediction") == 1.0)).count()
    fn = predictions_df.filter((col(label_col) == 1) & (col("prediction") == 0.0)).count()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision for fraud class: {precision:.4f}")
    print(f"Recall for fraud class: {recall:.4f}")
    print(f"F1-score for fraud class: {f1_score:.4f}")

    # --- 7. Save Model and Pipeline (Conceptual) ---
    print("\n--- Saving Model (Conceptual) ---")
    model_output_path = "/home/ubuntu/ml_capstone_scaling/spark_xgb_fraud_model"
    # To save the whole pipeline (preprocessing + model):
    # First, create a pipeline with unfitted preprocessors and the classifier
    # full_pipeline_stages = stages + [xgb_classifier] # This xgb_classifier is unfitted
    # full_pipeline_to_fit = Pipeline(stages=full_pipeline_stages)
    # fitted_full_pipeline = full_pipeline_to_fit.fit(df_features) # Train on df_features (before preprocessing split)
    # fitted_full_pipeline.write().overwrite().save(model_output_path)
    # For this script, we'll just save the trained XGBoost model part:
    xgb_model.write().overwrite().save(model_output_path)
    print(f"Trained XGBoost model saved to {model_output_path}")
    
    # The preprocessing pipeline model can also be saved:
    preprocessing_model_path = "/home/ubuntu/ml_capstone_scaling/spark_preprocessing_model"
    preprocessing_model.write().overwrite().save(preprocessing_model_path)
    print(f"Preprocessing pipeline model saved to {preprocessing_model_path}")

    # --- 8. Stop Spark Session ---
    print("\n--- Stopping Spark Session ---")
    spark.stop()

if __name__ == "__main__":
    main()


