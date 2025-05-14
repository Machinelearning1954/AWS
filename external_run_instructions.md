# Running the Scalable Fraud Detection Spark Pipeline Externally

This document provides instructions for running the `scaled_fraud_detection_spark.py` script in an environment outside the current sandbox, where Java and Apache Spark can be properly configured. The script was developed for the Machine Learning Engineering Bootcamp Capstone project to demonstrate scaling a fraud detection model using PySpark and distributed XGBoost.

## 1. Sandbox Environment Limitation

The provided `scaled_fraud_detection_spark.py` script cannot be fully executed within the current sandbox environment due to the absence of a pre-configured Java Development Kit (JDK) and Apache Spark installation. PySpark, the Python library for Spark, requires a Java runtime to function, and attempts to initialize a Spark session failed with a `PySparkRuntimeError: [JAVA_GATEWAY_EXITED]` indicating that the Java gateway process could not be started.

While the Python code for the pipeline (data generation, preprocessing, distributed XGBoost training, and evaluation) has been written and is complete, its execution and validation require an environment with a working Spark setup.

## 2. Prerequisites for External Execution

To run the script successfully, you will need the following installed and configured on your local machine or a suitable cluster environment:

1.  **Java Development Kit (JDK):** Version 8 or 11 is recommended for compatibility with recent Spark versions. OpenJDK is a good choice.
    *   Verify installation: `java -version`
2.  **Apache Spark:** Version 3.0.0 or later. Download from the [official Apache Spark website](https://spark.apache.org/downloads.html).
    *   Ensure Spark binaries are extracted to a known location.
3.  **Python:** Version 3.8 or later.
    *   Verify installation: `python --version` or `python3 --version`
4.  **Python Pip:** For installing Python packages.
    *   Verify installation: `pip --version` or `pip3 --version`
5.  **Required Python Libraries:**
    *   `pyspark` (version matching your Spark installation, e.g., 3.5.0)
    *   `xgboost` (version >= 1.6.0, which includes Spark integration)
    *   `scikit-learn` (as a dependency for XGBoost Spark module)
    *   `numpy` (usually a dependency of the above)

## 3. Environment Configuration

Before running the script, ensure the following environment variables are set correctly. The exact method depends on your operating system (e.g., in `.bashrc`, `.zshrc`, or system environment variables on Windows).

1.  **`JAVA_HOME`**: Set this to the installation directory of your JDK.
    *   Example (Linux/macOS): `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`
    *   Example (Windows): Set via System Properties -> Environment Variables.
2.  **`SPARK_HOME`**: Set this to the directory where you extracted Apache Spark.
    *   Example (Linux/macOS): `export SPARK_HOME=/opt/spark` (or your chosen path)
3.  **`PATH`**: Add Spark's `bin` and `sbin` directories, and JDK's `bin` directory to your system `PATH`.
    *   Example (Linux/macOS): `export PATH=$PATH:$JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin`

## 4. Installing Python Dependencies

If you haven't already, install the required Python libraries using pip:

```bash
pip install pyspark==3.5.0 xgboost>=1.6.0 scikit-learn
```

Adjust `pyspark` version if your Spark installation differs, though 3.5.0 is a recent stable version.

## 5. Obtaining the Script

The script `scaled_fraud_detection_spark.py` should be provided to you. Save it to a directory on your machine, for example, `~/ml_capstone_project/`.

## 6. Running the Script

There are two main ways to run a PySpark script:

### Option A: Using `spark-submit` (Recommended for cluster or more complex setups)

`spark-submit` is the standard way to submit Spark applications.

Navigate to the directory where you saved the script and run:

```bash
spark-submit scaled_fraud_detection_spark.py
```

If you need to specify master URL (e.g., `local[*]` for local mode with all available cores) or other configurations:

```bash
spark-submit --master local[*] scaled_fraud_detection_spark.py
```

The script is configured internally with `appName("ScalableFraudDetection")` and basic memory settings (`spark.driver.memory`, `spark.executor.memory`), which `spark-submit` will use. You can override these via `spark-submit` arguments if needed.

### Option B: Running as a standard Python script (If Spark is configured for local mode and PySpark is in `PYTHONPATH`)

If your `SPARK_HOME` is set and PySpark is correctly installed in your Python environment, you might be able to run it directly:

```bash
python scaled_fraud_detection_spark.py
```

However, `spark-submit` is generally more robust for managing Spark applications.

## 7. Expected Output and Artifacts

Upon successful execution, the script will:

1.  Print status messages to the console for each stage: data generation, ingestion, preprocessing, training, and evaluation.
2.  **Generate Simulated Data:** It will create a Parquet file with simulated transaction data. The script is currently configured to save this at `/home/ubuntu/ml_capstone_scaling/simulated_transactions.parquet`. When run externally, this path will be relative to where Spark is running or might need adjustment in the script if you want it in a specific local directory (e.g., change to a relative path like `simulated_transactions.parquet` or an absolute path suitable for your system).
3.  **Train an XGBoost Model:** A distributed XGBoost model will be trained.
4.  **Print Evaluation Metrics:** It will output AUPRC, AUROC, Precision, Recall, and F1-score for the fraud class on the test set.
5.  **Save Models:**
    *   The trained XGBoost model will be saved to a directory (e.g., `spark_xgb_fraud_model` relative to the script's working directory or the path specified in the script).
    *   The preprocessing pipeline model will be saved to a directory (e.g., `spark_preprocessing_model`).

**Note on Paths:** The script uses absolute paths like `/home/ubuntu/ml_capstone_scaling/`. You might need to modify these paths within the `scaled_fraud_detection_spark.py` script to suitable locations on your system or ensure the script is run from a context where these paths make sense (e.g., if you create that directory structure).
A simple modification would be to change these to relative paths, e.g.:
`output_path="simulated_transactions.parquet"`
`model_output_path="spark_xgb_fraud_model"`
`preprocessing_model_path="spark_preprocessing_model"`

This will ensure files are created in the directory from which you run `spark-submit` or the Python script.

## 8. Troubleshooting

*   **`JAVA_HOME` not found:** Double-check that `JAVA_HOME` is set correctly and points to a valid JDK installation.
*   **PySpark errors:** Ensure your `pyspark` library version is compatible with your Apache Spark installation.
*   **Memory issues (e.g., `OutOfMemoryError`):** For larger `num_rows` in data generation or more complex operations, you might need to increase Spark driver and executor memory. This can be done via `spark-submit` options (e.g., `--driver-memory 4g --executor-memory 4g`) or by modifying the `.config()` calls in the `create_spark_session()` function within the script.

By following these instructions, you should be able to execute the scalable fraud detection pipeline in a suitable environment and observe its functionality.
