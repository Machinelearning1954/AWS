# Scalable Fraud Detection ML Pipeline - Capstone Project

This repository contains the work for Step 8 of the Machine Learning Engineering Bootcamp Capstone project: **Scale Your Prototype with Large-Scale Data**. The project focuses on scaling a machine learning prototype for fraud detection to handle large volumes of data, conceptually up to billions of records, using Apache Spark and distributed XGBoost.

## Project Overview

The primary goal was to take a conceptual fraud detection prototype (assumed to be an XGBoost model developed with Python, pandas, and scikit-learn for smaller datasets) and re-engineer it into a scalable pipeline capable of processing and learning from web-scale data. This involved:

*   **Analyzing Scalability Challenges:** Identifying bottlenecks in single-node ML workflows when dealing with massive datasets.
*   **Designing a Distributed Pipeline:** Architecting a solution using Apache Spark for data ingestion, distributed preprocessing, and feature engineering.
*   **Implementing Distributed Model Training:** Utilizing distributed XGBoost (via `xgboost.spark.SparkXGBClassifier`) for training the fraud detection model on Spark DataFrames.
*   **Addressing Trade-offs:** Considering and documenting the trade-offs involved in choosing distributed computing tools and techniques.
*   **Comprehensive Documentation:** Providing detailed explanations of the design, implementation, and how to run the pipeline.

## Repository Structure

```
/
├── scaled_fraud_detection_spark.py       # The core PySpark script for the scalable pipeline.
├── scaled_fraud_detection_pipeline.ipynb # Jupyter Notebook documenting the pipeline, code, and explanations.
├── conceptual_prototype_fraud_detection.md # Document describing the initial conceptual prototype.
├── scalability_analysis_fraud_detection.md # Document analyzing data sources and scalability requirements.
├── scalable_pipeline_design_fraud_detection.md # Document detailing the design of the scalable ML pipeline.
├── external_run_instructions.md          # IMPORTANT: Instructions for running the Spark pipeline externally.
├── requirements.txt                        # Python dependencies for running the Spark script.
├── todo.md                                 # Task checklist used during development.
└── README.md                               # This file.
```

## Key Deliverables and Components

1.  **`scaled_fraud_detection_spark.py`**: This Python script contains the full implementation of the scalable fraud detection pipeline using PySpark. It includes simulated data generation, feature engineering, preprocessing, distributed XGBoost model training, and evaluation.
2.  **`scaled_fraud_detection_pipeline.ipynb`**: A Jupyter Notebook that provides a step-by-step walkthrough of the pipeline. It includes all code from the Python script, detailed explanations for each stage, discussions on scalability, and how the design handles web-scale data. **Crucially, it notes that Spark-dependent code cells cannot be run directly in the provided sandbox environment.**
3.  **Supporting Markdown Documents:**
    *   `conceptual_prototype_fraud_detection.md`: Outlines the baseline single-node prototype.
    *   `scalability_analysis_fraud_detection.md`: Discusses data characteristics, bottlenecks of the prototype, and trade-offs in scaling ML algorithms.
    *   `scalable_pipeline_design_fraud_detection.md`: Details the architecture and design choices for the distributed pipeline.
4.  **`external_run_instructions.md`**: **This is a critical document.** Due to limitations in the development sandbox (lack of a pre-configured Java/Spark environment), the PySpark script and notebook cells requiring Spark cannot be executed directly. This file provides comprehensive instructions on how to set up a suitable external environment (e.g., local machine or cloud cluster) with Java, Spark, and Python dependencies to run the pipeline successfully.
5.  **`requirements.txt`**: Lists the Python packages required to run the `scaled_fraud_detection_spark.py` script in an external environment.

## Meeting Capstone Criteria (Step 8)

*   **Code Updated to GitHub (Conceptual):** All code and documentation are prepared for a GitHub repository.
*   **Understanding of Scaling:** The design documents (`scalability_analysis_fraud_detection.md`, `scalable_pipeline_design_fraud_detection.md`) and the Jupyter Notebook (`scaled_fraud_detection_pipeline.ipynb`) demonstrate a thorough understanding of how to scale an ML model, the challenges involved, and the trade-offs made.
*   **Handling Complete/Real-World Data:** The pipeline is designed using Apache Spark, which is inherently capable of handling datasets far exceeding typical single-machine memory limits, up to billions of records as required for web-scale applications. The data generation component in the script can be adjusted to simulate such volumes in an appropriate environment.
*   **Well-Thought-Out Decisions:**
    *   **Choice of Tools/Libraries:** Apache Spark was chosen for its robust distributed data processing capabilities. Distributed XGBoost (via `xgboost.spark`) was selected for its state-of-the-art performance and scalability for tree-based models. Parquet is recommended for data storage due to its efficiency.
    *   **Choice of ML/DL Technique:** XGBoost is a powerful and widely used algorithm for fraud detection. The pipeline incorporates strategies to handle class imbalance (e.g., `scale_pos_weight`), a common characteristic of fraud datasets.
*   **Well-Documented Repository and Code:** The Jupyter Notebook provides step-by-step documentation. The Python script is commented, and supplementary Markdown files explain the design and analysis process in detail. This README provides a clear overview.

## Excellence Criteria

*   **Designed for Web-Scale Data:** The architecture explicitly uses Apache Spark and distributed training techniques, which are standard for handling billions of data points.
*   **Clean and Elegant Code (Conceptual):** The Python script and notebook aim for clarity and modularity, following good coding practices for a data science project.

## How to Run

**Due to sandbox limitations, please refer to `external_run_instructions.md` for detailed steps on setting up your environment and running the `scaled_fraud_detection_spark.py` script.**

Briefly, you will need:
1.  Java Development Kit (JDK)
2.  Apache Spark
3.  Python and Pip
4.  The Python libraries listed in `requirements.txt`.

Once the environment is set up, you can typically run the script using `spark-submit`:

```bash
spark-submit scaled_fraud_detection_spark.py
```

Remember to adjust any file paths within the script if you are not running it from the `/home/ubuntu/ml_capstone_scaling/` directory structure or if you wish to save outputs elsewhere.

