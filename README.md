# anomaly_detection_batteryfielddata
Framework for aging anomaly detection in battery field data using statistical learning. The framework encompasses clustering of SOH values and the fitting of empirical data with various probability density functions.  

## Description

The Python code includes the following components:

- **cluster_spark_data.py**: Preprocess aging data using PySpark session. Cluster data based on selected feature variables and their stepsize. Moreover, calculate upper and lower limits of confidence interval based on selected distribution function.  
- **fit_distributions.py**: Functions to fit various probability distributions to a given battery field dataset and calculate their corresponding probability density functions (pdfs) and cumulative distribution functions (cdfs). Perform residual calculation and Kolmogorov-Smirnov test.
- **detect_outliers.py**: Identify outlier datapoints within prepared data from selected distribution function.
