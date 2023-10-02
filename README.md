# anomaly_detection_batteryfielddata
Framework for aging anomaly detection in battery field data using statistical learning. The framework encompasses clustering of SOH values and the fitting of empirical data with various probability density functions.  

## Description

The Python code includes the following components:

- **fit_distributions.py**: Functions to fit various probability distributions to a given battery field dataset and calculate their corresponding probability density functions (pdfs) and cumulative distribution functions (cdfs).
- **load_data.py**: Loading and preprocessing field data from a CSV file located in the data directory.
