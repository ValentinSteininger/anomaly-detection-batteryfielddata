import numpy as np
import os
import pandas as pd


def load_data(filename):
    if not filename.endswith('.csv'):
        filename += '.csv'

    path_data = os.path.join('.', 'data', filename)
    pdf = pd.read_csv(path_data)

    cols = pdf.columns
    for c, dtype in enumerate(pdf.dtypes):
        if dtype == 'O':
            pdf[cols[c]] = pdf[cols[c]].apply(lambda x: eval(x.replace('=', ':').replace('''"''', '')))

    return pdf


def get_values_from_dict(soh_dict):
    vals_soh = []
    for val, count in soh_dict.items():
        vals_soh.extend([val] * count)
    return np.array(vals_soh)[:, np.newaxis]


def calc_relative_outliers(features, conf=95):
    """Detect outlier based on the selected distribution and confidence interval"""
    pdf = load_data('soh_countdict_cluster_select')

    def cond_ok(x):
        vals_soh = np.array(list(x['soh_%'].keys()))
        vals_dsoh = 100 - vals_soh
        lims = x[f'lims_{conf}%']
        return vals_soh[np.logical_and(vals_dsoh >= np.floor(lims[0]), vals_dsoh < np.ceil(lims[1]))]

    def cond_not_ok(x):
        vals_soh = np.array(list(x['soh_%'].keys()))
        vals_dsoh = 100 - vals_soh
        lims = x[f'lims_{conf}%']
        return vals_soh[np.logical_or(vals_dsoh < np.floor(lims[0]), vals_dsoh >= np.ceil(lims[1]))]

    rel_outlier = {}
    sum_datapoints = {}

    for c, feature in enumerate(features):

        pdf = pdf.sort_values(by=[feature]).reset_index(drop=True)
        rel_outlier[feature] = []
        sum_datapoints[feature] = []

        feature_values = pdf[feature].unique()

        for a in feature_values:

            indices = [i for i, val in enumerate(pdf[feature]) if val == a]

            count_ok = 0
            count_not_ok = 0
            for b in indices:
                soh_ok = cond_ok(pdf.iloc[b])
                soh_not_ok = cond_not_ok(pdf.iloc[b])

                count_ok = count_ok + sum(pdf.iloc[b]['soh_%'].get(int(value), 0) for value in soh_ok)
                count_not_ok = count_not_ok + sum(pdf.iloc[b]['soh_%'].get(int(value), 0) for value in soh_not_ok)

            rel_outlier[feature].append(count_not_ok/(count_ok+count_not_ok))
            sum_datapoints[feature].append(count_ok+count_not_ok)

    return rel_outlier, sum_datapoints


