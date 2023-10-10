import numpy as np
import math
import pandas as pd
import os

import scipy.stats as stats

from sklearn.neighbors import KernelDensity
from scipy import special
from scipy.special import beta as beta_func


"""
The following functions fit distributions on the clustered data. Therefore, csv files resulting from the 
'cluster_spark_data.py' scripts are read.
"""

def load_data(filename):
    """Load aging data from prepared csv file."""

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
    """Transform count dict into array"""

    vals_soh = []
    for val, count in soh_dict.items():
        vals_soh.extend([val] * count)
    return np.array(vals_soh)[:, np.newaxis]


class Normal:
    @staticmethod
    def pdf(mu, var):
        f = lambda x: 1 / math.sqrt(2 * math.pi * var) * np.exp(-0.5 * pow(x - mu, 2) / var)
        return f

    @staticmethod
    def cdf(x, mu, std):
        return stats.norm.cdf(x, mu, std)

    @staticmethod
    def calc_pars(arr_values):
        vals_soh = arr_values
        mu = np.mean(vals_soh)
        var = np.var(vals_soh)
        f = Normal.pdf(mu, var)
        x = np.linspace(0, 100, 1000)[:, np.newaxis]
        y = f(x)

        idx_filt = y > 10 ** (-9)
        x_filt = x[idx_filt]
        y_filt = y[idx_filt]
        std = var ** 0.5

        return x_filt.reshape(-1).tolist(), y_filt.reshape(-1).tolist(), mu, std


class LogNormal:
    @staticmethod
    def pdf(mu, var):
        sig = var ** .5
        f = lambda x: 1 / (math.sqrt(2 * math.pi) * sig * x) * np.exp(-0.5 * pow(np.log(x) - mu, 2) / var)
        return f

    @staticmethod
    def cdf(x, mu, std):
        f = 0.5 * (1 + special.erf((np.log(x) - mu) / (std * np.sqrt(2))))
        return f

    @staticmethod
    def calc_pars(arr_values):
        arr_values[arr_values == 0] = 0.1

        vals_soh = arr_values
        vals_soh_ln = np.log(vals_soh)

        mu = np.mean(vals_soh_ln)
        var = np.var(vals_soh_ln)
        f = LogNormal.pdf(mu, var)
        x = np.linspace(0.1, 100, 1000)[:, np.newaxis]
        y = f(x)
        std = var ** 0.5

        idx_filt = (y > 10 ** (-9)) & (y <= 1)
        x_filt = x[idx_filt]
        y_filt = y[idx_filt]

        return x_filt.reshape(-1).tolist(), y_filt.reshape(-1).tolist(), mu, std


class Weibull:
    @staticmethod
    def pdf(lmbda, k):
        f = lambda x: k / lmbda * pow(x / lmbda, k - 1) * np.exp(-pow(x / lmbda, k))

        return f

    @staticmethod
    def cdf(x, c, lmbda):
        f = 1 - np.exp(-(x / lmbda) ** c)
        return f

    @staticmethod
    def calc_pars(arr_values):
        arr_values[arr_values == 0] = 0.1
        vals_soh = arr_values

        x = np.linspace(0.1, 100, 1000)[:, np.newaxis]
        c, log, scale = stats.weibull_min.fit(vals_soh, floc=0)

        f = Weibull.pdf(scale, c)
        y = f(x)

        idx_filt = (y > 10 ** (-9)) & (y <= 1)
        x_filt = x[idx_filt]
        y_filt = y[idx_filt]
        return x_filt.reshape(-1).tolist(), y_filt.reshape(-1).tolist(), c, scale


class Exponential:
    @staticmethod
    def pdf(lmbda):
        f = lambda x: lmbda * np.exp(-lmbda * x)
        return f

    @staticmethod
    def cdf(x, lamda):
        f = 1 - np.exp(-np.array(lamda) * np.array(x))
        return f

    @staticmethod
    def calc_pars(arr_values):
        vals_soh = arr_values

        lmbda = 1 / np.mean(vals_soh)
        f = Exponential.pdf(lmbda)
        x = np.linspace(0, 100, 1000)[:, np.newaxis]
        y = f(x)

        idx_filt = y > 10 ** (-9)
        x_filt = x[idx_filt]
        y_filt = y[idx_filt]

        return x_filt.reshape(-1).tolist(), y_filt.reshape(-1).tolist(), lmbda


class Logistic:
    @staticmethod
    def pdf(mu, scale):
        f = lambda x: (1 / scale) * np.exp(-(x - mu) / scale) / (1 + np.exp(-(x - mu) / scale)) ** 2
        return f

    @staticmethod
    def cdf(x, mu, scale):
        f = 1 / (1 + np.exp(-(x - mu) / scale))
        return f

    @staticmethod
    def calc_pars(arr_values):
        vals_soh = arr_values

        mu = np.mean(vals_soh)
        scale = np.std(vals_soh) * np.sqrt(3) / np.pi
        f = Logistic.pdf(mu, scale)
        x = np.linspace(0, 100, 1000)[:, np.newaxis]
        y = f(x)

        idx_filt = y > 10 ** (-9)
        x_filt = x[idx_filt]
        y_filt = y[idx_filt]

        return x_filt.reshape(-1).tolist(), y_filt.reshape(-1).tolist(), mu, scale


class Gamma:
    @staticmethod
    def cdf(x, k, theta):
        f = stats.gamma.cdf(x, a=k, scale=1 / theta)
        return f
    @staticmethod
    def calc_pars(arr_values):
        vals_soh = arr_values
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(vals_soh, floc=0)

        x = np.linspace(0.1, 100, 1000)[:, np.newaxis]
        y = stats.gamma.pdf(x, fit_alpha, fit_loc, fit_beta)
        theta = 1 / fit_beta

        idx_filt = y > 10 ** (-9)
        x_filt = x[idx_filt]
        y_filt = y[idx_filt]

        return x_filt.reshape(-1).tolist(), y_filt.reshape(-1).tolist(), fit_alpha, theta


class Beta:
    @staticmethod
    def pdf(alpha, beta):
        f = lambda x: (1 / beta_func(alpha, beta)) * x ** (alpha - 1) * (1 - x) ** (beta - 1)
        return f

    @staticmethod
    def cdf(x, alpha, beta):
        x_1 = [value / 100 for value in x]
        f = stats.beta.cdf(x_1, alpha, beta)
        return f

    @staticmethod
    def calc_pars(arr_values):
        vals_soh = arr_values

        fit_alpha, fit_beta, fit_loc, fit_scale = stats.beta.fit(vals_soh, floc=0, fscale=100)

        f = Beta.pdf(fit_alpha, fit_beta)
        x = np.linspace(0.001, 1, 1000)[:, np.newaxis]
        y = f(x) / 100

        idx_filt = y > 10 ** (-9)
        x_filt = x[idx_filt] * 100
        y_filt = y[idx_filt]

        return x_filt.reshape(-1).tolist(), y_filt.reshape(-1).tolist(), fit_alpha, fit_beta


def calc_ecdf(arr_values):
    vals_soh = arr_values
    vals_soh_unique, counts = np.unique(vals_soh, return_counts=True)
    prob = counts / sum(counts)
    ecdf = np.cumsum(prob)
    x = np.insert(vals_soh_unique + 1, 0, vals_soh_unique[0])
    ecdf = np.insert(ecdf, 0, 0)
    return x, ecdf


def calc_rss():
    """Fit distribution functions and calculate residuals and chiquadrat to empirical data."""

    pdf = load_data('soh_countdict_cluster')
    pdf = pdf.sort_values(by=['energythrougput_kwh'])
    pdf = pdf[pdf['energythrougput_kwh'] <= 6000]

    distributions = [Normal, LogNormal, Weibull, Exponential, Logistic, Gamma, Beta]

    dict_rss_sum = {dist.__name__: [] for dist in distributions}
    dict_chisquare = {dist.__name__: [] for dist in distributions}

    for idx in range(0, len(pdf)):
        row_soh = pdf.iloc[idx, :]
        vals_soh = get_values_from_dict(row_soh['d_soh_%'])
        vals_soh_minus = 100-vals_soh

        vals_soh_unique, counts = np.unique(vals_soh_minus, return_counts=True)
        prob_reference = counts / sum(counts)

        for dist in distributions:
            dist_vals = dist.calc_pars(vals_soh_minus)
            expected_values = np.interp(vals_soh_unique, dist_vals[0], dist_vals[1])

            residual = (prob_reference - expected_values) ** 2
            chisquare = sum(counts) * sum(residual / expected_values)

            dict_rss_sum[dist.__name__].append(sum(residual))
            dict_chisquare[dist.__name__].append(sum(chisquare))

    return dict_rss_sum, dict_chisquare


def calc_ks_test():
    """Perform Kolmogorov_Smirnov test."""

    pdf = load_data('soh_countdict_cluster')
    pdf = pdf.sort_values(by=['energythrougput_kwh'])
    pdf = pdf[pdf['energythrougput_kwh'] <= 6000]

    distributions = [Normal, LogNormal, Weibull, Exponential, Logistic, Gamma, Beta]
    dict_ks = {dist.__name__: [] for dist in distributions}

    significance = [0.31416, 0.30397, 0.29471, 0.28626, 0.2785, 0.27135, 0.26473, 0.25857, 0.25283, 0.24746, 0.24241,
                    0.23767, 0.2332, 0.22897, 0.22497, 0.22117, 0.21756, 0.21412]  # 90%

    list_significance = []
    list_n = []

    for idx in range(0, len(pdf)):

        row_soh = pdf.iloc[idx, :]
        vals_soh = get_values_from_dict(row_soh['d_batt48_speicher_soh_%'])
        vals_soh_minus = 100-vals_soh

        vals_soh_unique, counts = np.unique(vals_soh_minus, return_counts=True)
        list_n.append(len(vals_soh_unique))

        critical_value = significance[len(vals_soh_unique) - 14]
        list_significance.append(critical_value)

        # eCDF
        x_orig, ecdf = calc_ecdf(vals_soh_minus)
        x_cdfs = np.linspace(0, 100, 1001)[:, np.newaxis]

        for dist in distributions:
            dist_vals = dist.calc_pars(vals_soh_minus)
            cdf_vals = dist.cdf(x_cdfs, *dist_vals[2:])

            idx = np.where(np.isin(x_cdfs, vals_soh_unique))[0]
            expected_values = np.concatenate([cdf_vals[i] for i in idx])

            diff_dist_x1 = np.abs(expected_values - ecdf[:-1])
            diff_dist_x = np.abs(expected_values - ecdf[1:])
            diff_dist = np.max([np.max(diff_dist_x1), np.max(diff_dist_x)])

            dict_ks[dist.__name__].append(diff_dist)

    return dict_ks, list_significance, list_n
