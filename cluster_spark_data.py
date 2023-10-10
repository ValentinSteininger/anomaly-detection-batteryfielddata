from pyspark.sql.functions import udf
from scipy.stats import norm
from pyspark.ml.feature import VectorAssembler
import pyspark.ml.feature
from pyspark.ml.functions import vector_to_array
import pyspark.sql.functions as F
import pyspark.sql.types as T

import numpy as np
import copy

from itertools import combinations

""" 
To execute the following functions on a local machine, 
a spark session needs to be set up using an exemplary structure as follows:

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("sparksession").getOrCreate()

    path_excel = r"PATH_TO_SAMPLE_EXCEL"
    pdf = pd.read_excel(path_excel)
    df_source = spark.createDataFrame(pdf)
"""


def preprocessing(df_source):
    """This function selects the required columns and performs a
    principal component analysis on histogram variables."""

    def compress_with_pca(df, feat_groups, feat_names=tuple(), n_pca=1):
        df_pca = df
        cols_pca = []

        for c, features in enumerate(feat_groups):
            # assemble feature columns in vector column
            vector_col = 'pca_features'
            assembler = VectorAssembler(inputCols=features, outputCol=vector_col, handleInvalid='skip')
            df_pca = assembler.transform(df_pca)

            # apply PCA on vector column
            if feat_names:
                col_pca = feat_names[c]
            else:
                col_pca = f'pca{len(features)}'

            cols_pca.append(col_pca)

            model = pyspark.ml.feature.PCA(k=n_pca, inputCol=vector_col, outputCol=col_pca).fit(df_pca)

            df_pca = model.transform(df_pca)
            df_pca = df_pca.withColumn(col_pca, vector_to_array(col_pca)[0])
            df_pca = df_pca.drop(vector_col)

        df_pca = df_pca.select(['readout_id_encrypted'] + cols_pca)
        return df_pca

    df_features = df_source.select([
        'readout_id_encrypted',
        'soh_%',
        'energy_troughput_kwh',
        'mileage_km'])

    cols_pca = [
        [f'time_soc_range_{i}_s' for i in range(1, 11)],
        [f'time_temperature_range_{i}_s' for i in range(1, 7)],
        [f'charge_count_temperature_range_{i}_ah' for i in range(1, 7)]
    ]
    names_pca = ('time_soc_range', 'time_temperature_range', 'charge_count_temperature_range')

    df_pca = compress_with_pca(df_source, cols_pca, names_pca)

    df_pca = df_pca.withColumn('time_soc_range', F.col('time_soc_range') / 3600 / 24)
    df_pca = df_pca.withColumn('time_temperature_range', F.col('time_temperature_range') / 3600 / 24)
    df_pca = df_pca.withColumn('charge_count_temperature_range',
                               F.col('charge_count_temperature_range') / 1000)

    df_features = df_features.join(df_pca, on='readout_id_encrypted', how='inner')
    return df_features


def compute_treediagram(source_df):
    """This function computes the average cluster size and standard deviation for all combinations of given features."""

    cols_cluster = {
        'mileage_km': 5000,
        'energy_troughput_kwh': 200,
        'time_soc_range': 10,
        'time_temperature_range': 5,
        'charge_count_temperature_range': 2
    }

    dict_translate = {
        'mileage_km': 'A',
        'energy_troughput_kwh': 'B',
        'time_soc_range': 'C',
        'time_temperature_range': 'D',
        'charge_count_temperature_range': 'E'
    }

    list_combinations = list()

    for n in range(len(cols_cluster.keys()) + 1):
        list_combinations += list(combinations(list(cols_cluster.keys()), n))

    for col, stepsize in cols_cluster.items():  # noqa
        source_df = source_df.withColumn(col, F.round((F.col(col)/stepsize))*stepsize)

    dict_nclusters = {}
    for combi in list_combinations:  # noqa
        df_group = source_df.groupBy(*combi).agg(
            F.count('*').alias('n_readouts')
        )
        df_group_calc = df_group.groupBy().agg(
            F.avg('n_readouts').alias('mu_nreadouts'),
            F.stddev('n_readouts').alias('sig_nreadouts'),
            F.count('n_readouts').alias('n_clusters')
        )
        dict_nclusters[combi] = (df_group_calc.first()['n_clusters'], df_group_calc.first()['mu_nreadouts'], df_group_calc.first()['sig_nreadouts'], df_group.count())

    dict_pdf = {'combi': [], 'combi_translate': [], 'nclusters': []}
    for group, n_clusters in dict_nclusters.items():
        str_group = '+'.join(group)
        dict_pdf['combi'].append(str_group)
        dict_pdf['nclusters'].append(n_clusters)

        str_group_translate = ''.join([dict_translate[col] for col in group])
        dict_pdf['combi_translate'].append(str_group_translate)

    return dict_pdf


def cluster_features(df_source, cols_cluster, calc_pars_dist=None, calc_perc_dist=None):
    """This function clusters SOH datapoints based on the defined stepsize and calculates the confidence intervals
    based on the lognorm distribution function."""

    col_soh = 'soh_%'
    df_array = df_source

    # fit distribution
    @udf(T.ArrayType(T.FloatType()))
    def calc_pars_lognorm(list_col):
        soh = np.array(list_col)
        dsoh = 100 - soh
        dsoh[dsoh == 0] = 0.1
        ln_soh = np.log(dsoh)

        mu = np.mean(ln_soh)
        sig = np.std(ln_soh)
        return [float(mu), float(sig)]

    @udf(T.ArrayType(T.FloatType()))
    def calc_perc_lognorm(list_pars, prob):
        mu = list_pars[0]
        sig = list_pars[1]

        prob = prob / 100 / 2
        z_score = (norm.ppf(prob, loc=mu, scale=sig) - mu)/sig
        low_lim = np.exp(mu + sig*z_score)
        up_lim = np.exp(mu - sig*z_score)
        return [float(low_lim), float(up_lim)]

    df_array = df_array.select([col_soh] + [col[0] for col in cols_cluster])
    for col in cols_cluster:  # noqa
        df_array = df_array.withColumn(col[0], F.round((F.col(col[0])/col[1]))*col[1])
    df_array = df_array.groupBy(*[col[0] for col in cols_cluster]).agg(
        F.collect_list(col_soh).alias(col_soh))

    df_array = df_array.where(F.size(F.array_distinct(col_soh)) > 1)

    df_array = df_array.withColumn('pars_dist', calc_pars_lognorm(F.col('batt48_speicher_soh_%')))
    df_array = df_array.withColumn('lims_95%', calc_perc_lognorm(F.col('pars_dist'), F.lit(5.0)))
    df_array = df_array.withColumn('lims_90%', calc_perc_lognorm(F.col('pars_dist'), F.lit(10.0)))
    df_array = df_array.withColumn('lims_80%', calc_perc_lognorm(F.col('pars_dist'), F.lit(20.0)))
    df_array = df_array.withColumn('lims_50%', calc_perc_lognorm(F.col('pars_dist'), F.lit(50.0)))

    # compute map df
    @udf(T.MapType(T.FloatType(), T.IntegerType()))
    def count_values(list_col):
        unique, counts = np.unique(list_col, return_counts=True)
        list_unique = [float(i) for i in unique.tolist()]
        list_counts = [int(i) for i in counts.tolist()]
        return dict(zip(list_unique, list_counts))

    df_map = df_array.withColumn('d_'+col_soh, count_values(F.col(col_soh)))
    df_map = df_map.drop(col_soh)
    df_map = df_map.withColumnRenamed('d_'+col_soh, col_soh)
    df_map = df_map.select(*df_array.schema.names)

    # compute count df
    colnames = copy.copy(df_array.schema.names)
    colnames.remove(col_soh)
    df_count = df_map.select(*colnames, F.explode(col_soh))
    df_count = df_count.withColumnRenamed('key', col_soh)
    df_count = df_count.withColumnRenamed('value', 'n_'+col_soh)

    return df_array, df_map, df_count