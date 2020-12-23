from scipy.spatial import distance
from scipy.stats import kstest
import numpy as np
from settings import distance_threshold, method, max_datasets, alpha
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from time import time
from mlxtend.evaluate import permutation_test


def split_dataset(feature_df):
    start = time()
    dfs = []
    attempts = []
    time_list = []
    ind_list = []
    feature_df_copy = feature_df.copy()
    first_sample = feature_df_copy.sample(frac=0.1)
    dfs.append(first_sample)
    for ind, i in enumerate(range(max_datasets - 1)):
        loop_start = time()
        found = False
        while not found:
            # print(len(feature_df_copy))
            next_sample = feature_df_copy.sample(frac=0.1)
            # print(len(next_sample))
            found = True
            for dfr in dfs:
                if method == 'permutation_pdist':
                    p_value = get_permutation_of_pdists(dfr, next_sample)
                    if p_value > alpha:
                        attempts.append(p_value)
                        found = False
                        break
                elif method == 'KS_pdist':
                    p_value = get_KS_of_pdists(dfr, next_sample)
                    if p_value > 0.05:
                        attempts.append(p_value)
                        found = False
                        break
                elif method == 'mean_cdist':
                    _, dist = get_mean_cdist(dfr, next_sample)
                    if dist < distance_threshold:
                        print(dist)
                        attempts.append(dist)
                        found = False
                        break
        dfs.append(next_sample)
        loop_end = time()
        time_diff = loop_end - loop_start
        # print('len dfs', len(dfs), time_diff)
        time_list.append(time_diff)
        ind_list.append(i)
    print('Failed attempts', len(attempts))
    # print(attempts)
    end = time()
    print(f"~split_datasets ended!It took {end - start} seconds!")
    # plt.scatter(ind_list, time_list)
    # plt.show()
    return dfs


def get_mean_cdist(df1, df2):
    A = df1.drop(['species'], axis=1) if 'species' in df1.keys() else df1
    A = A.to_numpy()
    B = df2.drop(['species'], axis=1) if 'species' in df2.keys() else df2
    B = B.to_numpy()
    cdist = distance.cdist(A, B, 'cosine')
    cdistflat = np.hstack(cdist)
    most_dist = np.quantile(cdistflat, 0.95)
    dist = np.mean(cdist)
    return most_dist, dist


def get_cosine_distances_scrapped(df_list):
    distances = np.zeros((len(df_list), len(df_list)))

    for i, df_i in enumerate(df_list):
        A = df_i.drop(['species'], axis=1) if 'species' in df_i.keys() else df_i
        A = A.to_numpy()
        # Aflat = np.hstack(A.to_numpy())
        for j, df_j in enumerate(df_list):
            if j <= i:
                continue
            B = df_j.drop(['species'], axis=1) if 'species' in df_j.keys() else df_j
            B = B.to_numpy()
            # Bflat = np.hstack(B.to_numpy())
            # distances[i][j] = distance.cosine(Aflat, Bflat)
            cdist = distance.cdist(A, B, 'cosine')
            cdistflat = np.hstack(cdist)
            # sns.displot(cdistflat, stat='probability')
            # sns.boxenplot(cdistflat)
            # plt.draw()
            # plt.show()
            most_dist = np.quantile(cdistflat, 0.95)
            # print("5% more distant data points",most_dist )
            # print("mean", cdist.mean())
            # C = pd.DataFrame(data=cdist)
            # sns.heatmap(cdist, annot=True)
            # plt.imshow(cdist, cmap='hot', interpolation='nearest')

            # distances[i][j] = np.median(cdist)
            distances[i][j] = np.mean(cdist)
    # plt.show()
    return most_dist, distances


def get_hellinger_distance(df_list):
    _SQRT2 = np.sqrt(2)  # sqrt(2) with default precision np.float64
    dists = np.zeros((len(df_list), len(df_list)))
    for i, df_i in enumerate(df_list):
        p = df_i.drop(['species'], axis=1) if 'species' in df_i.keys() else df_i
        p = p.to_numpy()
        for j, df_j in enumerate(df_list):
            q = df_j.drop(['species'], axis=1) if 'species' in df_j.keys() else df_j
            q = q.to_numpy()
            hellinger = np.sqrt(np.nansum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2
            dists[i][j] = hellinger
    return dists


def get_pdists(temp_df1, temp_df2):
    # metr = 'cosine'
    metr = 'euclidean'
    temp_df1_ndarray = temp_df1.drop(['species'], axis=1) if 'species' in temp_df1 else temp_df1
    temp_df1_ndarray = temp_df1_ndarray.to_numpy()
    temp_df1_pdist = distance.pdist(temp_df1_ndarray, metric=metr)

    temp_df2_ndarray = temp_df2.drop(['species'], axis=1) if 'species' in temp_df2 else temp_df2
    temp_df2_ndarray = temp_df2_ndarray.to_numpy()
    temp_df2_pdist = distance.pdist(temp_df2_ndarray, metric=metr)
    return temp_df1_pdist, temp_df2_pdist


def get_KS_of_pdists(temp_df1, temp_df2):
    temp_df1_pdist, temp_df2_pdist = get_pdists(temp_df1, temp_df2)
    k_s = kstest(temp_df1_pdist, temp_df2_pdist, )
    return k_s.pvalue


def get_permutation_of_pdists(temp_df1, temp_df2):
    temp_df1_pdist, temp_df2_pdist = get_pdists(temp_df1, temp_df2)

    # print("~Starting permutation test!")
    # start = time()
    p_value = permutation_test(temp_df1_pdist, temp_df2_pdist,
                               method='approximate',
                               num_rounds=1000,
                               seed=0)
    # end = time()
    # print(f"~Permutation test ended!It took {end - start} seconds!")

    return p_value
