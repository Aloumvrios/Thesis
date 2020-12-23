import os
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from repr_core.repr_core import ReprCore
from thesis_core.thesis_utils import get_mean_cdist, get_permutation_of_pdists, get_KS_of_pdists
from scipy.spatial import distance
from scipy.stats import shapiro, ttest_rel, wilcoxon
from mlxtend.evaluate import permutation_test
from time import time
from thesis_core.thesis_core import ThesisCore
import researchpy
import arff

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def read_data(directory):
    cwd = os.getcwd()
    root = os.path.abspath(os.path.join(cwd, os.pardir))
    root = os.path.abspath(os.path.join(root, os.pardir))
    data_dir = os.path.abspath(os.path.join(root, directory))
    paths = [(os.path.join(data_dir, file), file[:-9])
             for file in os.listdir(data_dir) if '.fas' in file]
    # Create a list of dicts.
    # Each dictionary will have keys (chromosome, position, sequence, species)
    ucne_dicts = []
    ucne_dict = {}
    dfs = []

    for path, species in paths:
        with open(path, "r") as f:
            file = f.readlines()
            count = 0
            for line in file:
                if line[0] == ">":
                    ucne_dict["chromosome"] = line.split(":")[0][4:]
                    ucne_dict["position"] = line.split(":")[1][:-2]
                else:
                    ucne_dict["sequence"] = line[:-2]
                count += 1
                if count % 2 == 0:
                    ucne_dict["species"] = species
                    ucne_dicts.append(ucne_dict)
                    ucne_dict = {}
        dfs.append(pd.DataFrame(ucne_dicts))
    return dfs


def investigate_similarity_threshold(repr1, repr2):
    distances = []
    temp_df1 = repr1.reprs['FCGSR'].repr_df.copy()
    temp_df2 = repr2.reprs['FCGSR'].repr_df.copy()
    dfs = [temp_df1, temp_df2]
    most_dist, sim_array = get_mean_cdist(dfs)
    # fracd  = abs(self._repr_descriptor.get_fractal_d(dfs[0]) - self._repr_descriptor.get_fractal_d(dfs[1]))
    # sim_array = get_hellinger_distance(dfs)
    # distances.append(sim_array[0][1])
    # distances[indi][indj] = fracd
    # df = pd.DataFrame(data=distances)
    # sns.displot(df, stat='probability', binwidth=0.001)
    # plt.show()
    # print(df.quantile(q=0.95))
    plt.show()
    return sim_array[0][1]


def compare_distance_between_species():
    paths = ['data/UCNEs/hUCNEs_-_InsectUCNEs',
             'data/UCNEs/hUCNEs_-_WormUCNEs',
             'data/UCNEs/InsectUCNEs_-_WormUCNEs']
    dists = []
    for path in paths:
        dfs = read_data(path)
        repr1 = ReprCore()
        repr1.set_dataframe(dfs[0])
        repr1.create_reprs(kmer=2)
        repr2 = ReprCore()
        repr2.set_dataframe(dfs[1])
        repr2.create_reprs(kmer=2)
        dists.append(investigate_similarity_threshold(repr1, repr2))
    print(dists)


def test_distances_w_random(noise_lvl=0):
    """
    Test pairwise cosine distances between samples with distances between sample-random dataset
    with paired t-test or mann- whitney to see if the mean difference between the paired samples is zero
    :param noise_lvl: Noise should be in the form of a random value in the range of the random dataset
    :return:
    """
    core = ThesisCore(in_package=True)
    core.create_reprs(kmer=3)
    for name, repr in core._repr_core.reprs.items():
        # create 10 subsets and 10 random of the same size
        subsets = []
        randoms = []
        test_results = []
        temp_df = repr.repr_df.copy().drop(['species'], axis=1)
        for indi, i in enumerate(range(0, 6000, 600)):
            subset = temp_df[i:i + 600]
            # apply noise if available
            noisy_subset = subset.applymap(
                lambda x: x + np.random.uniform(-1000, 1000) if np.random.uniform() < noise_lvl else x)
            subsets.append(noisy_subset)
            randoms.append(pd.DataFrame(np.random.randint(-1000, 1000, size=subset.shape),
                                        columns=subset.columns))
        for i in range(len(subsets)):
            m1 = []
            m2 = []
            for j in range(len(subsets)):
                if i == j:
                    continue
                _, intersubset_dist = get_mean_cdist(subsets[i], subsets[j])
                m1.append(intersubset_dist)
                _, subrandom_dist = get_mean_cdist(subsets[i], randoms[j])
                m2.append(subrandom_dist)
            # perform statistical test between the 2 lists
            # normality test
            shapiro_test_1 = shapiro(m1)
            shapiro_test_2 = shapiro(m2)
            result = 'cannot reject' if shapiro_test_1.pvalue > 0.05 and shapiro_test_2.pvalue > 0.05 else 'reject'
            # print('For subset', i, 'we', result, 'the H0 that the intersubset dists and '
            #                                      'the dists with the random were drawn from a
            #                                      normal distribution')
            test_result = 0
            if result == 'cannot reject':
                # paired t-test
                ttest = ttest_rel(m1, m2)
                test_result = 'cannot reject' if ttest.pvalue > 0.05 else 'reject'
                print('For subset', i, 'We', test_result, 'the H0 that intersubset dists and '
                                                          'the dists with the random have identical '
                                                          'average values, with pvalue', ttest.pvalue)
            else:
                print('cannot perform paired t-test')
                # wilcoxon for non normal dependent samples
                wtest = wilcoxon(m1, m2)
                test_result = 'cannot reject' if wtest.pvalue > 0.05 else 'reject'
                print('For subset', i, 'We', test_result, 'the H0 that intersubset dists and '
                                                          'the dists with the random have equal '
                                                          'median values, with pvalue', wtest.pvalue)
            test_results.append((test_result, np.mean(m1)))

        prevalent_result = max(set([x for x, _ in test_results]), key=[x for x, _ in test_results].count)
        threshold = max([y for x, y in test_results if x == prevalent_result])

    return prevalent_result, threshold


def test_distances_w_random_w_noise():
    """
    Progressively add noise to the subsets.
    Noise is in the form of a random value in the range of the random dataset
    and
    Test pairwise cosine distances between noisy samples with distances between noisy_sample-random dataset
    with paired t-test or mann- whitney to see if the mean difference between the paired samples is zero
    :return:
    """
    for noise_lvl in np.arange(0, 1, 0.1):
        print('For Noise lvl', noise_lvl)
        prevalent_result, threshold = test_distances_w_random(noise_lvl)
        print('For Noise_lvl', noise_lvl, 'the prevalent result is', prevalent_result, 'and threshold', threshold)
        if prevalent_result == 'cannot reject':
            break


def do_different_species_match(test):
    """
    Get distance matrix of each dataset and perform permutation/K-S test among them
    to see if the distance distributions come from the same distribution
    :return:
    """
    paths = ['data/UCNEs/hUCNEs_-_InsectUCNEs',
             'data/UCNEs/hUCNEs_-_WormUCNEs',
             'data/UCNEs/InsectUCNEs_-_WormUCNEs']
    for path in paths:
        dfs = read_data(path)
        repr1 = ReprCore()
        repr1.set_dataframe(dfs[0])
        repr1.create_reprs(kmer=2)
        repr2 = ReprCore()
        repr2.set_dataframe(dfs[1])
        repr2.create_reprs(kmer=2)
        temp_df1 = repr1.reprs['FCGSR'].repr_df.copy()
        temp_df2 = repr2.reprs['FCGSR'].repr_df.copy()

        if test == 'permutation':
            p_value = get_permutation_of_pdists(temp_df1, temp_df2)
            print(p_value)
        elif test == 'KS':
            p_value = get_KS_of_pdists(temp_df1, temp_df2)
            if p_value > 0.05:
                print('we Cannot reject the H0 that the 2 pdists are identical with pvalue',
                      p_value)
            else:
                print('we Reject the H0 that the 2 pdists are identical with pvalue', p_value)


def do_different_samples_match(test):
    core = ThesisCore(in_package=True)
    core.create_reprs(kmer=3)
    for name, repr in core._repr_core.reprs.items():
        temp_df = repr.repr_df.copy()
        for indi, i in enumerate(range(0, 6000, 600)):
            for indj, j in enumerate(range(0, 6000, 600)):
                if indj <= indi:
                    continue
                temp_df1, temp_df2 = temp_df[i:i + 600], temp_df[j:j + 600]
                if test == 'permutation':
                    p_value = get_permutation_of_pdists(temp_df1, temp_df2)
                    print(p_value)
                elif test == 'KS':
                    p_value = get_KS_of_pdists(temp_df1, temp_df2)
                    if p_value > 0.05:
                        print('we Cannot reject the H0 that the 2 dists', indi, indj, 'are identical with pvalue',
                              p_value)
                    else:
                        print('we Reject the H0 that the 2 dists ', indi, indj, 'are identical with pvalue', p_value)
                elif test == 'crosstab':
                    res = researchpy.crosstab(temp_df1, temp_df2, test="chi-square")
                    print(res)
                    print(res['p-value'])


def test_sampling_methods(method):
    """
    :param method: permutation_pdist, KS_pdist
    :return:
    """
    print("~test_sampling_methods!")
    start = time()
    core = ThesisCore(in_package=True)
    core.create_reprs(kmer=3)
    for name, repr in core._repr_core.reprs.items():
        attempts = []
        time_list = []
        ind_list = []
        temp_df = repr.repr_df.copy()
        dfs = []
        feature_df_copy = temp_df.copy()
        num_of_datasets = 300
        first_sample = feature_df_copy.sample(frac=0.1)
        dfs.append(first_sample)
        for ind, i in enumerate(range(num_of_datasets - 1)):
            loop_start = time()
            found = False
            while not found:
                next_sample = feature_df_copy.sample(frac=0.1)
                found = True
                for dfr in dfs:
                    if method == 'permutation_pdist':
                        p_value = get_permutation_of_pdists(dfr, next_sample)
                        if p_value > 0.05:
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
                        if dist < 0.9997031552913787:
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
        print(f"~test_sampling_methods ended!It took {end - start} seconds!")
        plt.scatter(ind_list, time_list)
        plt.show()


def read_arff_data():
    cwd = os.getcwd()
    root = os.path.abspath(os.path.join(cwd, os.pardir))
    root = os.path.abspath(os.path.join(root, os.pardir))
    data_dir = os.path.abspath(os.path.join(root, 'data'))
    repr_dict = {}
    repr_folders = os.listdir(data_dir)
    for folder in repr_folders:
        if '.ini' in folder or 'UCNEs' in folder:
            continue
        repr_dict_list = []
        repr_dir, repr_name = os.path.join(data_dir, folder), folder
        for file in os.listdir(repr_dir):
            if '.ini' in file:
                continue
            file_path = os.path.join(repr_dir, file)
            with open(file_path) as f:
                print('reading file', file_path)
                repr_dict_list += arff.load(f)['data']
        print(len(repr_dict_list))
        full_df = pd.DataFrame(repr_dict_list)
        print(full_df.shape)
        full_df.drop_duplicates()
        print(full_df.shape)
        print('description')
        print(full_df.describe())

        # encode nominal class
        class_col = full_df.columns[-1]
        for class_idx, class_name in enumerate(set(full_df[class_col])):
            print("class_name", class_name)
            idx = full_df[full_df.columns[-1]] == class_name
            full_df.loc[idx, class_col] = class_idx
        repr_dict[repr_name] = full_df

    return repr_dict


def test_h5():
    df1 = pd.DataFrame([[1, 1.0, 'a']], columns=['x', 'y', 'z'])
    df2 = pd.DataFrame([[4, 1.0, 'g','jikoukou']], columns=['t', 'k', 'm','mplou'])

    df1.to_hdf('./store.h5', 'data1')
    df2.to_hdf('./store.h5', 'data2')
    import h5py
    filename = "store.h5"

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        for i in list(f.keys()):
            reread = pd.read_hdf('./store.h5',i)
            print(reread)


if __name__ == '__main__':
    # compare_distance_between_species()
    # do_different_species_match(test='KS')
    # permutation Since p-value =0, we can reject the null hypothesis that
    # the two samples come from the same distribution.
    # KS Since p-value =0, we can reject the null hypothesis that
    # the two samples come from the same distribution.
    # do_samples_match()

    #
    # do_different_samples_match(test='crosstab')
    #
    # notes
    # with KS sxedon ola reject, dld ta vlepei oti ta pdists einai apo diaforetiko distribution
    # wih perm sxedon ola reject, dld ta vlepei oti ta pdists einai apo diaforetiko distribution
    # with crosstab

    #
    # test_sampling_methods(method='KS_pdist')
    #
    # notes
    # with KS_pdist gia 30 samples, kanena failed attempt
    # with perm_pdist ekane 8h na treksei
    # with mean_cdist den vriskei pote...

    # test_distances_w_random()

    # test_distances_w_random_w_noise()
    #
    # notes
    # faineta oti gia 0.3 -0.4 noise de mporoume na apokleisoume oti
    # ta subsets exoun idia mesh apostastash me ta random
    # kai oriaki apostash 0.9997031552913787

    # repr_dict = read_arff_data()
    test_h5()
