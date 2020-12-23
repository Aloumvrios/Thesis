import os
import sys

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import logging

logging.basicConfig(filename='mylog.log', level=logging.INFO)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
from repr_core.repr_core import ReprCore
from ml_core.ml_core import Classifiers, Regressors, Statistics
from thesis_core.thesis_utils import split_dataset, get_mean_cdist
from repr_core.repr_description import ReprDescriptor
import seaborn as sns
from time import time
import settings
import pickle
import arff
from scipy import stats


class ThesisCore(object):
    def __init__(self, in_package=False):
        self._in_package = in_package
        self._repr_core = ReprCore()
        self._ml_core_clf = Classifiers()
        self._ml_core_rgr = Regressors()
        self._ml_core_stats = Statistics()
        self._repr_descriptor = ReprDescriptor()

    def read_arff_data(self, filepath):
        """

        :param filepath:
        :return:
        """
        df_list = []
        for file in os.listdir(filepath):
            if '.ini' in file:
                continue
            file_path = os.path.join(filepath, file)
            with open(file_path) as f:
                data = arff.load(f)['data']
            file_df = pd.DataFrame(data)
            file_df['temp'] = file_df[file_df.columns[0]]
            file_df[file_df.columns[0]] = file_df[file_df.columns[-2]]
            file_df[file_df.columns[-2]] = file_df['temp']
            file_df = file_df.drop(columns=['temp'])  # now each file-df has the first column as species
            df_list.append(file_df)
        full_df = pd.concat(df_list, axis=0, ignore_index=True)

        # encode nominal class
        class_col = full_df.columns[0]
        for class_idx, class_name in enumerate(set(full_df[class_col])):
            idx = full_df[class_col] == class_name
            full_df.loc[idx, class_col] = class_idx
        full_df = full_df.fillna(0)
        full_df['temp'] = full_df[full_df.columns[0]]
        full_df[full_df.columns[0]] = full_df[full_df.columns[-2]]
        full_df[full_df.columns[-2]] = full_df['temp']
        full_df = full_df.drop(columns=['temp'])  # now fulldf has the last column as species

        return full_df

    def read_all_arff_data(self, filepath):
        """

        :param filepath:
        :return:
        """
        repr_dict = {}
        repr_folders = os.listdir(filepath)
        for folder in repr_folders:
            if '.ini' in folder:
                continue
            print(folder)
            df_list = []
            repr_dir, repr_name = os.path.join(filepath, folder), folder
            for file in os.listdir(repr_dir):
                if '.ini' in file:
                    continue
                file_path = os.path.join(repr_dir, file)
                with open(file_path) as f:
                    data = arff.load(f)['data']
                file_df = pd.DataFrame(data)
                file_df['temp'] = file_df[file_df.columns[0]]
                file_df[file_df.columns[0]] = file_df[file_df.columns[-2]]
                file_df[file_df.columns[-2]] = file_df['temp']
                file_df = file_df.drop(columns=['temp'])  # now each file-df has the first column as species
                df_list.append(file_df)
            full_df = pd.concat(df_list, axis=0, ignore_index=True)

            # encode nominal class
            class_col = full_df.columns[0]
            for class_idx, class_name in enumerate(set(full_df[class_col])):
                # print("class_name", class_name)
                idx = full_df[class_col] == class_name
                full_df.loc[idx, class_col] = class_idx
            full_df = full_df.fillna(0)
            repr_dict[repr_name] = full_df

        return repr_dict

    def read_fas_data(self, cwd):
        root = os.path.abspath(os.path.join(cwd, os.pardir))
        data_dir = os.path.abspath(os.path.join(root, 'data/UCNEs'))
        paths = [(os.path.join(root, file), file[:-9])
                 for root, dirs, files in os.walk(data_dir) for file in files
                 if '.fas' in file]
        # Create a list of dicts.
        # Each dictionary will have keys (chromosome, position, sequence, species)
        ucne_dicts = []
        ucne_dict = {}
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
        df = pd.DataFrame(ucne_dicts)
        return df

    def read_data(self, in_package):
        """Reads UCNEs and combines all into a single dataframe
        with species as a target variable"""

        cwd = os.getcwd() if in_package is False else 'C:/Users/Nick/Google Drive/Study/DataScience/_Nikolaou_DSC18014_Thesis/thesis'

        return self.read_fas_data(cwd)

    def pick_c_classes(self, df):
        c = settings.no_of_classes
        classes = df[df.columns[-1]].unique().tolist()
        chosen_classes = random.sample(classes, c)
        new_df = df[df[df.columns[-1]].isin(chosen_classes)]
        return new_df

    def load_reprs(self):
        """
        reads reprs from arff files and creates respective REPR objects
        :return:
        """
        start = time()
        print('Loading of arff started!')
        repr_dict = self.read_all_arff_data(
            'C:/Users/Nick/Google Drive/Study/DataScience/_Nikolaou_DSC18014_Thesis/data2')
        self._repr_core.create_reprs_from_dict(repr_dict)
        end = time()
        print(f"~Loading arff ended!It took {end - start} seconds!")

    def load_repr_names(self):
        filepath = 'C:/Users/Nick/Google Drive/Study/DataScience/_Nikolaou_DSC18014_Thesis/data2'
        repr_names = [name for name in os.listdir(filepath) if '.ini' not in name]
        self._repr_core.add_reprs(repr_names)

    def read_split_correlate(self):
        for name, repr in self._repr_core.reprs.items():
            self.read_split_dataset(repr)
            self.correlate_metafeatures_to_avg_scores(repr)

    def create_reprs(self, kmer):
        print("~Creating Reprs started!")
        start = time()
        df = self.read_data(self._in_package)
        df = self.pick_c_classes(df)
        self._repr_core.set_dataframe(df)
        self._repr_core.create_reprs(kmer)
        end = time()
        print(f"~Creating reprs ended!It took {end - start} seconds!")

    def read_split_dataset(self, repr):
        print("~Splitting started!")
        start = time()
        # fname = repr.name + settings.file_suffix
        fname = repr.name + '_' + str(settings.alpha) + '_' + settings.method
        got_enough_datasets = False
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                df_list = pickle.load(f)
            # with h5py.File(fname, "r") as f:
            #     # List all groups
            #     # print("Keys: %s" % f.keys())
            #     df_list = [pd.read_hdf(fname, i) for i in list(f.keys())]

            if len(df_list) >= settings.num_of_datasets:
                got_enough_datasets = True
        if got_enough_datasets:
            new_df_list = random.sample(df_list, settings.num_of_datasets)
            repr.set_repr_df_list(new_df_list)
        else:
            datapath = 'C:/Users/Nick/Google Drive/Study/DataScience/_Nikolaou_DSC18014_Thesis/data2'
            filepath = os.path.join(datapath, repr.name)
            # print(filepath)
            repr_df = self.read_arff_data(filepath)
            # print(repr_df.shape)
            repr_df = self.pick_c_classes(repr_df)
            # print(repr_df.shape)
            repr.set_repr_df(repr_df)
            df_list = split_dataset(repr.repr_df)
            repr.set_repr_df_list(df_list)
            # for i, df in enumerate(df_list):
            #     df.to_hdf(fname, 'data'+str(i))
            with open(fname, 'wb') as f:
                pickle.dump(df_list, f)
        end = time()
        print(f"~Splitting ended!It took {end - start} seconds!")

    def split_all_datasets(self):
        print("~Splitting started!")
        start = time()
        for name, repr in self._repr_core.reprs.items():
            fname = name + settings.file_suffix
            if os.path.exists(fname):
                with open(fname, 'rb') as f:
                    df_list = pickle.load(f)
                    repr.set_repr_df_list(df_list)
            else:
                df_list = split_dataset(repr.repr_df)
                repr.set_repr_df_list(df_list)
                with open(fname, 'wb') as f:
                    pickle.dump(df_list, f)
        end = time()
        print(f"~Splitting ended!It took {end - start} seconds!")

    def describe_subsets(self, repr):
        print("~Description started!")
        start = time()
        self._repr_descriptor.describe_repr_list(repr)
        end = time()
        print(f"~Description ended!It took {end - start} seconds!")

    def classify_subsets(self, repr):
        print("~Classification started!")
        start = time()
        self._ml_core_clf.classify_repr_list(repr)
        end = time()
        print(f"~Classification ended!It took {end - start} seconds!")

    def describe_reprs(self):
        print("~Description started!")
        start = time()
        for name, repr in self._repr_core.reprs.items():
            self._repr_descriptor.describe_repr_list(repr)
        end = time()
        print(f"~Description ended!It took {end - start} seconds!")

    def classify_reprs(self):
        print("~Classification started!")
        start = time()
        for name, repr in self._repr_core.reprs.items():
            self._ml_core_clf.classify_repr_list(repr)
        end = time()
        print(f"~Classification ended!It took {end - start} seconds!")

    def spearman_heatmap_metaf_and_scores(self):
        for name, repr in self._repr_core.reprs.items():
            fname = name + '_descriptions_' + settings.file_suffix
            if os.path.exists(fname):
                descriptions = pd.read_pickle(fname)
            else:
                self.describe_reprs()
                descriptions = repr.descriptions

            fname = name + "_scores_" + settings.file_suffix
            if os.path.exists(fname):
                scores = pd.read_pickle(fname)
            else:
                self.classify_reprs()
                scores = repr.scores

            combination = pd.concat([descriptions, scores], axis=1, sort=False)
            # plot correlation matrix
            sns.heatmap(combination.corr(method='pearson').iloc[9:, :9], annot=True, fmt=".2f")
            plt.show()

    def print_descriptions(self):
        for name, repr in self._repr_core.reprs.items():
            fname = name + "_descriptions_" + settings.file_suffix
            if os.path.exists(fname):
                descriptions = pd.read_pickle(fname)
            print(descriptions)
            print(descriptions['frac_dim'].mean(), descriptions['frac_dim'].std())

            for col in descriptions.keys():
                print(len(descriptions[col].unique()))

    def get_descriptions_by_clf(self, name):
        dfs = []
        fname = name + "_descriptions_" + settings.file_suffix
        if os.path.exists(fname):
            descriptions = pd.read_pickle(fname)
        else:
            print("descriptions don't exist")
        fname = name + "_scores_" + settings.file_suffix
        if os.path.exists(fname):
            scores = pd.read_pickle(fname)
        else:
            print("clf scores don't exist")
        for clf in list(scores):
            new = pd.concat([descriptions, scores[clf]], axis=1, sort=False)
            dfs.append(new)
        return dfs

    def get_descriptions_by_agr_clf(self, repr, score_agr='clf_avg'):
        """

        :param repr:
        :param score_agr: clf_avg OR clf_max
        :return:
        """
        dfs = []  # dummy list
        fname = repr.name + "_descriptions_" + settings.file_suffix
        if os.path.exists(fname):
            descriptions = pd.read_pickle(fname)
            logging.info('\n\t' + descriptions.to_string().replace('\n', '\n\t'))
        else:
            print(fname, "descriptions don't exist. Let's create them!")
            self.describe_subsets(repr)
            descriptions = repr.descriptions

        fname = repr.name + "_scores_" + settings.file_suffix + '_' + settings.fast_mode_str
        if os.path.exists(fname):
            scores = pd.read_pickle(fname)
        else:
            print("clf scores don't exist. Let's create them!")
            self.classify_subsets(repr)
            scores = repr.scores

        scores_temp = scores.copy()
        scores_temp[score_agr] = scores.mean(axis=1)
        new = pd.concat([descriptions, scores_temp[score_agr]], axis=1, sort=False)
        dfs.append(new)  # append to dummy list
        return dfs

    def regress_on_scores(self):
        for name, repr in self._repr_core.reprs.items():
            dfs = self.get_descriptions_by_clf(name)
            reg_scores = self._ml_core_rgr.apply_regr_to_list(dfs)
            print(reg_scores)

    def print_n_important_features(self, n):
        for name, repr in self._repr_core.reprs.items():
            dfs = self.get_descriptions_by_clf(name)
            self._ml_core_rgr.print_n_important_features_from_list(dfs, n)

    def regress_on_n_important_features(self, n):
        for name, repr in self._repr_core.reprs.items():
            dfs = self.get_descriptions_by_clf(name)
            reg_scores = self._ml_core_rgr.apply_regr_to_list_on_n_important(dfs, n)
            fname = repr.name + '_regr_scores_of_' + str(settings.num_of_important_features) + settings.file_suffix
            reg_scores.to_pickle(fname)
            print(reg_scores)

    def plot_most_important_feature_per_classifier(self):
        for name, repr in self._repr_core.reprs.items():
            dfs = self.get_descriptions_by_clf(name)
            self._ml_core_rgr.plot_most_important_feature_per_classifier(dfs)

    def correlate_metafeatures_to_avg_scores(self, repr):
        fname = 'Spearman_' + repr.name + settings.file_suffix + '_' + settings.fast_mode_str + '_avg'
        logging.info(fname)
        dfs_avg = self.get_descriptions_by_agr_clf(repr, score_agr='clf_avg')
        df_corrs = self._ml_core_stats.correlate_metafeatures_to_scores(dfs_avg, score_agr='clf_avg')
        logging.info("H0: There is no [monotonic] association between the two meta-feature and the classifier.")
        logging.info('\n\t' + df_corrs.to_string().replace('\n', '\n\t'))
        df_corrs.to_pickle(fname)

        # plot metafeature to avg_cld score
        for df in dfs_avg:
            fig, axs = plt.subplots(nrows=3, ncols=3)
            sns.scatterplot(data=df, x='overl_dens', y='clf_avg', ax=axs[0][0])
            ybottom, ytop = axs[0][0].get_ylim()  # return the current ylim
            xbottom, xtop = axs[0][0].get_xlim()  # return the current xlim
            axs[0][0].text(xtop, ytop, 'rho={r:.3f}\n p={p:.3f}'.format(r=df_corrs['overl_dens_rho'][0],
                                                                        p=df_corrs['overl_dens_p'][0]),
                           fontsize=9)
            sns.scatterplot(data=df, x='feat_eff', y='clf_avg', ax=axs[0][1])
            ybottom, ytop = axs[0][1].get_ylim()  # return the current ylim
            xbottom, xtop = axs[0][1].get_xlim()  # return the current xlim
            axs[0][1].text(xtop, ytop,
                           'rho={r:.3f}\n p={p:.3f}'.format(r=df_corrs['feat_eff_rho'][0], p=df_corrs['feat_eff_p'][0]),
                           fontsize=9)
            sns.scatterplot(data=df, x='overl_hvol', y='clf_avg', ax=axs[0][2])
            ybottom, ytop = axs[0][2].get_ylim()  # return the current ylim
            xbottom, xtop = axs[0][2].get_xlim()  # return the current xlim
            axs[0][2].text(xtop, ytop,
                           'rho={r:.3f}\n p={p:.3f}'.format(r=df_corrs['overl_hvol_rho'][0], p=df_corrs['overl_hvol_p'][0]),
                           fontsize=9)
            sns.scatterplot(data=df, x='skew', y='clf_avg', ax=axs[1][0])
            ybottom, ytop = axs[1][0].get_ylim()  # return the current ylim
            xbottom, xtop = axs[1][0].get_xlim()  # return the current xlim
            axs[1][0].text(xtop, ytop,
                           'rho={r:.3f}\n p={p:.3f}'.format(r=df_corrs['skew_rho'][0], p=df_corrs['skew_p'][0]),
                           fontsize=9)
            sns.scatterplot(data=df, x='kurt', y='clf_avg', ax=axs[1][1])
            ybottom, ytop = axs[1][1].get_ylim()  # return the current ylim
            xbottom, xtop = axs[1][1].get_xlim()  # return the current xlim
            axs[1][1].text(xtop, ytop,
                           'rho={r:.3f}\n p={p:.3f}'.format(r=df_corrs['kurt_rho'][0], p=df_corrs['kurt_p'][0]),
                           fontsize=9)
            sns.scatterplot(data=df, x='sd', y='clf_avg', ax=axs[1][2])
            ybottom, ytop = axs[1][2].get_ylim()  # return the current ylim
            xbottom, xtop = axs[1][2].get_xlim()  # return the current xlim
            axs[1][2].text(xtop, ytop,
                           'rho={r:.3f}\n p={p:.3f}'.format(r=df_corrs['sd_rho'][0], p=df_corrs['sd_p'][0]),
                           fontsize=9)
            sns.scatterplot(data=df, x='corr_std', y='clf_avg', ax=axs[2][0])
            ybottom, ytop = axs[2][0].get_ylim()  # return the current ylim
            xbottom, xtop = axs[2][0].get_xlim()  # return the current xlim
            axs[2][0].text(xtop, ytop,
                           'rho={r:.3f}\n p={p:.3f}'.format(r=df_corrs['corr_std_rho'][0], p=df_corrs['corr_std_p'][0]),
                           fontsize=9)
            title = repr.name + ' ' + str(settings.num_of_datasets) + ' subsets ' + str(
                settings.no_of_classes) + ' classes'
            fig.suptitle(title, fontsize=20)
            fig.set_size_inches(12, 7)
            fig.subplots_adjust(wspace=0.6, hspace=0.5)
            # plt.show()
            plt.savefig(fname + '_plot.png')

    def correlate_metafeatures_to_scores_of_all_reprs(self):
        for name, repr in self._repr_core.reprs.items():
            fname = 'Spearman_' + repr.name + settings.file_suffix
            # if os.path.exists(fname):
            #     df_corrs = pd.read_pickle(fname)
            #     print("H0: There is no [monotonic] association between the two meta-feature and the classifier.")
            #     print(df_corrs)
            # else:
            dfs = self.get_descriptions_by_agr_clf(repr, score_agr='clf_avg')
            df_corrs = self._ml_core_stats.correlate_metafeatures_to_scores(dfs)
            print("H0: There is no [monotonic] association between the two meta-feature and the classifier.")
            print(df_corrs)
            df_corrs.to_pickle(fname)

            # plot metafeature to avg_cld score
            for df in dfs:
                fig, axs = plt.subplots(nrows=3, ncols=3)
                sns.lineplot(data=df, x='overl_dens', y='clf_avg', ax=axs[0][0])
                sns.lineplot(data=df, x='feat_eff', y='clf_avg', ax=axs[0][1])
                sns.lineplot(data=df, x='vor', y='clf_avg', ax=axs[0][2])
                sns.lineplot(data=df, x='skew', y='clf_avg', ax=axs[1][0])
                sns.lineplot(data=df, x='kurt', y='clf_avg', ax=axs[1][1])
                sns.lineplot(data=df, x='sd', y='clf_avg', ax=axs[1][2])
                sns.lineplot(data=df, x='corr_std', y='clf_avg', ax=axs[2][0])
                fig.suptitle(str(settings.num_of_datasets) + 'subsets', fontsize=20)
                fig.tight_layout()
                plt.show()

    def print_dataset_similarity(self):
        print("Similarities started!")
        np.set_printoptions(threshold=sys.maxsize)
        start = time()
        for name, repr in self._repr_core.reprs.items():
            df_list = repr.df_list
            sims = get_mean_cdist(df_list)
            # print(sims)
            # print(sims.reshape(600, 600))
        end = time()
        print(f"~Similarities ended! It took {end - start} seconds!")

    def investigate_similarity_threshold_w_noise(self):
        sim_dicts = []
        for name, repr in self._repr_core.reprs.items():
            temp_df = repr.repr_df.copy()
            first_sample = temp_df.drop(['species'], axis=1)
            for noise_lvl in np.arange(0, 1, 0.1):
                sim_dict = {}
                dfs = []
                dfs.append(first_sample)
                # apply noise
                noise_sample = first_sample.applymap(
                    lambda x: x + np.random.uniform(-1000, 1000) if np.random.uniform() < noise_lvl else x)
                dfs.append(noise_sample)
                most_dist, sim_array = get_mean_cdist(dfs)
                sim = sim_array[0][1]
                sim_dict['Noise Probability'] = noise_lvl
                sim_dict['Cosine_Distance'] = sim
                sim_dict['5% most distant'] = most_dist
                sim_dicts.append(sim_dict)
        df = pd.DataFrame(sim_dicts)
        sns.lineplot(data=df, x='Noise Probability', y='Cosine_Distance', label='mean')
        sns.lineplot(data=df, x='Noise Probability', y='5% most distant', label='5% most distant')
        plt.legend()
        plt.show()

    def investigate_similarity_threshold(self):
        """
        compare my dataset with a random one
        :return:
        """
        sim_dicts = []
        for name, repr in self._repr_core.reprs.items():
            sim_dict = {}
            temp_df = repr.repr_df.copy()
            first_sample = temp_df.drop(['species'], axis=1)
            random_df = pd.DataFrame(np.random.randint(-1000, 1000, size=first_sample.shape),
                                     columns=first_sample.columns)
            dfs = [first_sample, random_df]
            most_dist, sim = get_mean_cdist(dfs)

        #     sim_dict['Cosine_Distance'] = sim
        #     sim_dict['5% most distant'] = most_dist
        #     sim_dicts.append(sim_dict)
        # df = pd.DataFrame(sim_dicts)
        # sns.lineplot(data=df, x='Noise Probability', y='Cosine_Distance', label='mean')
        # sns.lineplot(data=df, x='Noise Probability', y='5% most distant', label='5% most distant')
        # plt.legend()
        # plt.show()

    def investigate_similarity_threshold2(self):
        for name, repr in self._repr_core.reprs.items():
            distances = []
            temp_df = repr.repr_df.copy()
            for indi, i in enumerate(range(0, 6000, 600)):
                for indj, j in enumerate(range(0, 6000, 600)):
                    if indj <= indi:
                        continue
                    dfs = []
                    dfs.append(temp_df[i:i + 600])
                    dfs.append(temp_df[j:j + 600])
                    most_dist, sim_array = get_mean_cdist(dfs)

                    # fracd  = abs(self._repr_descriptor.get_fractal_d(dfs[0]) - self._repr_descriptor.get_fractal_d(dfs[1]))
                    # distances[indi][indj] = fracd

                    # sim_array = get_hellinger_distance(dfs)
                    # distances.append(sim_array[0][1])
                    distances.append(most_dist)

        df = pd.DataFrame(data=distances)

        sns.displot(df, stat='probability', binwidth=0.001)
        plt.show()
        print(df.quantile(q=0.95))

    def investigate_repr(self):
        for name, repr in self._repr_core.reprs.items():
            temp_df = repr.repr_df.copy()
            print(temp_df.describe())
            hist = temp_df.hist()
            plt.show()
            # note: feature not all features have just positive values
            # and 0 is not a good value for noise
            for col in temp_df.keys():
                if col == 'species':
                    continue
                print(stats.shapiro(temp_df[col]))
            # none of the features is drawn from a normal distribution

            # K-S is not multivariate
