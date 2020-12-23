import os
import sys
from cmd import Cmd
from thesis_core.thesis_core import ThesisCore
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
import settings


class UserInterface(Cmd):
    def __init__(self):
        super(UserInterface, self).__init__()
        self.thesis_core = ThesisCore()
        # self.thesis_core.setup()

    def create_reprs(self, kmer):
        self.thesis_core.create_reprs(kmer)

    def split_all_datasets(self):
        self.thesis_core.split_all_datasets()

    def describe_reprs(self):
        self.thesis_core.describe_reprs()

    def classify_reprs(self):
        self.thesis_core.classify_reprs()

    def spearman_heatmap_metaf_and_scores(self):
        self.thesis_core.spearman_heatmap_metaf_and_scores()

    def regress_on_scores(self):
        self.thesis_core.regress_on_scores()

    def print_n_important_features(self, n):
        self.thesis_core.print_n_important_features(n)

    def regress_on_n_important_features(self, n):
        self.thesis_core.regress_on_n_important_features(n)

    def plot_most_important_feature_per_classifier(self):
        self.thesis_core.plot_most_important_feature_per_classifier()

    def correlate_metafeatures_to_scores_of_all_reprs(self):
        self.thesis_core.correlate_metafeatures_to_scores_of_all_reprs()

    def print_descriptions(self):
        self.thesis_core.print_descriptions()

    def print_dataset_similarity(self):
        self.thesis_core.print_dataset_similarity()

    def investigate_similarity_threshold_w_noise(self):
        self.thesis_core.investigate_similarity_threshold_w_noise()

    def investigate_similarity_threshold(self):
        self.thesis_core.investigate_similarity_threshold()

    def investigate_similarity_threshold2(self):
        self.thesis_core.investigate_similarity_threshold2()

    def read_a_pickle(self):
        # filename = 'FCGSR_descriptions_600_ds_2mer_3_distance'
        filename = 'Spearman_FCGSR_2mer_600_ds_3_distance'
        df = pd.read_pickle(filename)
        print(df)

    def investigate_repr(self):
        self.thesis_core.investigate_repr()

    def load_reprs(self):
        self.thesis_core.load_reprs()

    def load_repr_names(self):
        self.thesis_core.load_repr_names()

    def read_split_correlate(self):
        self.thesis_core.read_split_correlate()


if __name__ == '__main__':
    ui = UserInterface()                    # here
    ui.load_repr_names()
    ui.read_split_correlate()
    # ui.create_reprs(settings.kmer)          # here
    # ui.load_reprs()
    # ui.split_all_datasets()                     # here
    # ui.correlate_metafeatures_to_scores_of_all_reprs()   # here
    # ------------------------------------------------------------------
    # ui.describe_reprs()
    # ui.classify_reprs(fast_mode=True)
    # ui.print_descriptions()
    # ui.print_dataset_similarity()
    # ui.spearman_heatmap_metaf_and_scores(fast_mode=False)
    # ui.regress_on_scores(fast_mode=False)
    # ui.print_n_important_features(settings.num_of_important_features)
    # ui.regress_on_n_important_features(False, settings.num_of_important_features)
    # ui.plot_most_important_feature_per_classifier()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # ui.investigate_similarity_threshold_w_noise()
    # ui.investigate_similarity_threshold()
    # ui.investigate_similarity_threshold2()
    # ui.investigate_repr()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ui.read_a_pickle()
    sys.exit(0)
