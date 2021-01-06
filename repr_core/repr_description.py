import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd
import settings
import logging

logging.basicConfig(filename='mylog.log', level=logging.INFO)


class ReprDescriptor:
    def __init__(self):
        self.dummy = None

    # 1
    def get_sd(self, df):
        """
        Computes pooled standard deviation
        :param df:
        :return:
        """
        feature_df = df.drop(df.columns[-1], axis=1)
        df_var = feature_df.var()
        df_len = len(feature_df)
        # calculate the pooled standard deviation
        s = 0
        for col in feature_df.keys():
            s += (df_len - 1) * df_var[col]
        ratio = s / (df_len * len(feature_df.columns))
        return np.sqrt(ratio)

    # 2
    def get_corr_stats(self, df):
        """
        Computes mean, standard deviation and the 3rd quantile of the correlation matrix
        :param df:
        :return:
        """
        feature_df = df.drop(df.columns[-1], axis=1)
        X = feature_df.corr().to_numpy()
        # get the upper triangular part of this matrix
        v = X[np.triu_indices(X.shape[0], k=1)]  # offset
        return np.nanmean(np.abs(v)), np.nanstd(abs(v)), np.nanquantile(v, .75)

    # 3
    def get_skew(self, df):
        """
        Computes skewness
        :param df:
        :return:
        """
        feature_df = df.drop(df.columns[-1], axis=1)
        return feature_df.skew().to_numpy().mean()

    # 4
    def get_kurt(self, df):
        """
        Computes kurtosis
        :param df:
        :return:
        """
        feature_df = df.drop(df.columns[-1], axis=1)
        return feature_df.kurtosis().to_numpy().mean()

    # 5
    def get_hx(self, df):
        """
        Computes entropy
        :param df:
        :return:
        """
        feature_df = df.drop(df.columns[-1], axis=1)
        values = []
        for col in feature_df.keys():
            value, counts = np.unique(feature_df[col].to_numpy(), return_counts=True)
            values.append(scipy.stats.entropy(counts))
        return np.array(values).mean()

    # 6 #TODO not clear how to compute for multiclass
    # def get_fd(X,Y):
    #    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    #    clf = LinearDiscriminantAnalysis(solver='eigen')
    #    clf.fit(X, Y)
    #    print(clf.coef_)
    #    arr_t = np.array(clf.coef_).T
    #    print(np.amax(arr_t,axis=1))

    def is_in_range(self, limits, row):
        result = True
        for feat in row.keys()[:-1]:
            for target in limits[feat].keys():
                if limits[feat][target]["min"] <= row[feat] <= limits[feat][target]["max"]:
                    result = result and True
                else:
                    result = result and False
        return str(result)

    def get_target_limits(self, df):
        """
        Finds per dimension the upper and lower values of a class
        :param df: input dataframe
        :return: dictionary like
        {column1:{class1:{max:maxvalue,min:minvalue},class2:{max:maxvalue,min:minvalue}}
         column2:{class1:{max:maxvalue,min:minvalue},class2:{max:maxvalue,min:minvalue}}}
        """
        d = {}
        for col in df.keys()[:-1]:
            e = {}
            for label in set(df[df.columns[-1]]):
                f = {}
                temp = df.loc[df[df.columns[-1]] == label]
                f["max"] = temp[col].max()
                f["min"] = temp[col].min()
                e[label] = f
            d[col] = e
        return d

    # 7.1
    def get_overlap_hypervolume(self, df):
        overlap_dims = []
        d = self.get_target_limits(df)
        temp_df = df.copy()
        temp_df['in_vor'] = temp_df.apply(lambda row: self.is_in_range(d, row), axis=1)
        # if there is an instance in overlap hypervolume
        if "True" in temp_df['in_vor'].values:
            invor_df = temp_df.loc[temp_df['in_vor'] == "True"]
            for col in invor_df.keys()[:-1]:
                if col == 'in_vor':
                    continue
                dim = temp_df.loc[temp_df['in_vor'] == "True"][col].max() - temp_df.loc[temp_df['in_vor'] == "True"][
                    col].min()
                overlap_dims.append(dim)
            result = np.prod(np.array(overlap_dims))
        else:
            result = 0
        return result

    # 7.2
    def get_overlap_density(self, df):
        vor = self.get_overlap_hypervolume(df)
        d = self.get_target_limits(df)
        temp_df = df.copy()
        temp_df['in_range'] = temp_df.apply(lambda row: self.is_in_range(d, row), axis=1)
        # if there is an instance in overlap hypervolume
        if "True" in temp_df['in_range'].values:
            invor_df = temp_df.loc[temp_df['in_range'] == "True"]
            result = len(invor_df.index) / vor
        else:
            result = 0
        return result

    def overlap_ratio(self, limits, row):
        """
        Counts the ratio of classes an instance participates in to the total of classes in the dataset
        :param limits: the class geometrical limits as computed by get_target_limits
        :param row: an instance. This method is used with pandas apply
        :return: ratio
        """
        overl_c = 0
        targets = list(limits.values())[0].keys()
        for target in targets:

            result = True
            for feat in row.keys()[:-1]:

                if limits[feat][target]["min"] <= row[feat] <= limits[feat][target]["max"]:
                    result = result and True
                else:
                    result = result and False
            if result:
                overl_c += 1
        return overl_c / len(targets)

    def get_overlap_ratio(self, df):
        d = self.get_target_limits(df)
        temp_df = df.copy()
        temp_df['over_r'] = temp_df.apply(lambda row: self.overlap_ratio(d, row), axis=1)
        return temp_df['over_r']

    # 7.3
    def get_overlap_ratio_stats(self, df):
        over_ratios = self.get_overlap_ratio(df)
        return over_ratios.mean(), over_ratios.std(), over_ratios.quantile(.75)

    # 8
    def get_feature_efficiency(self, df):
        """
        Counts the ratio of instances that do not participate in an overlap area
        :param df: input dataframe
        :return: ratio
        """
        d = self.get_target_limits(df)
        temp_df = df.copy()
        temp_df['over_r'] = temp_df.apply(lambda row: self.overlap_ratio(d, row), axis=1)
        return len(temp_df.loc[temp_df['over_r'] == 1 / settings.no_of_classes].index) / len(temp_df.index)

    # 9
    def get_fractal_d(self, df, threshold=0.9):
        feature_df = df.drop(df.columns[-1], axis=1)
        Z = feature_df.to_numpy()

        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k), axis=1)
            return len(np.where((S > 0) & (S < k * k))[0])

        Z = (Z < threshold)
        p = min(Z.shape)
        n = 2 ** np.floor(np.log(p) / np.log(2))
        n = int(np.log(n) / np.log(2))
        sizes = 2 ** np.arange(n, 1, -1)
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def describe_repr(self, df):
        dataset = df.reset_index(drop=True)
        d = {}
        dataset_chars = dict()
        dataset_chars['sd'] = self.get_sd
        dataset_chars['skew'] = self.get_skew
        dataset_chars['kurt'] = self.get_kurt
        dataset_chars['entropy'] = self.get_hx
        dataset_chars['overl_hvol'] = self.get_overlap_hypervolume
        dataset_chars['overl_dens'] = self.get_overlap_density
        dataset_chars['feat_eff'] = self.get_feature_efficiency
        dataset_chars['frac_dim'] = self.get_fractal_d

        for feat, func in dataset_chars.items():
            d[feat] = func(dataset)
            if feat == 'corr_mean':
                d['corr_mean^2'] = d[feat] * d[feat]
            elif feat == 'entropy':
                d['entropy^2'] = d[feat] * d[feat]
        d['overl_mean'], d['overl_std'], d['overl_q3'] = self.get_overlap_ratio_stats(dataset)
        d['corr_mean'], d['corr_std'], d['corr_q3'] = self.get_corr_stats(dataset)
        logging.info(d)
        return d

    def describe_repr_list(self, repr):
        """
        describe every subset of a repr
        :param repr:
        :return:
        """
        descriptions = [self.describe_repr(df) for df in repr.df_list]
        repr.descriptions = pd.DataFrame(descriptions)  # so this object will hold metrics for each df
        fname = repr.name + "_descriptions_" + settings.file_suffix
        repr.descriptions.to_pickle(fname)
