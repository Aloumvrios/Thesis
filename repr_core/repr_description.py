import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd
import settings
import logging

logging.basicConfig(filename='debug.log', level=logging.INFO)


class ReprDescriptor:
    def __init__(self):
        self.dummy = None

    # 1
    def get_sd(self, df):
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
    def get_corr_mean(self, df):
        feature_df = df.drop(df.columns[-1], axis=1)
        X = feature_df.corr().to_numpy()
        # get the upper triangular part of this matrix
        v = X[np.triu_indices(X.shape[0], k=1)]  # offset
        result = np.nanmean(np.abs(v))
        return result

    # 2.1
    def get_corr_std(self, df):
        feature_df = df.drop(df.columns[-1], axis=1)
        X = feature_df.corr().to_numpy()
        # get the upper triangular part of this matrix
        v = X[np.triu_indices(X.shape[0], k=1)]  # offset
        result = np.nanstd(abs(v))
        return result

    # 2.2
    def get_corr_q3(self, df):
        feature_df = df.drop(df.columns[-1], axis=1)
        X = feature_df.corr().to_numpy()
        # get the upper triangular part of this matrix
        v = X[np.triu_indices(X.shape[0], k=1)]  # offset
        result = np.nanquantile(v, .75)
        return result

    # 3
    def get_skew(self, df):
        feature_df = df.drop(df.columns[-1], axis=1)
        return feature_df.skew().to_numpy().mean()

    # 4
    def get_kurt(self, df):
        feature_df = df.drop(df.columns[-1], axis=1)
        return feature_df.kurtosis().to_numpy().mean()

    # 5
    def get_hx(self, df):
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
                    # print('target', target)
                    # print('min', limits[feat][target]["min"], type(limits[feat][target]["min"]))
                    # print('value', row[feat], type(row[feat]))
                    # print('max', limits[feat][target]["max"], type(limits[feat][target]["max"]))
        return str(result)

    def get_target_limits(self, df):
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
        # logging.info('for row {r}')
        overl_c = 0
        targets = list(limits.values())[0].keys()
        for target in targets:

            result = True
            for feat in row.keys()[:-1]:

                if limits[feat][target]["min"] <= row[feat] <= limits[feat][target]["max"]:
                    result = result and True
                    # print('TRUE ', limits[feat][target]["min"], row[feat], limits[feat][target]["max"])
                else:
                    result = result and False
                    # logging.info('for class {c}'.format(c=target))
                    # logging.info('for col {c}'.format(c=feat))
                    # logging.info('FALSE {min} {v} {max}'.format(min=limits[feat][target]["min"], v=row[feat],
                    #                                             max=limits[feat][target]["max"]))
            if result:
                overl_c += 1
        #         logging.info('overl_c {c}'.format(c=overl_c))
        # logging.info('for overl_c {c}, ratio is {r}'.format(c=overl_c, r=overl_c / len(targets)))

        return overl_c / len(targets)

    def get_overlap_ratio(self, df):
        d = self.get_target_limits(df)
        temp_df = df.copy()
        temp_df['over_r'] = temp_df.apply(lambda row: self.overlap_ratio(d, row), axis=1)
        return temp_df['over_r']

    # 7.3
    def get_overlap_ratio_mean(self, df):
        over_ratios = self.get_overlap_ratio(df)
        logging.info('~~over_r~~')
        logging.info('\n\t' + over_ratios.to_string().replace('\n', '\n\t'))
        logging.info('\n\t' + over_ratios.describe().to_string().replace('\n', '\n\t'))
        logging.info(over_ratios.mean())
        return over_ratios.mean()

    # 7.4
    def get_overlap_ratio_std(self, df):
        over_ratios = self.get_overlap_ratio(df)
        return over_ratios.std()

    # 7.5
    def get_overlap_ratio_q3(self, df):
        over_ratios = self.get_overlap_ratio(df)
        return over_ratios.quantile(.75)

    def get_overlap_ratio_stats(self, df):
        over_ratios = self.get_overlap_ratio(df)
        return over_ratios.mean(), over_ratios.std(), over_ratios.quantile(.75)

    # 8
    def get_feature_efficiency(self, df):
        d = self.get_target_limits(df)
        temp_df = df.copy()
        temp_df['in_range'] = temp_df.apply(lambda row: self.is_in_range(d, row), axis=1)
        return len(temp_df.loc[temp_df['in_range'] == "False"].index) / len(temp_df.index)

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
        dataset_chars['corr_mean'] = self.get_corr_mean
        dataset_chars['corr_std'] = self.get_corr_std
        dataset_chars['corr_q3'] = self.get_corr_q3
        dataset_chars['skew'] = self.get_skew
        dataset_chars['kurt'] = self.get_kurt
        dataset_chars['entropy'] = self.get_hx
        # dataset_chars['overl_mean'] = self.get_overlap_ratio_mean
        # dataset_chars['overl_std'] = self.get_overlap_ratio_std
        # dataset_chars['overl_q3'] = self.get_overlap_ratio_q3
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
        # print(repr.descriptions.head())
        fname = repr.name + "_descriptions_" + settings.file_suffix
        repr.descriptions.to_pickle(fname)
