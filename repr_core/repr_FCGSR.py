import numpy as np
import scipy.signal
import scipy.stats
from repr_core import repr_utils
import arff


class FCGSR():
    def __init__(self):
        self.name = 'FCGSR'
        self.repr_df = None
        self.df_list = None
        self.descriptions = None
        self.scores = None

    def set_repr_df_list(self, df_list):
        self.df_list = df_list

    # fuzzy c-means (FCM) method of clustering
    # ???
    # mahalanobis distance
    # ???
    # maximum number of peaks divided by the length of the sequence

    def get_max_peaks(self, signal):
        peaks, _ = scipy.signal.find_peaks(signal)
        return len(peaks) / len(signal)

    # average(mean)
    def get_average(self, signal):
        return np.mean(signal)

    # median
    def get_median(self, signal):
        return np.median(signal)

    # standard deviation
    def get_st_dev(self, signal):
        return np.std(signal)

    # variance
    def get_variance(self, signal):
        return np.var(signal)

    # energy
    def get_energy(self, signal):
        return np.sum(np.square(signal))

    # root mean square
    def get_rms(self, signal):
        e = self.get_energy(signal)
        return np.sqrt(e / len(signal))

    # Smean
    def get_Smean(self, signal):
        var = self.get_variance(signal)
        return np.mean((signal / np.sqrt(var)) - np.mean(signal / np.sqrt(var)))

    # Frequency Domain
    # mean power spectral density
    def get_mean_psd(self, signal):
        _, Pxx_den = scipy.signal.welch(signal)
        return np.mean(Pxx_den)

    # power root mean square
    def get_prms(self, signal):
        return np.real(np.sqrt(np.sum(np.fft.fft(np.square(signal)))))

    def seq_to_sig(self, sequence, k):
        p = repr_utils.probabilities(sequence, k)
        kmers = repr_utils.split_to_kmer(sequence, k)
        signal = [p[key] for key in kmers]
        return signal

    def construct_repr(self, df, kmer):
        """
        :param kmer: how do you want to split the sequence
        :param df: input dataframe
        :return: dataframe with extracted features plus target label
        """
        # features dict
        features = dict()
        features['average'] = self.get_average
        features['median'] = self.get_median
        features['standard dev'] = self.get_st_dev
        features['variance'] = self.get_variance
        features['energy'] = self.get_energy
        features['rms'] = self.get_rms
        features['Smean'] = self.get_Smean
        features['max peaks'] = self.get_max_peaks
        features['mean psd'] = self.get_mean_psd
        features['prms'] = self.get_prms

        # keep in a new dataframe just the sequence and the species label
        df2 = df[['sequence', 'species']].copy()

        # convert the sequence to kmer
        df2['kmers'] = df2.apply(lambda row: self.seq_to_sig(row.sequence, kmer), axis=1)

        # extract the features from each row-sequence and add feature columns
        for feat, func in features.items():
            df2[feat] = df2.apply(lambda row: func(row.kmers), axis=1)
        self.repr_df = df2.drop(['sequence', 'kmers'], axis=1)
        if self.repr_df.columns[0] =='species':
            self.repr_df['temp'] = self.repr_df[self.repr_df.columns[0]]
            self.repr_df[self.repr_df.columns[0]] = self.repr_df[self.repr_df.columns[-2]]
            self.repr_df[self.repr_df.columns[-2]] = self.repr_df['temp']
            self.repr_df = self.repr_df.drop(columns=['temp'])  # now each file-df has the first column as species


        attributes = [(c, 'NUMERIC') for c in self.repr_df.columns.values[:-1]]
        t = self.repr_df.columns[-1]
        attributes += [('target', self.repr_df[t].unique().astype(str).tolist())]
        data = [self.repr_df.loc[i].values[:-1].tolist() + [self.repr_df[t].loc[i]] for i in range(self.repr_df.shape[0])]

        arff_dic = {
            'attributes': attributes,
            'data': data,
            'relation': 'myRel',
            'description': ''
        }

        with open("myfile.arff", "w", encoding="utf8") as f:
            arff.dump(arff_dic, f)


