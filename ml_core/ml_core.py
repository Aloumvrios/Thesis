from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
import settings
from sklearn.feature_selection import SelectFromModel
from scipy import stats
# from pyearth import Earth
import seaborn as sns
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

clfs = ["Nearest_Neighbors", "Linear_SVM", "RBF_SVM",
        "Gaussian_Process", "Decision_Tree", "Random_Forest",
        "Neural_Net", "AdaBoost", "Naive_Bayes", "QDA"
        ]
rgrs = ["LR", "AdaBoostR"
        # , "MARS"
        ]


class Classifiers:
    def __init__(self):
        """
        Constructor
        """
        self.available_classifiers = clfs

        self.classifiers = [KNeighborsClassifier(), SVC(kernel="linear"), SVC(),
                            GaussianProcessClassifier(1.0 * RBF(1.0)),
                            DecisionTreeClassifier(), RandomForestClassifier(),
                            # MLPClassifier(),
                            AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                            GaussianNB(), QuadraticDiscriminantAnalysis()]
        self.params = [
            # KNN
            {'n_neighbors': [3, 5, 10, 15],
             'weights': ['uniform', 'distance'],
             'metric': ['euclidean', 'manhattan']},
            # linear svm
            {'kernel': ['linear'],
             'C': [1, 100, 1000, 2000, 3000]},
            # rbf svm
            {'kernel': ['rbf'],
             'gamma': [1e-3, 1e-4, 1, 2],
             'C': [1, 10, 100, 1000]},
            # gaussian
            None,
            # DT
            {'max_depth': np.arange(1, 10),
             'min_samples_leaf': [1, 5, 10, 20, 50],
             'criterion': ['gini', 'entropy']},
            # Random forest
            {'n_estimators': [100, 200, 300],
             'max_depth': [5, 10, 20],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 5, 10, 20]},
            # MLP
            # {'activation': ['identity', 'logistic', 'tanh', 'relu'],
            #  'solver': ['lbfgs', 'sgd', 'adam'],
            #  # 'max_iter': [1000, 2000, 3000],
            #  'alpha': 10.0 ** -np.arange(1, 10),
            #  'hidden_layer_sizes': np.arange(10, 15)},
            # adaboost
            {'n_estimators': np.arange(1, 5),
             'base_estimator__max_depth': np.arange(1, 10),
             'base_estimator__min_samples_leaf': [1, 5, 10, 20, 50],
             'base_estimator__criterion': ['gini', 'entropy'],
             'algorithm': ['SAMME', 'SAMME.R']},
            # Gaussian NB
            None,
            # quadraticDA
            None]

    def classify_repr(self, dataset):
        # rescale each feature to 0,1
        rescaled_dataset = pd.DataFrame()
        for col in dataset.keys()[:-1]:
            maxv = dataset[col].max()
            minv = dataset[col].min()

            scale = maxv - minv
            rescaled_dataset[col] = (dataset[col] - minv) / scale if scale != 0 else dataset[col] - minv  # else 0

        X = rescaled_dataset.values
        Y = dataset[dataset.columns[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=7)
        d = {}
        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        for name, clf, param in zip(self.available_classifiers, self.classifiers, self.params):
            if param is not None and settings.fast_mode is False:
                search = GridSearchCV(clf, param, n_jobs=-1, cv=kfold, scoring='f1_weighted')
                search.fit(X_train, y_train)
                score = search.best_estimator_.score(X_test, y_test)
                # print(name)
                # print(search.best_params_)
            else:
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
            d[name] = score
        return d

    def classify_repr_list(self, repr):
        scores = [self.classify_repr(df) for df in repr.df_list]
        repr.scores = pd.DataFrame(scores)  # so this object will hold scores for each df
        fname = repr.name + '_scores_' + settings.file_suffix + '_' + settings.fast_mode_str
        repr.scores.to_pickle(fname)


class Regressors:
    def __init__(self):
        """
        Constructor
        """
        self.available_regressors = rgrs

        self.regressors = [LinearRegression(),
                           AdaBoostRegressor()
                           #  ,
                           # Earth(feature_importance_type='rss')
                           ]
        self.params = [
            # LR
            {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]},
            # AdaBoost
            {'n_estimators': [10, 100], 'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
             'loss': ['linear', 'square', 'exponential']
             #    ,
             # 'base_estimator__max_depth': np.arange(1, 10),
             # 'base_estimator__min_samples_leaf': [1, 5, 10, 20, 50],
             # 'base_estimator__criterion': ['mse', 'friedman_mse', 'mae']
             }
            # ,
            # # Earth
            # {'max_degree': [1, 2, 3, 4], 'allow_linear': [False, True],
            #  'penalty': [0., 1., 2., 3., 4., 5., 6.]}
        ]

    def apply_regression(self, dataset, certain_features):
        # rescale each feature to 0,1
        rescaled_dataset = pd.DataFrame()
        target = ""
        for col in dataset.keys():
            if col in clfs:
                target = col
                continue
            if certain_features is not None and col not in list(certain_features.index):
                continue
            maxv = dataset[col].max()
            minv = dataset[col].min()
            scale = maxv - minv
            rescaled_dataset[col] = (dataset[col] - minv) / scale
        X = rescaled_dataset.values
        Y = dataset[target].values
        # print(dataset[target].head())
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=7)
        d = {}
        # iterate over regressors and extract score
        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        scoring = 'neg_mean_squared_error'
        for name, rgr, param in zip(self.available_regressors, self.regressors, self.params):
            if param is not None and settings.fast_mode is False:
                search = GridSearchCV(rgr, param, n_jobs=-1, cv=kfold, scoring=scoring)
                search.fit(X_train, y_train)
                score = search.best_estimator_.score(X_test, y_test)
                # print("best estimator score ", abs(score))
                # print(search.best_params_)
            else:
                rgr.fit(X_train, y_train)
                score = rgr.score(X_test, y_test)
            d[name] = abs(score)
        d['Classifier'] = target
        d['avg_score'] = dataset[target].mean()
        d['var_score'] = dataset[target].var()
        if certain_features is not None:
            for feat in list(certain_features.index):
                d[feat] = rescaled_dataset[feat].mean()
        return d

    def apply_regr_to_list(self, df_list, certain_features=None):
        scores = [self.apply_regression(df, certain_features) for df in df_list]
        df_scores = pd.DataFrame(scores)
        return df_scores

    def get_n_important_features(self, dataset, n):
        # rescale each feature to 0,1
        rescaled_dataset = pd.DataFrame()
        target = ""
        for col in dataset.keys():
            if col in clfs:
                target = col
                continue
            maxv = dataset[col].max()
            minv = dataset[col].min()
            scale = maxv - minv
            rescaled_dataset[col] = (dataset[col] - minv) / scale
        X = rescaled_dataset
        Y = dataset[target]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=7)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, y_train)
        feature_importances = pd.DataFrame(rf.feature_importances_,
                                           index=X_train.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)
        return feature_importances[:n]

    def print_n_important_features_from_list(self, df_list, n):
        for df in df_list:
            important_features = self.get_n_important_features(df, n)
            clf = list(set(df.columns) & set(clfs))[0]
            print(clf)
            print(important_features)

    def apply_regr_to_list_on_n_important(self, df_list, n):
        n_scores = []
        for df in df_list:
            certain_features = self.get_n_important_features(df, n)
            n_d = self.apply_regression(df, certain_features)
            n_scores.append(n_d)
        n_df_scores = pd.DataFrame(n_scores)
        return n_df_scores

    def plot_most_important_feature_per_classifier(self, df_list):
        for df in df_list:
            certain_features = self.get_n_important_features(df, 1)
            most_important = list(certain_features.index)[0]
            clf = list(set(df.columns) & set(clfs))[0]
            # print(most_important, 'is most important for', clf)
            # sns.scatterplot(data=df, x=most_important, y=clf)
            sns.relplot(x=most_important, y=clf, kind="line", data=df)
            plt.show()


class Statistics:

    def correlate_metafeatures_to_scores(self, df_list, score_agr):
        corrs = []
        for df in df_list:
            clf = set(df.columns) & set(clfs)
            if clf:
                clf = list(set(df.columns.values) & set(clfs))[0]
            else:
                clf = score_agr
            metafeatures = list(set(df.columns.values) - {clf})
            d = {}
            d["Classifier"] = clf
            for metafeature in metafeatures:
                d[metafeature + "_rho"], d[metafeature + "_p"] = stats.spearmanr(df[metafeature], df[clf])
                if d[metafeature + "_p"] > settings.alpha:
                    d["Reject " + metafeature] = 'NO'
                elif d[metafeature + "_p"] <= settings.alpha:
                    d["Reject " + metafeature] = 'YES'
                else:
                    d["Reject " + metafeature] = 'NaN'
            corrs.append(d)
        df_corrs = pd.DataFrame(corrs)
        return df_corrs
