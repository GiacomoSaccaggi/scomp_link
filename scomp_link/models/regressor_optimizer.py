# -*- coding: utf-8 -*-
"""

██████╗ ███████╗ ██████╗ ██████╗ ███████╗ ██████╗ ██████╗ █████╗ ██████╗ 
██╔══██╗██╔════╝██╔════╝ ██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗
██████╔╝█████╗  ██║  ██╗ ██████╔╝█████╗  ╚█████╗ ╚█████╗ ██║  ██║██████╔╝
██╔══██╗██╔══╝  ██║  ╚██╗██╔══██╗██╔══╝   ╚═══██╗ ╚═══██╗██║  ██║██╔══██╗
██║  ██║███████╗╚██████╔╝██║  ██║███████╗██████╔╝██████╔╝╚█████╔╝██║  ██║
╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═════╝  ╚════╝ ╚═╝  ╚═╝

 █████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗███████╗██████╗ 
██╔══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚════██║██╔════╝██╔══██╗
██║  ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔═╝█████╗  ██████╔╝
██║  ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║██╔══╝  ██╔══╝  ██╔══██╗
╚█████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗███████╗██║  ██║
 ╚════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝

"""
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.svm import SVR
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor, Lasso, LinearRegression, ElasticNet
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler



class Boruta(BaseEstimator, TransformerMixin):
    """
       Improved Python implementation of the Boruta algorithm for feature selection.
        Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

        Boruta is an all relevant feature selection method, while most other are
        minimal optimal; this means it tries to find all features carrying
        information usable for prediction, rather than finding a possibly compact
        subset of features on which some classifier has a minimal error.

        Why bother with all relevant feature selection?
        When you try to understand the phenomenon that made your data, you should
        care about all factors that contribute to it, not just the bluntest signs
        of it in context of your methodology (yes, minimal optimal set of features
        by definition depends on your classifier choice).


       Dependencies:
       - numpy
       - scipy
       - pandas
       - sklearn
       - matplotlib

       Parameters:
       - estimator: Estimator object with a 'fit' method and 'feature_importances_' attribute
       - n_estimators: Number of estimators in the ensemble method or 'auto'
       - perc: Percentile for feature threshold comparison (default = 100)
       - alpha: Level for corrected p-value rejection in corrections (default = 0.05)
       - two_step: Boolean for two-step correction (default = True)
       - max_iter: Maximum iterations to perform (default = 100)
       - random_state: RandomState instance or seed (default=None)
       - verbose: Verbosity level (0, 1, or 2) for output control (default=0)

       Methods:
       - fit: Fits Boruta feature selection with the provided estimator
       - transform: Reduces input to selected Boruta features
       - fit_transform: Fits Boruta and reduces input to selected features
       """
    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05,
                 two_step=True, max_iter=100, random_state=None, verbose=1):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fits the Boruta feature selection with the provided estimator.

        Parameters:
        - X: Training input samples
        - y: Target values
        """

        return self._fit(X, y)

    def transform(self, X, weak=False):
        """
        Reduces the input X to the features selected by Boruta.

        Parameters:
        - X: Training input samples
        - weak: Boolean for considering tentative features (default=False)
        """

        return self._transform(X, weak)

    def fit_transform(self, X, y, weak=False):
        """
        Fits Boruta and reduces the input X to selected features.

        Parameters:
        - X: Training input samples
        - y: Target values
        - weak: Boolean for considering tentative features (default=False)
        """

        self._fit(X, y)
        return self._transform(X, weak)

    def _fit(self, X, y):
        # check input params
        self._check_params(X, y)
        self.random_state = check_random_state(self.random_state)
        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype = int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype = int)
        # these record the history of the iterations
        imp_history = np.zeros(n_feat, dtype = float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators = self.n_estimators)

        # main feature selection loop
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            if self.n_estimators == 'auto':
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator.set_params(n_estimators = n_tree)

            # make sure we start with a new tree in each iteration
            self.estimator.set_params(random_state = self.random_state)

            # add shadow attributes, shuffle them and train estimator, get imps
            cur_imp = self._add_shadows_get_imps(X, y, dec_reg)

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(cur_imp[1], self.perc)

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, cur_imp[0]))

            # register which feature is more imp than the max of shadows
            hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)

            # based on hit_reg we check if a feature is doing better than
            # expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, _iter)

            # print out confirmed features
            if self.verbose > 0 and _iter < self.max_iter:
                self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1

        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        # ignore the first row of zeros
        tentative_median = np.median(imp_history[1:, tentative], axis = 0)
        # which tentative to keep
        tentative_confirmed = np.where(tentative_median
                                       > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype = bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype = bool)
        self.support_weak_[tentative] = 1

        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype = int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[1:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
            # calculate ranks in each iteration, then median of ranks across feats
            iter_ranks = self._nanrankdata(imp_history_rejected, axis = 1)
            rank_medians = np.nanmedian(iter_ranks, axis = 0)
            ranks = self._nanrankdata(rank_medians, axis = 0)

            # set smallest rank to 3 if there are tentative feats
            if tentative.shape[0] > 0:
                ranks = ranks - np.min(ranks) + 3
            else:
                # and 2 otherwise
                ranks = ranks - np.min(ranks) + 2
            self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype = bool)

        # notify user
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)
        return self

    def _transform(self, X, weak=False):
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')

        if weak:
            X = X[:, self.support_ + self.support_weak_]
        else:
            X = X[:, self.support_]
        return X

    def _get_tree_num(self, n_feat):
        depth = self.estimator.get_params()['max_depth']
        if depth == None:
            depth = 10
        # how many times a feature should be considered on average
        f_repr = 100
        # n_feat * 2 because the training matrix is extended with n shadow features
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _get_imp(self, X, y):
        try:
            self.estimator.fit(X, y)
        except Exception as e:
            raise ValueError('Please check your X and y variable. The provided'
                             'estimator cannot be fitted to your data.\n' + str(e))
        try:
            imp = self.estimator.feature_importances_
        except Exception:
            raise ValueError('Only methods with feature_importance_ attribute '
                             'are currently supported in  Boruta.')
        return imp

    def _get_shuffle(self, seq):
        self.random_state.shuffle(seq)
        return seq

    def _add_shadows_get_imps(self, X, y, dec_reg):
        # find features that are tentative still
        x_cur_ind = np.where(dec_reg >= 0)[0]
        x_cur = np.copy(X[:, x_cur_ind])
        x_cur_w = x_cur.shape[1]
        # deep copy the matrix for the shadow matrix
        x_sha = np.copy(x_cur)
        # make sure there's at least 5 columns in the shadow matrix for
        while (x_sha.shape[1] < 5):
            x_sha = np.hstack((x_sha, x_sha))
        # shuffle xSha
        x_sha = np.apply_along_axis(self._get_shuffle, 0, x_sha)
        # get importance of the merged matrix
        imp = self._get_imp(np.hstack((x_cur, x_sha)), y)
        # separate importances of real and shadow features
        imp_sha = imp[x_cur_w:]
        imp_real = np.zeros(X.shape[1])
        imp_real[:] = np.nan
        imp_real[x_cur_ind] = imp[:x_cur_w]
        return imp_real, imp_sha

    def _assign_hits(self, hit_reg, cur_imp, imp_sha_max):
        # register hits for features that did better than the best of shadows
        cur_imp_no_nan = cur_imp[0]
        cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
        hits = np.where(cur_imp_no_nan > imp_sha_max)[0]
        hit_reg[hits] += 1
        return hit_reg

    def _do_tests(self, dec_reg, hit_reg, _iter):
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, .5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, _iter, .5).flatten()

        if self.two_step:
            # two step multicor process
            # first we correct for testing several features in each round using FDR
            to_accept = self._fdrcorrection(to_accept_ps, alpha = self.alpha)[0]
            to_reject = self._fdrcorrection(to_reject_ps, alpha = self.alpha)[0]

            # second we correct for testing the same feature over and over again
            # using bonferroni
            to_accept2 = to_accept_ps <= self.alpha / float(_iter)
            to_reject2 = to_reject_ps <= self.alpha / float(_iter)

            # combine the two multi corrections, and get indexes
            to_accept *= to_accept2
            to_reject *= to_reject2
        else:
            # as in th original Boruta, we simply do bonferroni correction
            # with the total n_feat in each iteration
            to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))
            to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))

        # find features which are 0 and have been rejected or accepted
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1
        return dec_reg

    def _fdrcorrection(self, pvals, alpha=0.05):
        """
        Benjamini/Hochberg p-value correction for false discovery rate, from
        statsmodels package. Included here for decoupling dependency on statsmodels.

        Parameters
        ----------
        pvals : array_like
            set of p-values of the individual tests.
        alpha : float
            error rate

        Returns
        -------
        rejected : array, bool
            True if a hypothesis is rejected, False if not
        pvalue-corrected : array
            pvalues adjusted for multiple hypothesis testing to limit FDR
        """
        pvals = np.asarray(pvals)
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
        nobs = len(pvals_sorted)
        ecdffactor = np.arange(1, nobs + 1) / float(nobs)

        reject = pvals_sorted <= ecdffactor * alpha
        if reject.any():
            rejectmax = max(np.nonzero(reject)[0])
            reject[:rejectmax] = True

        pvals_corrected_raw = pvals_sorted / ecdffactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1
        # reorder p-values and rejection mask to original order of pvals
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_

    def _nanrankdata(self, X, axis=1):
        """
        Replaces bottleneck's nanrankdata with scipy and numpy alternative.
        """
        ranks = sp.stats.mstats.rankdata(X, axis = axis)
        ranks[np.isnan(X)] = np.nan
        return ranks

    def _check_params(self, X, y):
        """
        Check hyperparameters as well as X and y before proceeding with fit.
        """
        # check X and y are consistent len, X is Array and y is column
        X, y = check_X_y(X, y)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError('The percentile should be between 0 and 100.')

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('Alpha should be between 0 and 1.')

    def _print_results(self, dec_reg, _iter, flag):
        n_iter = str(_iter) + ' / ' + str(self.max_iter)
        n_confirmed = np.where(dec_reg == 1)[0].shape[0]
        n_rejected = np.where(dec_reg == -1)[0].shape[0]
        cols = ['Iteration: ', 'Confirmed: ', 'Tentative: ', 'Rejected: ']

        # still in feature selection
        if flag == 0:
            n_tentative = np.where(dec_reg == 0)[0].shape[0]
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            if self.verbose == 1:
                output = cols[0] + n_iter
            elif self.verbose > 1:
                output = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])

        # Boruta finished running and tentatives have been filtered
        else:
            n_tentative = np.sum(self.support_weak_)
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            result = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])
            output = "\n\n Boruta finished running.\n\n" + result
        print(output)


class RegressorOptimizer:
    """
    RegressorOptimizer is a class designed to optimize regression models by performing feature selection,
    hyperparameter tuning, and evaluating multiple regression algorithms.

    Dependencies:
    - numpy
    - pandas
    - Boruta
    - sklearn.ensemble.RandomForestRegressor
    - sklearn.preprocessing.OneHotEncoder
    - sklearn.model_selection.GridSearchCV
    - sklearn.pipeline.Pipeline
    - matplotlib.pyplot
    - scipy.optimize.curve_fit

    Parameters:
    - df: DataFrame containing the data
    - y_col: Name of the target column
    - x_cols: List of predictive columns
    - x_complexity_col: Column indicating the complexity of the predictor
    - select_features: Flag to determine whether to perform feature selection (default=True)

    Methods:
    - __init__: Initializes the RegressorOptimizer object
    - select_features: Preprocesses data and selects significant features using Boruta algorithm
    - estimate_optimization_time: Estimates the time required for optimizing models with given hyperparameters
    - select_hyperparameters: Performs grid search to find the best hyperparameters for a given regressor
    - test_models_regression: Tests various regression models using optimized hyperparameters
    - grafico_fit_con_errore: Generates a graph displaying observed versus predicted values and residuals for a specific model
    """
    def __init__(self, df, y_col, x_cols, x_complexity_col, models_to_test, select_features=False):
        # Dividi il dataset in features (X) e target (y)
        if select_features:
            self.X, self.y, self.dropped_columns = self.select_features(df, y_col, x_cols)
            col_dropped = "\n\t".join(self.dropped_columns)
            print(f'Columns dropped: \n\t{col_dropped}')
        else:
            self.X = df[x_cols]
            self.y = df[y_col]
        self.x_graph_col = df[x_complexity_col]

        # Identifica le colonne categoriche, numeriche e binarie e fa il preproc adeguato
        self.categorical_cols = self.X.select_dtypes(include = ['object']).columns.tolist()
        self.binary_cols = self.X.select_dtypes(include = ['bool']).columns.tolist()
        self.numeric_cols = self.X.select_dtypes(include = ['int64', 'float64']).columns.tolist()

        # Crea i transformer per le colonne categoriche e binarie
        self.categorical_transformer = OneHotEncoder()
        self.binary_transformer = OneHotEncoder(drop='if_binary', sparse_output=False)
        self.numeric_transformer = StandardScaler()

        # Combiniamo i transformer
        self.preprocessor = ColumnTransformer(
            transformers = [
                ('categorical', self.categorical_transformer, self.categorical_cols),
                ('binary', self.binary_transformer, self.binary_cols),
                ('numeric', self.numeric_transformer, self.numeric_cols)
            ]
        )

        # Lista di modelli con relativi parametri da testare per regressione
        self.models_to_test = models_to_test

        # Dizionario per salvare i risultati dei modelli
        self.model_results = {}

        # Dividi il dataset in set di training e test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)
        # Definisci la strategia di cross-validation
        self.cv_strategy = KFold(n_splits=5, shuffle=True, random_state = 15121)

    @staticmethod
    def select_features(df, y_col, x_cols):
        """
        Descrizione della funzione:
        Questa funzione preprocessa il dataset, crea dummy per le variabili categoriche e binarie, utilizza un Random Forest per valutare
        l'importanza delle features e applica l'algoritmo Boruta per selezionare le variabili più significative rispetto al target.

        L'algoritmo Boruta è un metodo wrapper per la selezione delle feature che si basa sull'utilizzo di un classificatore
        (in questo caso, un regressore Random Forest) per identificare le variabili più importanti. Utilizza iterazioni e confronti
        tra feature 'reali' e feature 'ombre' generate casualmente per determinare l'importanza delle variabili e selezionare quelle
        rilevanti per il modello.

        Pacchetti da cui dipende la funzione sono:
        numpy, pandas,  Boruta, sklearn.ensemble.RandomForestRegressor, sklearn.preprocessing.OneHotEncoder

        PARAMETRI:
         1. df: DataFrame contenente i dati
         2. y_col: Nome della colonna target
         3. x_cols: Lista delle colonne predittive

        Esempio di utilizzo:

        df = ...  # Carica il DataFrame
        y_column = 'target_column'
        X_columns = ['feature_1', 'feature_2', ...]

        Final_X, Y = RegressorOptimizer.select_features(df, y_column, X_columns)


        \n\n\033[100mBest Practice: mettere sempre la descrizione di una funzione così da poter sempre capire a cosa serve\033[0m
        """


        columns_to_drop = []
        binary_cols = df.select_dtypes(include = ['bool']).columns.tolist()
        ############ Valutazione colonne BOOLEANE
        if len(binary_cols) > 1:
            # Creazione di DataFrame per la correlazione tetracorica
            tetracorrelation_matrix = pd.DataFrame(index = binary_cols, columns = binary_cols)
            fisher_matrix = pd.DataFrame(index = binary_cols, columns = binary_cols)
            for var1, var2 in list(product(binary_cols, binary_cols)):
                if var1 != var2:
                    a = df[(df[var1]) & (df[var2])].shape[0]
                    b = df[(df[var1]) & (~df[var2])].shape[0]
                    c = df[(~df[var1]) & (df[var2])].shape[0]
                    d = df[(~df[var1]) & (~df[var2])].shape[0]

                    try:
                        r_xy = (a * d - b * c) / ((a + b) * (c + d) * (a + c) * (b + d)) ** (1 / 2)
                    except:
                        r_xy = -1 if a == 0 and c == 0 else 1
                    fisher_p_value = fisher_exact([[a, b], [c, d]])[1]

                    tetracorrelation_matrix.loc[var1, var2] = r_xy
                    fisher_matrix.loc[var1, var2] = fisher_p_value
                else:
                    tetracorrelation_matrix.loc[var1, var2] = 1
                    fisher_matrix.loc[var1, var2] = 1
            for [col_name1, tetracorrelation], [col_name2, fisher_index] in zip(tetracorrelation_matrix.iterrows(),
                                                                                fisher_matrix.iterrows()):
                if col_name1 == col_name2 \
                        and np.mean(fisher_index) == 1 \
                        and (np.mean(tetracorrelation) == 1 or np.mean(tetracorrelation) == -1)\
                        and col_name1 != y_col:
                    columns_to_drop.append(col_name1)
        elif len(binary_cols)==1 and sum(df[binary_cols[0]]) == len(df[binary_cols[0]]):
            columns_to_drop.append(binary_cols[0])

        # Dividi il dataset in features (X) e target (y)
        X = df[[i for i in x_cols if i not in columns_to_drop]]
        y = df[y_col]

        # Identifica le colonne numeriche, categoriche e binarie
        categorical_cols = X.select_dtypes(include = ['object']).columns.tolist()
        print(categorical_cols)
        binary_cols = X.select_dtypes(include = ['bool']).columns.tolist()
        numeric_cols = X.select_dtypes(include = ['int64', 'float64']).columns.tolist()


        # Crea le dummy per le colonne categoriche e binarie
        categorical_transformer = OneHotEncoder(drop = 'first', sparse_output = False)
        X_categorical = categorical_transformer.fit_transform(X[categorical_cols])
        X_binary = X[binary_cols].astype(int)  # Le colonne booleane sono già binarie

        # Concatena le colonne dummy alle features esistenti
        X_encoded = np.concatenate([X_categorical, X_binary, X[numeric_cols]], axis = 1)

        # Creazione del DataFrame con le nuove features dummy
        new_columns = list(categorical_transformer.get_feature_names_out(categorical_cols)) + binary_cols + numeric_cols
        X_encoded_df = pd.DataFrame(X_encoded, columns = new_columns)

        # Unione dei dataframe con le nuove features dummy
        X_processed = pd.concat([X.drop(columns = categorical_cols + binary_cols + numeric_cols), X_encoded_df],
                                axis = 1)

        # Utilizza un algoritmo Random Forest per valutare l'importanza delle features
        rf = RandomForestRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 5, min_samples_leaf = 2,
                                   random_state = 1)
        rf.fit(X_processed, y)

        # Utilizza Boruta per selezionare le feature più importanti
        boruta_selector =  Boruta(rf, n_estimators = 'auto', verbose = 0, random_state = 1)
        boruta_selector.fit(X_processed.values, y.values)

        # Ottieni l'elenco delle colonne selezionate da Boruta
        selected_columns = X_processed.columns[boruta_selector.support_].tolist()
        for col in [i for i in x_cols if i not in columns_to_drop]:
            if True not in [col == c[:len(col)] for c in selected_columns]:
                columns_to_drop.append(col)
        return df[[i for i in x_cols if i not in columns_to_drop]], y, columns_to_drop

    def estimate_optimization_time(self, time_per_combination):
        """
        Descrizione della funzione:
        Questa funzione stima il tempo totale necessario per ottimizzare i modelli di regressione usando una combinazione di iperparametri.

        Pacchetti da cui dipende la funzione sono:
        Nessun pacchetto esterno.

        PARAMETRI:
        1. time_per_combination: Tempo stimato per valutare una singola combinazione di iperparametri

        Esempio di utilizzo:

        optimizer = RegressorOptimizer(...)
        optimizer.estimate_optimization_time(5)  # Stima il tempo con 5 secondi per combinazione


        \n\n\033[100mBest Practice: mettere sempre la descrizione di una funzione così da poter sempre capire a cosa serve\033[0m
        """
        total_time = 0

        for model_name, model_data in self.models_to_test.items():
            params_grid = model_data.get('params_grid', {})
            num_combinations = 1

            for param_name, param_values in params_grid.items():
                num_combinations *= len(param_values)

            model_time = num_combinations * time_per_combination
            total_time += model_time

            print(f"Estimated time for optimizing {model_name}: {round(model_time/60/60, 2)} hour")

        print(f"Total estimated time for optimization: {round(total_time/60/60, 2)} hour")

    def select_hyperparameters(self, regressor, params_grid):
        """
        Descrizione della funzione:
        Questa funzione esegue una ricerca dei migliori iperparametri per il regressore utilizzando una Grid Search con Cross-Validation.

        Pacchetti da cui dipende la funzione sono:
        sklearn.pipeline.Pipeline, sklearn.model_selection.GridSearchCV

        PARAMETRI:
         1. regressor: Il regressore per il quale si vogliono ottimizzare gli iperparametri
         2. params_grid: Dizionario contenente la griglia degli iperparametri da testare

        Esempio di utilizzo:

        regressor_to_optimize = RandomForestRegressor()
        parameters_to_test = {'n_estimators': [100, 200], 'max_depth': [5, 10]}

        optimizer = RegressorOptimizer(...)
        best_regressor, best_params = optimizer.select_hyperparameters(regressor_to_optimize, parameters_to_test)


        \n\n\033[100mBest Practice: mettere sempre la descrizione di una funzione così da poter sempre capire a cosa serve\033[0m
        """
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', regressor)
        ])

        grid_search = GridSearchCV(pipeline, {'regressor__'+k: v for k,v in params_grid.items()}, cv=self.cv_strategy, scoring='neg_mean_squared_error', verbose = True)
        grid_search.fit(self.X_train, self.y_train)

        best_regressor = grid_search.best_estimator_
        best_params = grid_search.best_params_

        return best_regressor, best_params



    def test_models_regression(self):
        """
        Descrizione della funzione:
        Questa funzione testa diversi modelli di regressione con gli iperparametri ottimizzati e valuta le prestazioni
        su un set di dati di test.

        Pacchetti da cui dipende la funzione sono:
        Nessun pacchetto esterno.

        PARAMETRI:
        Nessun parametro esplicito, ma utilizza gli attributi della classe `RegressorOptimizer`

        Esempio di utilizzo:

        optimizer = RegressorOptimizer(...)
        optimizer.test_models_regression()


        \n\n\033[100mBest Practice: mettere sempre la descrizione di una funzione così da poter sempre capire a cosa serve\033[0m
        """
        for model_name, model_data in self.models_to_test.items():
            print(f"\n\t... Testing {model_name}:\n\t", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"\n\t... Optimizing HyperParameters ...\n\t")
            # Ottieni il modello e la griglia dei parametri da testare
            model = model_data['model']
            params_grid = model_data['params_grid']

            # Seleziona i migliori iperparametri
            best_regressor, best_params = self.select_hyperparameters( model, params_grid)
            print(f"\n\t", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"\n\t... Training the Final Model ...\n\t")

            # Fai il fit del regressore migliore sul set di training
            best_regressor.fit(self.X_train, self.y_train)

            # Valuta il regressore migliore sul set di test
            y_pred = best_regressor.predict(self.X_test)


            # Salva i risultati nel dizionario
            self.model_results[model_name] = {
                'Model': best_regressor,
                'Params': best_params,
                'Fitted_Test': y_pred,
                'True_Test': self.y_test,
            }
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"\n\t... Finished ...\n\t")




    def grafico_fit_con_errore(self, model_name, h=16 ,w=9):
        """
        Descrizione della funzione:
        Questa funzione restituisce grafici sui fit relativamente ad un regressore.

        Pacchetti da cui dipende la funzione sono:
        numpy, matplotlib.pyplot, scipy.optimize.curve_fit

        PARAMETRI:
         1. x: regressore
         2. y: valori osservati
         3. fitted: valori previsti
         4. h: altezza dell'immagine finale
         5. w: larghezza dell'immagine finale

        Esempio di utilizzo:

        epon = lambda x, a, b, c: a*np.exp(b*x)+c
        a, b, c = [0.4, 0.3, 0.2]
        x = np.linspace(1, 20, 2000)
        p = [1.36, 1.21]
        noise = np.random.randn(2000)
        y = epon(x, a, b, c) + noise * 0.4 * x
        popt, pcov = curve_fit(epon, x, y)
        fitted = epon(x, popt[0], popt[1], popt[2])

        grafico_fit_con_errore(x, y, fitted, 8, 6)


        \n\n\033[100mBest Practice: mettere sempre la descrizione di una funzione così da poter sempre capire a cosa serve\033[0m
        """

        x = self.x_graph_col
        y = self.y
        if len(self.model_results.keys()) == 0:
            raise Exception('You have first to run test_models_regression function!')
        fitted = self.model_results[model_name]['Model'].predict(self.X)


        fig, ax = plt.subplots(figsize=[h,w],nrows=3,ncols=1)
        ax[0].plot(x, y, '.r')
        ax[0].grid(True)
        ax[0].plot(x, fitted, 'sb-', ms=1)
        ax[0].fill_between(x,fitted + np.sqrt(np.var(fitted)/len(fitted)), fitted - np.sqrt(np.var(fitted)/len(fitted)),color='r',alpha=.2)
        ax[0].fill_between(x,fitted + 1.96 * np.sqrt(np.var(fitted)/len(fitted)), fitted - 1.96 * np.sqrt(np.var(fitted)/len(fitted)),color='r',alpha=.1)
        ax[0].set_xticklabels([])
        ax[1].bar(x, fitted-y, color='b',width=0.1)
        ax[1].grid(True)
        ax[1].plot(np.zeros(21),'sr-', ms=1)
        ax[1].set_xticklabels([])
        difference= fitted-y
        intrange=np.arange(0,len(x),int(len(x)*0.1))
        media = []
        varianza = []
        laX = [0]
        for i, val in enumerate(intrange):
            if i != (len(intrange) - 1):
                media.append(difference[val:intrange[i + 1]].mean())
                varianza.append(np.var(difference[val:intrange[i + 1]]))
                laX.append(x[intrange[i + 1]])
        h = int(len(x) * 0.001)
        if h < 1:
            h = 1

        h = (laX[1] + laX[0]) / 4
        ax[2].plot(laX, (np.ones(len(media) + 1) * sum(media) / len(media)), 'r-')
        for i in range(len(media)):
            ax[2].plot((laX[i + 1] + laX[i]) / 2, media[i], 'ob')
            ax[2].plot([(laX[i + 1] + laX[i]) / 2, (laX[i + 1] + laX[i]) / 2],
                            [media[i] + 1.96 * np.sqrt(varianza[i]), media[i] - 1.96 * np.sqrt(varianza[i])], 'b--')
            ax[2].plot([(laX[i + 1] + laX[i]) / 2 - h, (laX[i + 1] + laX[i]) / 2 + h],
                            [media[i] + 1.96 * np.sqrt(varianza[i]), media[i] + 1.96 * np.sqrt(varianza[i])], 'b--')
            ax[2].plot([(laX[i + 1] + laX[i]) / 2 - h, (laX[i + 1] + laX[i]) / 2 + h],
                            [media[i] - 1.96 * np.sqrt(varianza[i]), media[i] - 1.96 * np.sqrt(varianza[i])], 'b--')
        ax[2].set_xlabel('X')
        ax[0].set_ylabel('Fitted vs observed')
        ax[1].set_ylabel('Residui')
        ax[2].set_ylabel('Residui')
        ax[2].grid(True)
        ax[2].set_xlabel('X')
        ax[0].set_ylabel('Fitted vs observed')
        ax[1].set_ylabel('Residui')
        ax[2].set_ylabel('Residui')
        fig.tight_layout()

        return fig

if __name__ == '__main__':

    # Dati di esempio
    import hashlib
    size_df = 2000
    random_from_string = lambda x: int(hashlib.sha256(str(x).encode('utf-8')).hexdigest(), 16) % 11
    epon = lambda x, a, b, c: a * np.exp(b * x) + c
    a, b, c = [0.4, 0.3, 0.2]
    x = np.linspace(1, 20, size_df)
    p = [1.36, 1.21]
    noise = np.random.randn(size_df)
    countries = np.random.choice(['US', 'IT', 'AU', 'GB'], size = size_df)
    colors = np.random.choice(['Blue', 'Red', 'Orange', 'Yellow', 'Purple', 'Green'], size = size_df)
    genders = np.random.choice([True, False], size = size_df)
    y = epon(x, a, b, c) + [10 if g else 5 for g in genders] + [random_from_string(c) for c in colors] + [random_from_string(c) for c in countries] + noise * 0.4 * x


    # Creazione del DataFrame
    data = {
        'x': x,
        'y': y,
        'country': countries,
        'colors': colors,
        'gender': genders
    }

    df = pd.DataFrame(data)

    # Utilizzo della classe RegressorOptimizer
    reg_opt = RegressorOptimizer(df, 'y', ['x', 'country', 'colors', 'gender'], 'x',
                                 models_to_test = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params_grid': {
                    'fit_intercept': [True, False],
                    'copy_X': [True, False],
                    'n_jobs': [None, 1, 2, 4]
                }
            },
            # 'RandomForestRegressor': {
            #     'model': RandomForestRegressor(),
            #     'params_grid': {
            #         'bootstrap': [True, False],
            #         'max_depth': [10, 50, None],
            #         'max_features': ['auto', 'sqrt', 'log2'],
            #         'min_samples_leaf': [1, 2, 4],
            #         'min_samples_split': [2, 5, 10],
            #         'n_estimators': [600, 1000, 2000]
            #     },
            # },

            'SVR': {
                'model': SVR(),
                'params_grid': {
                    'kernel': ['linear', 'poly', 'rbf'],
                    'C': [1, 10, 100],
                    'gamma': ['scale', 'auto'],
                }
            },
            # 'GradientBoostingRegressor': {
            #     'model': GradientBoostingRegressor(),
            #     'params_grid': {
            #         'n_estimators': [100, 200, 300],
            #         'learning_rate': [0.01, 0.1, 0.2],
            #         'loss': ['ls', 'lad', 'huber', 'quantile'],
            #         'subsample': [0.8, 1.0],
            #     }
            # },
            # 'SGDRegressor': {
            #     'model': SGDRegressor(),
            #     'params_grid': {
            #         'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
            #     }
            # },
            # 'KNeighborsRegressor': {
            #     'model': KNeighborsRegressor(),
            #     'params_grid': {
            #         'n_neighbors': [3, 5, 7],
            #         'weights': ['uniform', 'distance'],
            #     }
            # },
            'MLPRegressor': {
                'model': MLPRegressor(activation = 'relu', learning_rate= 'adaptive', max_iter= 400, verbose=True),
                'params_grid': {
                    'solver': [ 'lbfgs',  'adam'],
                    'hidden_layer_sizes': [(100), (100, 200), (100, 200, 100)],
                    'alpha': [0.0001, 0.001, 0.00001],
                }
            },
            'ElasticNet': {
                'model': ElasticNet(),
                'params_grid': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
                    'fit_intercept': [True, False]
                }
            },
            'Polynomial': {
                'model': make_pipeline(PolynomialFeatures(), LinearRegression()),
                'params_grid': {
                    'polynomialfeatures__degree': [2, 3, 4],
                    'linearregression__fit_intercept': [True, False]
                }
            }
        })
    reg_opt.estimate_optimization_time(60*5)
    reg_opt.test_models_regression()










