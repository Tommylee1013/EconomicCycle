import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from typing import Callable
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline

from scipy.stats import rv_continuous, kstest

def linear_trend_t_values(close : np.array) :
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sm.OLS(close, x).fit()
    return ols.tvalues[1]

def trend_labeling(molecule, close : pd.Series, span : list) :
    out = pd.DataFrame(index = molecule, columns = ['t1','tVal','bin'])
    horizons = range(*span)
    for dt0 in molecule :
        df0 = pd.Series()
        iloc0 = close.index.get_loc(dt0)
        if iloc0 + max(horizons) > close.shape[0] : continue
        for horizon in horizons :
            dt1 = close.index[iloc0 + horizon -1]
            df1 = close.loc[dt0 : dt1]
            df0.loc[dt1] = linear_trend_t_values(df1.values)
        dt1 = df0.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
        out.loc[dt0, ['t1','tVal','bin']] = df0.index[-1], df0[dt1], np.sign(df0[dt1])
    out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast = 'signed')
    return out.dropna(subset = ['bin'])

def getYX(series, constant, lags):
    series_ = series.diff().dropna()
    x = lagDF(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] -1: -1]
    y = series_.iloc[-x.shape[0]:].values
    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis = 1)
        if constant[:2] == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis = 1)
        if constant == 'ctt':
            x = np.append(x, trend ** 2, axis = 1)
    return y, x

def lagDF(series, lags):
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = series.shift(lag).copy(deep = True)
        df_.name = str(series.name) + '_' + str(lag)
        df1 = df1.join(df_, how = 'outer')
    return df1

def get_bSADF(logP, minSL, constant, lags):
    y, x= getYX(logP, constant = constant, lags=lags)
    startPoints, bsadf, allADF = range(0, y.shape[0] + lags - minSL + 1), 0, []
    for start in startPoints:
        y_, x_ = y[start:], x[start:]
        bMean_, bStd_ = getBetas(y_, x_)
        bMean_, bStd_ = bMean_[0], bStd_[0, 0] ** .5
        allADF.append(bMean_ / bStd_)
        if allADF[-1] > bsadf : bsadf = allADF[-1]
    return bsadf

def getBetas(y, x):
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    if np.linalg.matrix_rank(xx) < x.shape[1]:
        pass
    else:
        xxinv = np.linalg.inv(xx)
        bMean = np.dot(xxinv, xy)
        err = y - np.dot(x, bMean)
        bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
        return bMean, bVar

def get_bSADF_test_statistics(logP, minSL, constant, lags):
    test_statistics = []
    for i in range(len(logP)):
        logP_ = logP.iloc[:i+1]
        bsadf = get_bSADF(logP_, minSL, constant, lags)
        test_statistics.append(bsadf)
    test_statistics = pd.Series(test_statistics)
    test_statistics.index = logP.index
    test_statistics.name = 'GSADF'
    return test_statistics

def mean_decrease_impurity(model, feature_names):
    feature_imp_df = {i: tree.feature_importances_ for i, tree in enumerate(model.estimators_)}
    feature_imp_df = pd.DataFrame.from_dict(feature_imp_df, orient='index')
    feature_imp_df.columns = feature_names
    feature_imp_df = feature_imp_df.replace(0, np.nan)

    importance = pd.concat({'mean': feature_imp_df.mean(),
                            'std': feature_imp_df.std() * feature_imp_df.shape[0] ** -0.5},
                           axis=1)
    importance /= importance['mean'].sum()
    return importance

def mean_decrease_accuracy(model, X, y, cv_gen, sample_weight=None, scoring=log_loss):
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    fold_metrics_values, features_metrics_values = pd.Series(), pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        fit = model.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight[train])
        pred = fit.predict(X.iloc[test, :])
        if scoring == log_loss:
            prob = fit.predict_proba(X.iloc[test, :])
            fold_metrics_values.loc[i] = -scoring(y.iloc[test], prob, sample_weight=sample_weight[test],
                                                  labels=model.classes_)
        else:
            fold_metrics_values.loc[i] = scoring(y.iloc[test], pred, sample_weight=sample_weight[test])
        for j in X.columns:
            X1_ = X.iloc[test, :].copy(deep=True)
            np.random.shuffle(X1_[j].values)
            if scoring == log_loss:
                prob = fit.predict_proba(X1_)
                features_metrics_values.loc[i, j] = -scoring(y.iloc[test], prob, sample_weight=sample_weight[test],
                                                             labels=model.classes_)
            else:
                pred = fit.predict(X1_)
                features_metrics_values.loc[i, j] = scoring(y.iloc[test], pred,
                                                            sample_weight=sample_weight[test])

    importance = (-features_metrics_values).add(fold_metrics_values, axis=0)
    if scoring == log_loss:
        importance = importance / -features_metrics_values
    else:
        importance = importance / (1.0 - features_metrics_values)
    importance = pd.concat({'mean': importance.mean(), 'std': importance.std() * importance.shape[0] ** -.5}, axis=1)
    importance.replace([-np.inf, np.nan], 0, inplace=True)

    return importance

def single_feature_importance(clf, X, y, cv_gen, sample_weight = None, scoring = log_loss):
    feature_names = X.columns
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    imp = pd.DataFrame(columns=['mean', 'std'])
    for feat in feature_names:
        feat_cross_val_scores = ml_cross_val_score(clf, X=X[[feat]], y=y, sample_weight=sample_weight,
                                                   scoring=scoring, cv_gen=cv_gen)
        imp.loc[feat, 'mean'] = feat_cross_val_scores.mean()
        imp.loc[feat, 'std'] = feat_cross_val_scores.std() * feat_cross_val_scores.shape[0] ** -.5
    return imp

def plot_feature_importance(importance_df, oob_score, oos_score, save_fig=False, output_path=None):
    plt.figure(figsize=(10, importance_df.shape[0] / 5))
    importance_df.sort_values('mean', ascending=True, inplace=True)
    importance_df['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=importance_df['std'], error_kw={'ecolor': 'r'})
    plt.title('Feature importance. OOB Score:{}; OOS score:{}'.format(round(oob_score, 4), round(oos_score, 4)))

    if save_fig is True:
        plt.savefig(output_path)
    else:
        plt.show()

def ml_get_train_times(samples_info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    train = samples_info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.iteritems():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index
        df1 = train[(start_ix <= train) & (train <= end_ix)].index
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index
        train = train.drop(df0.union(df1).union(df2))
    return train


class PurgedKFold(KFold):
    def __init__(self,
                 n_splits: int = 3,
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.):

        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError('The samples_info_sets param must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)

        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups = None):
        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        indices: np.ndarray = np.arange(X.shape[0])
        embargo: int = int(X.shape[0] * self.pct_embargo)

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]

            if end_ix < X.shape[0]:
                end_ix += embargo

            test_times = pd.Series(index=[self.samples_info_sets[start_ix]], data=[self.samples_info_sets[end_ix-1]])
            train_times = ml_get_train_times(self.samples_info_sets, test_times)

            train_indices = []
            for train_ix in train_times.index:
                train_indices.append(self.samples_info_sets.index.get_loc(train_ix))
            yield np.array(train_indices), test_indices

class SampledPipeline(Pipeline) :
    def fit(self, X, y, sample_weight = None, **fit_params):
        if sample_weight is not None :
            fit_params[self.steps[-1][0] + ' sample_weight'] = sample_weight
        return super(SampledPipeline, self).fit(X, y, **fit_params)

def ml_cross_val_score(
        classifier: ClassifierMixin,
        X: pd.DataFrame,
        y: pd.Series,
        cv_gen: BaseCrossValidator,
        sample_weight: np.ndarray = None,
        scoring: Callable[[np.array, np.array], float] = log_loss):
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))
    ret_scores = []
    for train, test in cv_gen.split(X=X, y=y):
        fit = classifier.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight[train])
        if scoring == log_loss:
            prob = fit.predict_proba(X.iloc[test, :])
            score = -1 * scoring(y.iloc[test], prob, sample_weight=sample_weight[test], labels=classifier.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score = scoring(y.iloc[test], pred, sample_weight=sample_weight[test])
        ret_scores.append(score)
    return np.array(ret_scores)

def grid_search_cross_validation(feat,
                                 label : pd.Series,
                                 samples_info_sets : pd.Series,
                                 pipe_clf, param_grid,
                                 cv : int = 3,
                                 bagging : list = [0, None, 1],
                                 random_search_iterator : int = 0,
                                 n_jobs : int = -1,
                                 pct_embargo : float = 0.0,
                                 **fit_params) :
    if set(label.values) == {0,1} : scoring = 'f1'
    else : scoring = 'neg_log_loss'

    inner_cv = PurgedKFold(n_splits = cv, samples_info_sets = samples_info_sets, pct_embargo = pct_embargo)
    if random_search_iterator == 0:
        grid_search = GridSearchCV(estimator = pipe_clf, param_grid = param_grid,
                                    scoring = scoring, cv = inner_cv, n_jobs = n_jobs, iid = False)
    else :
        grid_search = RandomizedSearchCV(estimator = pipe_clf, param_distributions = param_grid,
                                         scoring = scoring, cv = inner_cv, n_jobs = n_jobs,
                                         iid = False, n_iter = random_search_iterator)
    grid_search = grid_search.fit(feat, label, **fit_params).best_extimator_

    if bagging[1] > 0 :
        grid_search = BaggingClassifier(base_estimator = SampledPipeline(grid_search.steps),
                                        n_estimators = int(bagging[0]), max_samples = float(bagging[1]),
                                        max_features = float(bagging[2]), n_jobs = n_jobs)
        grid_search = grid_search.fit(feat, label,
                                      sample_weight = fit_params[grid_search.base_estimator.steps[-1][0]+' sample_weight'])
        grid_search = Pipeline([('bag', grid_search)])
    return grid_search

class logUniform_gen(rv_continuous) :
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)
def log_uniform(a = 1, b = np.exp(1)) :
    return logUniform_gen(a = a, b = b, name = 'logUniform')