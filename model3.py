# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:05:44 2016

@author: VRavuri
"""
import time
import pandas as pd
import numpy as np
from sklearn import pipeline, grid_search
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import common_utils
import pickle
from operator import itemgetter
import xgboost as xgb

import config


@common_utils.timing
def load_data():
    print("Load data...")

    with open(config.file_words_count_feat_train, "rb") as f:
        df_words_count_feat_train = pickle.load(f)
    with open(config.file_words_count_feat_test, "rb") as f:
        df_words_count_feat_test = pickle.load(f)

    with open(config.file_brand_count_feat_train, "rb") as f:
        df_brand_feat_train = pickle.load(f)
    with open(config.file_brand_count_feat_test, "rb") as f:
        df_brand_feat_test = pickle.load(f)

    with open(config.file_distance_feat_train, "rb") as f:
        df_distance_feat_train = pickle.load(f)
    with open(config.file_distance_feat_test, "rb") as f:
        df_distance_feat_test = pickle.load(f)

    with open(config.file_cosine_feat_train, "rb") as f:
        df_cosine_feat_train = pickle.load(f)
    with open(config.file_cosine_feat_test, "rb") as f:
        df_cosine_feat_test = pickle.load(f)

    with open("%s/train.tfidf.feat.pkl" % (config.path_tfidf_features), "rb") as f:
        nd_tfidf_train = pickle.load(f)
    print("nd_tfidf_train shape:{0}".format(nd_tfidf_train.shape))
    with open("%s/test.tfidf.feat.pkl" % (config.path_tfidf_features), "rb") as f:
        nd_tfidf_test = pickle.load(f)
    print("nd_tfidf_test shape:{0}".format(nd_tfidf_test.shape))

    df_train = pd.merge(pd.merge(df_words_count_feat_train, df_distance_feat_train, how='left', on='id'), df_cosine_feat_train, how='left',
                        on='id')
    df_train = pd.merge(df_train, df_brand_feat_train, how='left', on='id')
    df_test = pd.merge(pd.merge(df_words_count_feat_test, df_distance_feat_test, how='left', on='id'), df_cosine_feat_test, how='left',
                       on='id')
    df_test = pd.merge(df_test, df_brand_feat_test, how='left', on='id')

    # df_train = pd.merge(df_words_count_feat_train, df_distance_feat_train, how='left', on='id')
    # df_train = pd.merge(df_words_count_feat_train, df_distance_feat_train, how='left', on='id')

    df_train_original = pd.read_csv(config.path_raw + config.file_train, engine='python')
    y_train = df_train_original['relevance']

    print("Done.")
    return df_train, df_test, y_train, nd_tfidf_train, nd_tfidf_test


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_


def load_feature_names(file_feat_name, id_removal=False):
    ################
    ## Word count ##
    ################
    feat_names = [line.strip() for line in open(file_feat_name, 'r')]
    if id_removal:
        feat_names.remove('id')

    return feat_names


def gbm_model():
    # rfr = RandomForestRegressor()

    gbm = GradientBoostingRegressor()

    clf = pipeline.Pipeline([('gbm', gbm)])
    param_grid = {'gbm__n_estimators': [100],  # 300 top
                  'gbm__max_depth': [3],  # list(range(7,8,1))
                  }

    model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=5, verbose=150, scoring=RMSE)

    return model, param_grid


def xgboost_model():
    # Create the pipeline
    clf = pipeline.Pipeline([('model', xgb.XGBRegressor())])

    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {
        'model__objective': ['reg:linear'], # Default value
        'model__min_child_weight': [5],  # 1 is default
        'model__subsample': [0.9],  # 1 is default
        'model__max_depth': [9],  # 6 is default
        'model__learning_rate': [0.05],
        'model__n_estimators': [300] # Was 200
    }

    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator=clf,
                                     param_grid=param_grid,
                                     scoring=RMSE,
                                     verbose=10,
                                     n_jobs=4,
                                     iid=True,
                                     refit=True,
                                     cv=5)

    return model, param_grid

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

if __name__ == '__main__':
    feat_names = list()
    feat_names.extend(load_feature_names(config.file_feat_name, id_removal=True))
    feat_names.extend(load_feature_names(config.file_distance_feat_name, id_removal=True))
    feat_names.extend(load_feature_names(config.file_cosine_feat_name, id_removal=True))
    feat_names.extend(load_feature_names(config.file_brand_feat_name, id_removal=True))

    print("Total features: {}".format(len(feat_names)))

    df_train, df_test, y_train, nd_tfidf_train, nd_tfidf_test = load_data()

    RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

    print(df_train.columns)
    X_train = df_train[feat_names]

    # Storing id in temporary variable, later used during generating submission file.
    id_test = df_test['id']
    X_test = df_test[feat_names]

    X_train_ndarray = X_train.as_matrix()
    print("X_train_ndarray shape: {0}".format(X_train_ndarray.shape))
    X_train_ndarray = np.hstack([X_train_ndarray, nd_tfidf_train])
    print("X_train_ndarray shape: {0}".format(X_train_ndarray.shape))

    X_test_ndarray = X_test.as_matrix()
    print("X_test_ndarray shape: {0}".format(X_test_ndarray.shape))
    X_test_ndarray = np.hstack([X_test_ndarray, nd_tfidf_test])
    print("X_test_ndarray shape: {0}".format(X_test_ndarray.shape))

    model, param_grid = xgboost_model()

    # Fit Grid Search Model
    model.fit(X_train_ndarray, y_train)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    report(model.grid_scores_)

    # Get best model
    best_model = model.best_estimator_

    # Fit model with best parameters optimized for normalized_gini
    best_model.fit(X_train_ndarray, y_train)

    y_pred = best_model.predict(X_test_ndarray)

    print(len(y_pred))
    print("Generating submission file.")
    Y_pred = pd.DataFrame({"id": id_test, "relevance": y_pred})
    Y_pred.relevance[Y_pred.relevance > 3] = 3
    Y_pred.to_csv(config.path_submit + config.file_submission_name + time.strftime("-%Y%m%d-%H%M%S") + ".csv", index=False)
