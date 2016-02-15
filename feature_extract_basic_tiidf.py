"""
__file__
    feature_extraction_distance_feat.py
__description__

    This idea is borrowed from Chenglong Chen < c.chenglong@gmail.com >

    This file generates the following features for each run and fold, and for the entire training and testing set.
        1. jaccard coefficient/dice distance between query & title, query & description, title & description pairs
            - just plain jaccard coefficient/dice distance
            - compute for unigram/bigram/trigram
        2. jaccard coefficient/dice distance stats features for title/description
__author__
    Venkata Ravuri < venkata.ravuri@gmail.com >
"""
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

import config
import feature_utils
import nlp_utils
import common_utils

stats_feat_flag = True


#####################
## Helper function ##
#####################
## compute cosine similarity
def cosine_sim(x, y):
    try:
        d = cosine_similarity(x, y)
        d = d[0][0]
    except:
        print(x)
        print(y)
        d = 0.
    return d


## generate distance stats feat
def generate_dist_stats_feat(metric, X_train, ids_train, X_test, ids_test, indices_dict, qids_test=None):
    if metric == "cosine":
        stats_feat = 0 * np.ones((len(ids_test), stats_feat_num * n_classes), dtype=float)
        sim = 1. - pairwise_distances(X_test, X_train, metric=metric, n_jobs=1)
    elif metric == "euclidean":
        stats_feat = -1 * np.ones((len(ids_test), stats_feat_num * n_classes), dtype=float)
        sim = pairwise_distances(X_test, X_train, metric=metric, n_jobs=1)

    for i in range(len(ids_test)):
        id = ids_test[i]
        if qids_test is not None:
            qid = qids_test[i]
        for j in range(n_classes):
            key = (qid, j + 1) if qids_test is not None else j + 1
            if indices_dict.has_key(key):
                inds = indices_dict[key]
                # exclude this sample itself from the list of indices
                inds = [ind for ind in inds if id != ids_train[ind]]
                sim_tmp = sim[i][inds]
                if len(sim_tmp) != 0:
                    feat = [func(sim_tmp) for func in stats_func]
                    ## quantile
                    sim_tmp = pd.Series(sim_tmp)
                    quantiles = sim_tmp.quantile(quantiles_range)
                    feat = np.hstack((feat, quantiles))
                    stats_feat[i, j * stats_feat_num:(j + 1) * stats_feat_num] = feat
    return stats_feat


## extract all features
def extract_feat():
    nd_train = []
    nd_test = []
    ## first fit a bow/tfidf on the all_text to get
    ## the common vocabulary to ensure query/title/description
    ## has the same length bow/tfidf for computing the similarity
    if vocabulary_type == "common":
        if vec_type == "tfidf":
            vec = feature_utils.getTFV(ngram_range=ngram_range)
        elif vec_type == "bow":
            vec = feature_utils.getBOW(ngram_range=ngram_range)
        vec.fit(df_train["all_text"])
        vocabulary = vec.vocabulary_
    elif vocabulary_type == "individual":
        vocabulary = None
    for i, (feat_name, column_name) in enumerate(zip(feat_names, column_names)):

        ##########################
        ## basic bow/tfidf feat ##
        ##########################
        print
        "generate %s feat for %s" % (vec_type, column_name)
        if vec_type == "tfidf":
            vec = feature_utils.getTFV(ngram_range=ngram_range, vocabulary=vocabulary)
        elif vec_type == "bow":
            vec = feature_utils.getBOW(ngram_range=ngram_range, vocabulary=vocabulary)
        X_train = vec.fit_transform(df_train[column_name])
        X_test = vec.transform(df_test[column_name])

        ## svd
        svd = TruncatedSVD(n_components=svd_n_components)

        X_svd_train = svd.fit_transform(X_train)
        print(type(X_svd_train))
        print("X_svd_train shape: {0}".format(X_svd_train.shape))
        if i == 0:
            nd_train = X_svd_train
        else:
            nd_train = np.hstack([nd_train, X_svd_train])
        print("nd_train shape: {0}".format(nd_train.shape))

        X_svd_test = svd.transform(X_test)
        print("X_svd_test shape: {0}".format(X_svd_test.shape))
        if i == 0:
            nd_test = X_svd_test
        else:
            nd_test = np.hstack([nd_test, X_svd_test])
        print("df_train shape: {0}".format(nd_test.shape))

    print("Done.")


if __name__ == "__main__":

    ############
    ## Config ##
    ############
    # stats to extract
    quantiles_range = np.arange(0, 1.5, 0.5)
    stats_func = [np.mean, np.std]
    stats_feat_num = len(quantiles_range) + len(stats_func)

    # tfidf config
    vec_types = ["tfidf", "bow"]
    ngram_range = (1, 3)
    vocabulary_type = "common"
    svd_n_components = [100]
    tsne_n_components = [2]
    n_classes = 4

    # feat name config
    column_names = ["search_term", "product_title", "product_description"]

    ###############
    ## Load Data ##
    ###############
    # load data
    print("Load data...")
    with open(config.file_preprocess_ngrams_train, "rb") as f:
        df_train = pickle.load(f)
    with open(config.file_preprocess_ngrams_test, "rb") as f:
        df_test = pickle.load(f)
    print("Done.")


    ## for fitting common vocabulary
    def cat_text(x):
        res = '%s %s %s' % (x['search_term'], x['product_title'], x['product_description'])
        return res


    df_train["all_text"] = list(df_train.apply(cat_text, axis=1))
    df_test["all_text"] = list(df_test.apply(cat_text, axis=1))

    for vec_type in vec_types:
        ## save feat names
        feat_names = ["search_term", "title", "description"]
        feat_names = [name + "_%s_%s_vocabulary" % (vec_type, vocabulary_type) for name in feat_names]

        #######################
        ## Generate Features ##
        #######################
        print("==================================================")
        print("Generate basic %s features..." % vec_type)

        print("For training and testing...")
        ## exfeature_utilstract feat
        nd_tfidf_train, nd_tfidf_test = extract_feat()
        nd_bow_train, nd_bow_test = extract_feat()

        with open("%s/train.basic.tfidf.%s.feat.pkl" % (config.path_basic_tfidf_features, vec_type), "wb") as f:
            pickle.dump(np.hstack[nd_tfidf_train, nd_bow_train], f, -1)
        with open("%s/test.basic.tfidf.%s.feat.pkl" % (config.path_basic_tfidf_features, vec_type), "wb") as f:
            pickle.dump(np.hstack[nd_tfidf_test, nd_bow_test], f, -1)

    print("All Done.")
