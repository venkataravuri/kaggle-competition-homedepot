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

import config
import feature_utils
import nlp_utils
import common_utils


########################
## Cooccurrence terms ##
########################
def cooccurrence_terms(lst1, lst2, join_str):
    terms = [""] * len(lst1) * len(lst2)
    cnt = 0
    for item1 in lst1:
        for item2 in lst2:
            terms[cnt] = item1 + join_str + item2
            cnt += 1
    res = " ".join(terms)
    return res


@common_utils.timing
def extract_tfidf_features(df):
    ## cooccurrence terms
    join_str = "X"

    # query unigram
    print("Generating search_term_unigram_title_unigram feature...")
    df["search_term_unigram_title_unigram"] = list(
        df.apply(lambda x: cooccurrence_terms(x["search_term_unigram"], x["title_unigram"], join_str), axis=1))

    print("Generating search_term_unigram_description_unigram feature...")
    df["search_term_unigram_description_unigram"] = list(
        df.apply(lambda x: cooccurrence_terms(x["search_term_unigram"], x["description_unigram"], join_str), axis=1))

    # query bigram
    print("Generating search_term_bigram_title_unigram feature...")
    df["search_term_bigram_title_unigram"] = list(
        df.apply(lambda x: cooccurrence_terms(x["search_term_bigram"], x["title_unigram"], join_str), axis=1))

    print("Generating search_term_bigram_description_unigram feature...")
    df["search_term_bigram_description_unigram"] = list(
        df.apply(lambda x: cooccurrence_terms(x["search_term_bigram"], x["description_unigram"], join_str), axis=1))


if __name__ == "__main__":
    ############
    ## Config ##
    ############
    ## cooccurrence terms column names
    column_names = [
        "search_term_unigram_title_unigram",
        "search_term_unigram_description_unigram",
        "search_term_bigram_title_unigram",
        "search_term_bigram_description_unigram"
    ]
    ## feature names
    feat_names = [name + "_tfidf" for name in column_names]

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

    if not os.path.exists(config.path_tfidf_features):
        os.makedirs(config.path_tfidf_features)

    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate distance features...")

    extract_tfidf_features(df_train)
    extract_tfidf_features(df_test)

    ngram_range = (1, 1)
    svd_n_components = 10
    n_iter = 5

    #################
    ## Re-training ##
    #################
    print("For training and testing...")
    nd_train = []
    nd_test = []
    for i, (feat_name, column_name) in enumerate(zip(feat_names, column_names)):
        print("Generate %s feat" % feat_name)
        tfv = nlp_utils.getTFV(ngram_range=ngram_range)

        X_tfidf_train = tfv.fit_transform(df_train[column_name])
        print(type(X_tfidf_train))
        print("X_tfidf_train shape: {0}".format(X_tfidf_train.shape))

        X_tfidf_test = tfv.transform(df_test[column_name])
        print(type(X_tfidf_test))
        print("X_tfidf_test shape: {0}".format(X_tfidf_test.shape))

        ## svd
        svd = TruncatedSVD(n_components=svd_n_components, n_iter=n_iter)

        X_svd_train = svd.fit_transform(X_tfidf_train)
        print(type(X_svd_train))
        print("X_svd_train shape: {0}".format(X_svd_train.shape))
        if i == 0:
            nd_train = X_svd_train
        else:
            nd_train = np.hstack([nd_train, X_svd_train])
        print("nd_train shape: {0}".format(nd_train.shape))

        X_svd_test = svd.transform(X_tfidf_test)
        print("X_svd_test shape: {0}".format(X_svd_test.shape))
        if i == 0:
            nd_test = X_svd_test
        else:
            nd_test = np.hstack([nd_test, X_svd_test])
        print("df_train shape: {0}".format(nd_test.shape))

    print("Done.")

    with open("%s/train.tfidf.feat.pkl" % (config.path_tfidf_features), "wb") as f:
        pickle.dump(nd_train, f, -1)
    with open("%s/test.tfidf.feat.pkl" % (config.path_tfidf_features), "wb") as f:
        pickle.dump(nd_test, f, -1)

    # save feat names
    print("Feature names are stored in %s" % config.file_tfidf_feat_name)
    svd_feat_names = ["%s_tfidf_individual_svd%d" % (name, svd_n_components) for name in column_names]
    feature_utils.dump_feat_name(svd_feat_names, config.file_tfidf_feat_name)

    print("All Done.")
