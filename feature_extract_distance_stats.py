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
import os
import pickle
import numpy as np
import pandas as pd

import config
import feature_utils
import common_utils


#####################
## Distance metric ##
#####################
def JaccardCoef(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = try_divide(intersect, union)
    return coef


def DiceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2 * intersect, union)
    return d


def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = JaccardCoef(A, B)
    elif dist == "dice_dist":
        d = DiceDist(A, B)
    return d


def try_divide(x, y, val=0.0):
    """
    	Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val


#####################################
## Extract basic distance features ##
#####################################
@common_utils.timing
def extract_basic_distance_feat(df):
    ## jaccard coef/dice dist of n-gram
    print("generate jaccard coef and dice dist for n-gram")
    dists = ["jaccard_coef", "dice_dist"]
    grams = ["unigram", "bigram", "trigram"]
    feat_names = ["search_term", "title", "description"]
    for dist in dists:
        for gram in grams:
            for i in range(len(feat_names) - 1):
                for j in range(i + 1, len(feat_names)):
                    target_name = feat_names[i]
                    obs_name = feat_names[j]
                    df["%s_of_%s_between_%s_%s" % (dist, gram, target_name, obs_name)] = \
                        list(df.apply(
                            lambda x: compute_dist(x[target_name + "_" + gram], x[obs_name + "_" + gram], dist),
                            axis=1))


if __name__ == "__main__":
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

    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate distance features...")

    extract_basic_distance_feat(df_train)
    extract_basic_distance_feat(df_test)

    feat_names = list()
    feat_names.append("id")
    for name in df_train.columns:
        if "jaccard_coef" in name or "dice_dist" in name:
            feat_names.append(name)

    X_train = df_train[feat_names]
    print(X_train.shape)
    X_test = df_test[feat_names]
    print(X_test.shape)
    if not os.path.exists(config.path_distance_features):
        os.makedirs(config.path_distance_features)

    with open(config.file_distance_feat_train, "wb") as f:
        pickle.dump(X_train, f, -1)
    with open(config.file_distance_feat_test, "wb") as f:
        pickle.dump(X_test, f, -1)

    # save feat names
    print("Feature names are stored in %s" % config.file_distance_feat_name)
    # dump feat name
    feature_utils.dump_feat_name(feat_names, config.file_distance_feat_name)

    print("All Done.")
