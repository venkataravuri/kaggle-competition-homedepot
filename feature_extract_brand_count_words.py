import os
import numpy as np
import pickle
import pandas as pd

import config
import feature_utils
import common_utils


@common_utils.timing
def generate_intersect_word_count(df):
    ##############################
    ## intersect word count     ##
    ##############################
    print("generate intersect word counting features")
    grams = ["unigram", "bigram", "trigram"]

    for gram in grams:
        # word count
        print("Generating count_of_brand_{0} feature...".format(gram))
        df["count_of_brand_%s" % (gram)] = df.apply(lambda x: len(x["brand_" + gram]), axis=1)
        # search term
        print("Generating count_of_search_term_{0}_in_brand_{1} feature...".format(gram, gram))
        df["count_of_search_term_%s_in_brand_%s" % (gram, gram)] = list(
            df.apply(lambda x: sum([1. for w in x["search_term_" + gram] if w in set(x["title_" + gram])]), axis=1))
        print("Generating ratio_of_search_term_{0}_in_title_{1} feature...".format(gram, gram))
        df["ratio_of_search_term_%s_in_brand_%s" % (gram, gram)] = df.apply(
            lambda x: feature_utils.try_divide(x["count_of_search_term_%s_in_brand_%s" % (gram, gram)],
                                               x["count_of_brand_%s" % (gram)]), axis=1)


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
    print("Generate counting features...")

    df_train = df_train.fillna('')
    df_test = df_test.fillna('')
    generate_intersect_word_count(df_train)
    generate_intersect_word_count(df_test)

    feat_names = list()
    feat_names.append("id")
    for name in df_train.columns:
        if "count_of_" in name or "ratio_of_" in name:
            feat_names.append(name)

    X_train = df_train[feat_names]
    print(X_train.shape)
    X_test = df_test[feat_names]
    print(X_test.shape)
    if not os.path.exists(config.path_brand_counting_features):
        os.makedirs(config.path_brand_counting_features)

    with open(config.file_brand_count_feat_train, "wb") as f:
        pickle.dump(X_train, f, -1)
    with open(config.file_brand_count_feat_test, "wb") as f:
        pickle.dump(X_test, f, -1)

    # save feat names
    print("Feature names are stored in %s" % config.file_brand_feat_name)
    # dump feat name
    feature_utils.dump_feat_name(feat_names, config.file_brand_feat_name)

    print("All Done.")
