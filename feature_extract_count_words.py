"""
__file__
    feature_extract_count_words.py
__description__
    Adopted from @Chenglong Chen's code < https://github.com/ChenglongChen/Kaggle_CrowdFlower >
    This file generates the following features for the entire training and testing set.
        1. Basic Counting Features
            1. Count of n-gram in query/title/description
            2. Count & Ratio of Digit in query/title/description
            3. Count & Ratio of Unique n-gram in query/title/description
        2. Intersect Counting Features
            1. Count & Ratio of a's n-gram in b's n-gram
        3. Intersect Position Features
            1. Statistics of Positions of a's n-gram in b's n-gram
            2. Statistics of Normalized Positions of a's n-gram in b's n-gram
__author__
    Venkata Ravuri < venkat@nikhu.com >
"""
import os
import numpy as np
import pickle
import pandas as pd

import config
import feature_utils
import common_utils


def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j, w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


@common_utils.timing
def generate_word_counting_features(df):
    ################################
    ## word count and digit count ##
    ################################
    print("generate word counting features")
    feat_names = ["search_term", "title", "description"]
    grams = ["unigram", "bigram", "trigram"]
    count_digit = lambda x: sum([1. for w in x if w.isdigit()])
    for feat_name in feat_names:
        for gram in grams:
            # word count
            print("Generating count_of_{0}_{1} feature...".format(feat_name, gram))
            df["count_of_%s_%s" % (feat_name, gram)] = df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1)
            print("Generating count_of_unique_{0}_{1} feature...".format(feat_name, gram))
            df["count_of_unique_%s_%s" % (feat_name, gram)] = df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1)
            print("Generating ratio_of_unique_{0}_{1} feature...".format(feat_name, gram))
            df["ratio_of_unique_%s_%s" % (feat_name, gram)] = df.apply(
                lambda x: feature_utils.try_divide(x["count_of_unique_%s_%s" % (feat_name, gram)],
                                                   x["count_of_%s_%s" % (feat_name, gram)]), axis=1)

        # digit count
        print("Generating count_of_digit_in_{0} feature...".format(feat_name))
        df["count_of_digit_in_%s" % feat_name] = df.apply(lambda x: count_digit(x[feat_name + "_unigram"]), axis=1)
        print("Generating ratio_of_digit_in_{0} feature...".format(feat_name))
        df["ratio_of_digit_in_%s" % feat_name] = df.apply(lambda x: feature_utils.try_divide(x["count_of_digit_in_%s" % feat_name],
                                                                                             x["count_of_%s_unigram" % (feat_name)]),
                                                          axis=1)
    # description missing indicator
    print("Generating description_missing feature...")
    df["description_missing"] = df.apply(lambda x: int(x["description_unigram"] == ""), axis=1)


@common_utils.timing
def generate_intersect_word_count(df):
    ##############################
    ## intersect word count     ##
    ##############################
    print("generate intersect word counting features")
    feat_names = ["search_term", "title", "description"]
    grams = ["unigram", "bigram", "trigram"]

    for gram in grams:
        for obs_name in feat_names:
            for target_name in feat_names:
                if target_name != obs_name:
                    ## query
                    print("Generating count_of_{0}_{1}_in_{2} feature...".format(obs_name, gram, target_name))
                    df["count_of_%s_%s_in_%s" % (obs_name, gram, target_name)] = list(
                        df.apply(lambda x: sum([1. for w in x[obs_name + "_" + gram] if w in set(x[target_name + "_" + gram])]), axis=1))
                    print("Generating ratio_of_{0}_{1}_in_{2} feature...".format(obs_name, gram, target_name))
                    df["ratio_of_%s_%s_in_%s" % (obs_name, gram, target_name)] = df.apply(
                        lambda x: feature_utils.try_divide(x["count_of_%s_%s_in_%s" % (obs_name, gram, target_name)],
                                                           x["count_of_%s_%s" % (obs_name, gram)]), axis=1)
        ## some other feat
        print("Generating title_{0}_in_search_term_div_search_term_{1} feature...".format(gram, gram))
        df["title_%s_in_search_term_div_search_term_%s" % (gram, gram)] = df.apply(
            lambda x: feature_utils.try_divide(x["count_of_title_%s_in_search_term" % gram], x["count_of_search_term_%s" % gram]), axis=1)
        print("Generating title_{0}_in_search_term_div_search_term_{1}_in_title feature...".format(gram, gram))
        df["title_%s_in_search_term_div_search_term_%s_in_title" % (gram, gram)] = df.apply(
            lambda x: feature_utils.try_divide(x["count_of_title_%s_in_search_term" % gram], x["count_of_search_term_%s_in_title" % gram]),
            axis=1)
        print("Generating description_{0}_in_search_term_div_search_term_{1} feature...".format(gram, gram))
        df["description_%s_in_search_term_div_search_term_%s" % (gram, gram)] = df.apply(
            lambda x: feature_utils.try_divide(x["count_of_description_%s_in_search_term" % gram], x["count_of_search_term_%s" % gram]),
            axis=1)
        print("Generating description_{0}_in_search_term_div_search_term_{1}_in_description feature...".format(gram, gram))
        df["description_%s_in_search_term_div_search_term_%s_in_description" % (gram, gram)] = df.apply(
            lambda x: feature_utils.try_divide(x["count_of_description_%s_in_search_term" % gram],
                                               x["count_of_search_term_%s_in_description" % gram]),
            axis=1)


@common_utils.timing
def generate_intersect_word_position_features(df):
    ######################################
    ## intersect word position feat ##
    ######################################
    print("generate intersect word position features")
    feat_names = ["search_term", "title", "description"]
    grams = ["unigram"]
    for gram in grams:
        for target_name in feat_names:
            for obs_name in feat_names:
                if target_name != obs_name:
                    pos = df.apply(lambda x: get_position_list(x[target_name + "_" + gram], obs=x[obs_name + "_" + gram]), axis=1)
                    # stats feat on pos
                    print("Generating pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name))
                    df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = pos.apply(lambda x: np.min(x))  # np.min(pos)
                    # print(df['pos_of_title_unigram_in_search_term_min'])
                    print("Generating pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name))
                    df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = pos.apply(lambda x: np.mean(x))
                    print("Generating pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name))
                    df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = pos.apply(lambda x: np.median(x))
                    print("Generating pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name))
                    df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = pos.apply(lambda x: np.max(x))
                    print("Generating pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name))
                    df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = pos.apply(lambda x: np.std(x))
                    # stats feat on normalized_pos
                    print("Generating normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name))
                    df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = df.apply(
                        lambda x: feature_utils.try_divide(x["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)],
                                                           x["count_of_%s_%s" % (obs_name, gram)]), axis=1)
                    print("Generating normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name))
                    df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = df.apply(
                        lambda x: feature_utils.try_divide(x["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)],
                                                           x["count_of_%s_%s" % (obs_name, gram)]), axis=1)
                    print("Generating normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name))
                    df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = df.apply(
                        lambda x: feature_utils.try_divide(x["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)],
                                                           x["count_of_%s_%s" % (obs_name, gram)]), axis=1)
                    print("Generating normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name))
                    df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = df.apply(
                        lambda x: feature_utils.try_divide(x["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)],
                                                           x["count_of_%s_%s" % (obs_name, gram)]), axis=1)
                    print("Generating normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name))
                    df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = df.apply(
                        lambda x: feature_utils.try_divide(x["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)],
                                                           x["count_of_%s_%s" % (obs_name, gram)]), axis=1)


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

    generate_word_counting_features(df_train)
    generate_word_counting_features(df_test)

    generate_intersect_word_count(df_train)
    generate_intersect_word_count(df_test)

    generate_intersect_word_position_features(df_train)
    generate_intersect_word_position_features(df_test)

    feat_names = list()
    feat_names.append("id")
    for name in df_train.columns:
        if "count" in name or "ratio" in name or "div" in name or "pos_of" in name:
            feat_names.append(name)
    feat_names.append("description_missing")

    X_train = df_train[feat_names]
    print(X_train.shape)
    X_test = df_test[feat_names]
    print(X_test.shape)
    if not os.path.exists(config.path_counting_features):
        os.makedirs(config.path_counting_features)

    with open(config.file_words_count_feat_train, "wb") as f:
        pickle.dump(X_train, f, -1)
    with open(config.file_words_count_feat_test, "wb") as f:
        pickle.dump(X_test, f, -1)

    # save feat names
    print("Feature names are stored in %s" % config.file_feat_name)
    # dump feat name
    feature_utils.dump_feat_name(feat_names, config.file_feat_name)

    print("All Done.")
