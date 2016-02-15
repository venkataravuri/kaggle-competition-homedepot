"""
__file__
    combine_feat.py
__description__
    This file generates one combination of feature set (Low).
__author__
    Venkata Ravuri < venkata.ravuri@gmail.com >
"""
from sklearn.base import BaseEstimator
import pickle
import numpy as np
from sklearn.datasets import dump_svmlight_file
import pandas as pd

import config


def identity(x):
    return x


class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return self.transformer(X)


#### function to combine features
def combine_feat(feat_names):
    print("==================================================")
    print("Combine features...")

    ##########################
    ## Training and Testing ##
    ##########################
    print("For training and testing...")

    with open(config.file_words_count_feat_train, "rb") as f:
        df_train = pickle.load(f)
    with open(config.file_words_count_feat_test, "rb") as f:
        df_test = pickle.load(f)

    for i, (feat_name, transformer) in enumerate(feat_names):

        ## apply transformation
        x_train = transformer.fit_transform(df_train[feat_name])
        x_test = transformer.transform(df_test[feat_name])

        ## stack feat
        ## stack feat
        if i == 0:
            X_train, X_test = x_train, x_test
        else:
            X_train, X_test = np.hstack([X_train, x_train]), np.hstack([X_test, x_test])

        print("Combine {:>2}/{:>2} feat: {} ({}D)".format(i + 1, len(feat_names), feat_name, x_train.shape))

    print("Feat dim: {}D".format(X_train.shape))

    df_train_original = pd.read_csv(config.path_raw + config.file_train, engine='python')
    Y_train = df_train_original['relevance']

    Y_test = pd.DataFrame(np.zeros(X_test.shape[0]), columns=['relevance'])

    print(X_train.shape[0])
    print(Y_train.shape[0])
    ## dump feat
    dump_svmlight_file(X_train, Y_train, "all.train.feat")
    dump_svmlight_file(X_test, Y_test, "all.test.feat")


if __name__ == "__main__":
    feat_names = [

        ################
        ## Word count ##
        ################
        ('count_of_search_term_unigram', SimpleTransform()),
        ('count_of_unique_search_term_unigram', SimpleTransform()),
        ('ratio_of_unique_search_term_unigram', SimpleTransform()),
        ('count_of_search_term_bigram', SimpleTransform()),
        ('count_of_unique_search_term_bigram', SimpleTransform()),
        ('ratio_of_unique_search_term_bigram', SimpleTransform()),
        ('count_of_search_term_trigram', SimpleTransform()),
        ('count_of_unique_search_term_trigram', SimpleTransform()),
        ('ratio_of_unique_search_term_trigram', SimpleTransform()),
        ('count_of_digit_in_search_term', SimpleTransform()),
        ('ratio_of_digit_in_search_term', SimpleTransform()),
        ('count_of_title_unigram', SimpleTransform()),
        ('count_of_unique_title_unigram', SimpleTransform()),
        ('ratio_of_unique_title_unigram', SimpleTransform()),
        ('count_of_title_bigram', SimpleTransform()),
        ('count_of_unique_title_bigram', SimpleTransform()),
        ('ratio_of_unique_title_bigram', SimpleTransform()),
        ('count_of_title_trigram', SimpleTransform()),
        ('count_of_unique_title_trigram', SimpleTransform()),
        ('ratio_of_unique_title_trigram', SimpleTransform()),
        ('count_of_digit_in_title', SimpleTransform()),
        ('ratio_of_digit_in_title', SimpleTransform()),
        ('count_of_description_unigram', SimpleTransform()),
        ('count_of_unique_description_unigram', SimpleTransform()),
        ('ratio_of_unique_description_unigram', SimpleTransform()),
        ('count_of_description_bigram', SimpleTransform()),
        ('count_of_unique_description_bigram', SimpleTransform()),
        ('ratio_of_unique_description_bigram', SimpleTransform()),
        ('count_of_description_trigram', SimpleTransform()),
        ('count_of_unique_description_trigram', SimpleTransform()),
        ('ratio_of_unique_description_trigram', SimpleTransform()),
        ('count_of_digit_in_description', SimpleTransform()),
        ('ratio_of_digit_in_description', SimpleTransform()),
        ('description_missing', SimpleTransform())

        ##############
        ## Position ##
        ##############

        ############
        ## TF-IDF ##
        ############

    ]

    combine_feat(feat_names)
