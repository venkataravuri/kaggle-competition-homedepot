"""
__file__
    preprocess.py
__description__
    Adopted from @Chenglong Chen's code < https://github.com/ChenglongChen/Kaggle_CrowdFlower >
    This file generates ngrams which is reused across different feature generation modules.
__author__
    Venkata Ravuri < venkat@nikhu.com >
"""
import re
import pickle
import pandas as pd
import numpy as np

import config
import nlp_utils
import ngram
import common_utils


######################
## Pre-process data ##
######################
def preprocess_data(line):
    # tokenize
    tokens = token_pattern.findall(line)
    # stem
    tokens_stemmed = nlp_utils.stem_tokens(tokens, nlp_utils.english_stemmer)
    # Stop words removal
    tokens_stemmed = [x for x in tokens_stemmed if x not in nlp_utils.stopwords]
    return tokens_stemmed


@common_utils.timing
def generate_ngrams(df):
    # unigram
    print("generate unigram")
    df["title_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_title"]), axis=1))
    df["search_term_unigram"] = list(df.apply(lambda x: preprocess_data(x["search_term"]), axis=1))

    # bigram
    print("generate bigram")
    join_str = "_"
    df["title_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
    df["search_term_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["search_term_unigram"], join_str), axis=1))
    # trigram
    print("generate trigram")
    join_str = "_"
    df["search_term_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["search_term_unigram"], join_str), axis=1))
    df["title_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))


@common_utils.timing
def generate_product_ngrams(df):
    print("Generate unigram")
    df["description_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_description"]), axis=1))
    print("Generate bigram")
    join_str = "_"
    df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
    print("Generate trigram")
    join_str = "_"
    df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))


@common_utils.timing
def generate_brand_ngrams(df):
    print("Generate brand unigram")
    df["brand_unigram"] = list(df.apply(lambda x: preprocess_data(x["brand"]), axis=1))
    print("Generate brand bigram")
    join_str = "_"
    df["brand_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["brand_unigram"], join_str), axis=1))
    print("Generate brand trigram")
    join_str = "_"
    df["brand_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["brand_unigram"], join_str), axis=1))


if __name__ == "__main__":
    token_pattern = re.compile(r"(?u)\b\w\w+\b", flags=re.UNICODE | re.LOCALE)

    ###############
    ## Load Data ##
    ###############
    # load data
    print("==================================================")
    print("Load data...")

    df_train = pd.read_csv(config.file_preprocess_clean_train, engine='python')
    df_test = pd.read_csv(config.file_preprocess_clean_test, engine='python')
    df_product = pd.read_csv(config.file_preprocess_clean_product, engine='python')
    df_brand = pd.read_csv(config.file_preprocess_clean_brand, engine='python').fillna('')

    print("Done.")

    #######################
    ## Generate ngrams   ##
    #######################

    # X = pd.DataFrame(df_train[:100])
    # Y = pd.DataFrame(df_test[:100])
    # Z = pd.DataFrame(df_product[:100])
    # generate_ngrams(X)
    # generate_ngrams(Y)
    # generate_product_ngrams(Z)
    # df_train = X
    # df_test = Y
    # df_product = Z

    print("Generate n-grams for training dataset ...")
    generate_ngrams(df_train)

    print("Generate n-grams for testing dataset ...")
    generate_ngrams(df_test)

    print("Generate n-grams for product dataset ...")
    generate_product_ngrams(df_product)

    print("Generate n-grams for brand dataset ...")
    generate_brand_ngrams(df_brand)

    df_train.drop(['product_title', 'search_term'], inplace=True, axis=1)
    df_test.drop(['product_title', 'search_term'], inplace=True, axis=1)
    df_product.drop(['product_description'], inplace=True, axis=1)
    df_brand.drop(['brand'], inplace=True, axis=1)

    ######################
    ## Merge Datasets   ##
    ######################
    # Merge training data with product descriptions
    df_train = pd.merge(df_train, df_product, how='left', on='product_uid')
    df_train = pd.merge(df_train, df_brand, how='left', on='product_uid')
    df_test = pd.merge(df_test, df_product, how='left', on='product_uid')
    df_test = pd.merge(df_test, df_brand, how='left', on='product_uid')

    ###############
    ## Save Data ##
    ##############
    print("Save data...")

    with open(config.file_preprocess_ngrams_train, "wb") as f:
        pickle.dump(df_train, f, -1)
    with open(config.file_preprocess_ngrams_test, "wb") as f:
        pickle.dump(df_test, f, -1)
    with open(config.file_preprocess_ngrams_product, "wb") as f:
        pickle.dump(df_product, f, -1)

    print("Done.")
