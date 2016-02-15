import pickle
import pandas as pd
from nltk import TreebankWordTokenizer
from nltk.stem import wordnet

import config
import os
import common_utils


def text_preprocessor(x):
    '''
    Get one string and clean\lemm it
    '''
    tokens = toker.tokenize(x)
    return " ".join([lemmer.lemmatize(z) for z in tokens])


@common_utils.timing
def lemmatizing(df):
    # lemm title
    print("Generating title_stem")
    df['title_stem'] = df['product_title'].apply(text_preprocessor)
    # lemm query
    print("Generating search_term_stem")
    df['search_term_stem'] = df['search_term'].apply(text_preprocessor)


@common_utils.timing
def lemmatizing_product(df):
    # lemm description
    print("Generating description_stem")
    df['description_stem'] = df['product_description'].apply(text_preprocessor)


if __name__ == "__main__":
    toker = TreebankWordTokenizer()
    lemmer = wordnet.WordNetLemmatizer()

    ###############
    ## Load Data ##
    ###############
    # load data
    print("==================================================")
    print("Load data...")

    df_train = pd.read_csv(config.file_preprocess_clean_train, engine='python')
    df_test = pd.read_csv(config.file_preprocess_clean_test, engine='python')
    df_product = pd.read_csv(config.file_preprocess_clean_product, engine='python')

    print("Done.")

    #######################
    ## Generate ngrams   ##
    #######################

    print("Lemmatizing for training dataset ...")
    lemmatizing(df_train)

    print("Lemmatizing for testing dataset ...")
    lemmatizing(df_test)

    print("Lemmatizing for product dataset ...")
    lemmatizing_product(df_product)

    # df_train.drop(['product_title', 'search_term'], inplace=True, axis=1)
    # df_test.drop(['product_title', 'search_term'], inplace=True, axis=1)
    # df_product.drop(['product_description'], inplace=True, axis=1)

    ######################
    ## Merge Datasets   ##
    ######################
    # Merge training data with product descriptions
    df_train = pd.merge(df_train, df_product, how='left', on='product_uid')
    df_test = pd.merge(df_test, df_product, how='left', on='product_uid')

    ###############
    ## Save Data ##
    ##############
    print("Save data...")

    if not os.path.exists(config.path_stem_features):
        os.makedirs(config.path_stem_features)

    with open(config.file_stem_feat_train, "wb") as f:
        pickle.dump(df_train, f, -1)
    with open(config.file_stem_feat_test, "wb") as f:
        pickle.dump(df_test, f, -1)
    with open(config.file_stem_feat_product, "wb") as f:
        pickle.dump(df_product, f, -1)

    print("Done.")
