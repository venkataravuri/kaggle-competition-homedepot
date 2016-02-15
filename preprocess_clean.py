"""
__file__
    preprocess.py
__description__
    Adopted from @Chenglong Chen's code < https://github.com/ChenglongChen/Kaggle_CrowdFlower >
    This file pre-processes data.
__author__
    Venkata Ravuri < venkat@nikhu.com >
"""
import pickle
import pandas as pd

import config
import nlp_utils

if __name__ == "__main__":
    ###############
    ## Load Data ##
    ###############
    print("Load data...")

    df_train = pd.read_csv(config.path_raw + config.file_train, engine='python')
    df_test = pd.read_csv(config.path_raw + config.file_test, engine='python')
    df_product = pd.read_csv(config.path_raw + config.file_product_descriptions, engine='python')
    df_attr = pd.read_csv(config.path_raw + config.file_attributes, engine='python')

    # number of train/test samples
    num_train, num_test = df_train.shape[0], df_test.shape[0]

    print("Done.")

    ######################
    ## Pre-process Data ##
    ######################
    print("Pre-process data...")

    ## clean text
    clean = lambda line: nlp_utils.clean_text(line)

    print("Processing " + config.path_raw + config.file_product_descriptions)
    df_product.product_description = df_product.product_description.apply(clean)

    print("Processing " + config.path_raw + config.file_train)
    df_train.search_term = df_train.search_term.apply(clean)
    df_train.product_title = df_train.product_title.apply(clean)

    print("Processing " + config.path_raw + config.file_test)
    df_test.search_term = df_test.search_term.apply(clean)
    df_test.product_title = df_test.product_title.apply(clean)

    print("Extracting brand...")
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"}).fillna('')
    print(df_brand.info())
    df_brand.brand = df_brand.brand.apply(clean)

    print("Done.")

    ###############
    ## Save Data ##
    ###############
    print("Save data...")

    df_train.to_csv(config.file_preprocess_clean_train, index=False)
    df_test.to_csv(config.file_preprocess_clean_test, index=False)
    df_product.to_csv(config.file_preprocess_clean_product, index=False)
    df_brand.to_csv(config.file_preprocess_clean_brand, index=False)

    print("Done.")
