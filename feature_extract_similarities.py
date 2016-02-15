import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

import config
import feature_utils
import common_utils


####################
### Similarities ###
####################

def calc_cosine_dist(text_a ,text_b, vect):
    return pairwise_distances(vect.transform([text_a]), vect.transform([text_b]), metric='cosine')[0][0]

def calc_set_intersection(text_a, text_b):
    a = set(text_a.split())
    b = set(text_b.split())
    return len(a.intersection(b)) *1.0 / len(a)


#####################################
## Extract basic distance features ##
#####################################
@common_utils.timing
def extract_cosine_feat(df):
    cosine_orig = []
    cosine_stem = []
    cosine_desc = []
    set_stem = []
    for i, row in df.iterrows():
        cosine_orig.append(calc_cosine_dist(row['search_term'], row['product_title'], tfv_orig))
        cosine_stem.append(calc_cosine_dist(row['search_term_stem'], row['title_stem'], tfv_stem))
        cosine_desc.append(calc_cosine_dist(row['search_term_stem'], row['description_stem'], tfv_desc))
        set_stem.append(calc_set_intersection(row['search_term_stem'], row['title_stem']))
    df['cosine_st_orig'] = cosine_orig
    print("cosine_st_orig shape:{0}".format(df['cosine_st_orig'].shape))
    df['cosine_st_t_stem'] = cosine_stem
    df['cosine_st_d_stem'] = cosine_desc
    df['set_st_t_stem'] = set_stem


if __name__ == "__main__":
    ###############
    ## Load Data ##
    ###############
    # load data
    print("Load data...")
    with open(config.file_stem_feat_train, "rb") as f:
        df_train = pickle.load(f)
    with open(config.file_stem_feat_test, "rb") as f:
        df_test = pickle.load(f)
    print("Done.")

    # vectorizers for similarities
    print('fit vectorizers')
    tfv_orig = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    tfv_stem = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    tfv_desc = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    tfv_orig.fit(
        list(df_train['search_term'].values) +
        list(df_test['search_term'].values) +
        list(df_train['product_title'].values) +
        list(df_test['product_title'].values)
    )
    tfv_stem.fit(
        list(df_train['search_term_stem'].values) +
        list(df_test['search_term_stem'].values) +
        list(df_train['title_stem'].values) +
        list(df_test['title_stem'].values)
    )
    tfv_desc.fit(
        list(df_train['search_term_stem'].values) +
        list(df_test['search_term_stem'].values) +
        list(df_train['description_stem'].values) +
        list(df_test['description_stem'].values)
    )

    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate distance features...")

    extract_cosine_feat(df_train)
    extract_cosine_feat(df_test)

    feat_names = ['id', 'cosine_st_orig', 'cosine_st_t_stem', 'cosine_st_d_stem', 'set_st_t_stem']

    if not os.path.exists(config.path_cosine_features):
        os.makedirs(config.path_cosine_features)

    with open(config.file_cosine_feat_train, "wb") as f:
        pickle.dump(df_train[feat_names], f, -1)
    with open(config.file_cosine_feat_test, "wb") as f:
        pickle.dump(df_test[feat_names], f, -1)

    # save feat names
    print("Feature names are stored in %s" % config.file_cosine_feat_name)
    # dump feat name
    feature_utils.dump_feat_name(feat_names, config.file_cosine_feat_name)

    print("All Done.")
