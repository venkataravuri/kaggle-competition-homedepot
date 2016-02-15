"""
__file__
    config.py
__description__
    Adopted from @Chenglong Chen's code < https://github.com/ChenglongChen/Kaggle_CrowdFlower >
    This file provides global parameter configurations for the project.
__author__
    Venkata Ravuri < venkat@nikhu.com >
"""

# Path parameters
path_data = './data'
path_raw = path_data + '/raw/'
path_preprocess_clean = path_data + '/clean'
path_preprocess_ngrams = path_data + '/ngrams/'
path_id_features = path_data + '/id-feat'
path_counting_features = path_data + '/words-count-feat/'
path_brand_counting_features = path_data + '/brand-count-feat/'
path_distance_features = path_data + '/distance-feat/'
path_tfidf_features = path_data + '/tfidf-feat'
path_basic_tfidf_features = path_data + "/basic-tfidf-feat"
path_stem_features = path_data + "/stem-feat"
path_cosine_features = path_data + "/cosine-feat"
path_submit = path_data + '/submit/'

# File name parameters
file_train = 'train.csv'
file_test = 'test.csv'
file_product_descriptions = 'product_descriptions.csv'
file_attributes = 'attributes.csv'
file_submission_name = 'submission'

file_preprocess_clean_train = "%s/train.clean.csv" % path_preprocess_clean
file_preprocess_clean_test = "%s/test.clean.csv" % path_preprocess_clean
file_preprocess_clean_product = "%s/product.clean.csv" % path_preprocess_clean
file_preprocess_clean_brand = "%s/brand.clean.csv" % path_preprocess_clean

file_preprocess_ngrams_train = "%s/train.ngrams.csv.pkl" % path_preprocess_ngrams
file_preprocess_ngrams_test = "%s/test.ngrams.csv.pkl" % path_preprocess_ngrams
file_preprocess_ngrams_product = "%s/product.ngrams.csv.pkl" % path_preprocess_ngrams

file_words_count_feat_train = "%s/train.words.count.feat.pkl" % path_counting_features
file_words_count_feat_test = "%s/test.words.count.feat.pkl" % path_counting_features

file_brand_count_feat_train = "%s/train.brand.count.feat.pkl" % path_brand_counting_features
file_brand_count_feat_test = "%s/test.brand.count.feat.pkl" % path_brand_counting_features

file_distance_feat_train = "%s/train.distance.feat.pkl" % path_distance_features
file_distance_feat_test = "%s/test.distance.feat.pkl" % path_distance_features

file_tfidf_feat_train = "%s/train.tfidf.feat.pkl" % path_tfidf_features
file_tfidf_feat_test = "%s/test.tfidf.feat.pkl" % path_tfidf_features

file_cosine_feat_train = "%s/train.tfidf.feat.pkl" % path_cosine_features
file_cosine_feat_test = "%s/test.tfidf.feat.pkl" % path_cosine_features

file_stem_feat_train = "%s/train.stem.feat.pkl" % path_stem_features
file_stem_feat_test = "%s/test.stem.feat.pkl" % path_stem_features
file_stem_feat_product = "%s/product.stem.feat.pkl" % path_stem_features


file_stem_feat_name = "%s/stem.feat_name" % path_data
# file to save counting words feat names
file_feat_name = "%s/counting.feat_name" % path_data
file_brand_feat_name = "%s/brand.count.feat_name" % path_data
file_distance_feat_name = "%s/distance.feat_names" % path_data
# file to save tfidf feat names
file_tfidf_feat_name = "%s/intersect_tfidf.feat_names" % path_data
file_basic_tfidf_feat_name = "%s/basic_tfidf.feat_names" % path_data

file_cosine_feat_name = "%s/cosine.feat_names" % path_data

# NLP parameters
stemmer_type = "snowball"
