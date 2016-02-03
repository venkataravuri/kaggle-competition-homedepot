# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:03:57 2016

@author: VRavuri
"""
# import required libraries
import time
import os
import pandas as pd
import pickle
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import nltk

#nltk.download()
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)
stemmer = SnowballStemmer("english")

def correct_text(s):
    s = s.replace('in.', 'inch')
    s = s.replace('sq.', 'square')
    s = s.replace('ft.', 'feet')
    s = s.replace('cu.', 'cubic')
    s = s.replace('lbs.', 'pounds')
    s = s.replace('gal.', 'gallons')
    s = s.replace('W x', 'width')
    s = s.replace('H x', 'height')
    s = s.replace('Ah ', 'amphere')
    return s

def load_data():
    print("loading data...")
    start_time = time.time()
    
    df_train = pd.read_csv(path_raw + file_train, engine='python')
    df_test = pd.read_csv(path_raw + file_test, engine='python')
    df_product_desc = pd.read_csv(path_raw + file_product_descriptions, engine='python')
    print("loading data took %.2f seconds.\n" % (time.time() - start_time))
    
    return  (df_train, df_test, df_product_desc)

def clean_text(text):
    text = correct_text(text)
    # Remove '&nbsp;' from the text content before HTML tags strip off.
    text.replace('&nbsp;', ' ')
    # Remove HTML tags
    text = BeautifulSoup(text, "lxml").get_text(separator=" ")
    # Replace all punctuation and special characters by space
    text.replace("[ &<>)(_,.;:!?/-]+", " ")
    # Remove the apostrophe's
    text.replace("'s\\b", "")
    # Remove the apostrophe
    text.replace("[']+", "")
    # Remove the double quotes
    text.replace("[\"]+", "")
    # Convert to lower case, split into individual words
    words = text.lower().split()
    # Stemming words
    stemmed_words = [stemmer.stem(word) for word in words]
    # Remove stop words stopwords("english")
    meaningful_words = [w for w in stemmed_words if not w in stopwords]
    # Join the words back into one string separated by space, and return the result.
    return( " ".join( meaningful_words ))

def clean_dataset(df_dataset):
    df_dataset.search_term = df_dataset.search_term.apply(lambda text: clean_text(text))
    df_dataset.product_title = df_dataset.product_title.apply(lambda text: clean_text(text))
    return df_dataset    
    
df_train, df_test, df_product_desc = load_data()

print("Cleaning product description dataset...")
start_time = time.time()
df_product_desc.product_description = df_product_desc.product_description.apply(lambda text: clean_text(text))
print("Cleaning product description dataset took %s seconds." % (time.time() - start_time))

# Merge training data with product descriptions
df_train_product_desc = pd.merge(df_train, df_product_desc, how='left', on='product_uid')
df_test_product_desc = pd.merge(df_test, df_product_desc, how='left', on='product_uid')

print("Cleaning train & test datasets...")
start_time = time.time()
df_train_product_desc = clean_dataset(df_train_product_desc)
df_test_product_desc = clean_dataset(df_test_product_desc)
print("Cleaning datasets took %s seconds." % (time.time() - start_time))

# Persist cleaned datasets to /processed-clean folder.
df_train_product_desc.to_csv(os.path.join(path_processed_clean, file_train_product_descriptions), index=False)
df_test_product_desc.to_csv(os.path.join(path_processed_clean, file_test_product_descriptions), index=False)
    