# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:47:06 2016

@author: VRavuri
"""
# import required libraries
import time
import pandas as pd
import pickle
import numpy as np

def load_clean_data():
    print ("Loading cleaned dataset.")
    start_time = time.time()
    
    df_train_product_desc = pd.read_csv(path_processed_clean + file_train_product_descriptions, engine='python')
    df_test_product_desc = pd.read_csv(path_processed_clean + file_test_product_descriptions, engine='python') 
    
    print ("Loading cleaned datasets took %.2f minutes." % ((time.time() - start_time) / 60.))
    return (df_train_product_desc, df_test_product_desc)

def str_common_word(str1, str2):
	str1, str2 = str1.lower(), str2.lower()
	words, cnt = str1.split(), 0
	for word in words:
		if str2.find(word)>=0:
			cnt+=1
	return cnt

def generate_features(df_dataset):
    print ("Generating custom feature...")
    start_time = time.time()
    # TODO: Refactor

    df_dataset.search_term = df_dataset.search_term.astype(np.str)

    df_dataset['len_of_query'] = df_dataset.search_term.map(lambda x:len(x.split())).astype(np.int64)
    
    df_dataset['product_info'] = df_dataset.search_term + '\t' + df_dataset.product_title + '\t' + df_dataset.product_description
    
    df_dataset['word_in_title'] = df_dataset.product_info.map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
       
    df_dataset['word_in_description'] = df_dataset.product_info.map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

    print ("Generating custom feature took %.2f minutes." % ((time.time() - start_time) / 60.))
    return df_dataset

# Load cleaned train & test dataset.
df_train_product_desc, df_test_product_desc = load_clean_data()

# Generate addtional features.
df_train_product_desc = generate_features(df_train_product_desc)
df_test_product_desc = generate_features(df_test_product_desc)

# Persist enriched datasets to /processed-features folder.
df_train_product_desc.to_csv(path_processed_features + file_train_product_descriptions, index=False) 
df_test_product_desc.to_csv(path_processed_features + file_test_product_descriptions, index=False)
