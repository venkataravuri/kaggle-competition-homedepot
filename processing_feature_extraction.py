# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:47:06 2016

@author: VRavuri
"""
# import required libraries
import time
import pandas as pd
import re
import numpy as np

def load_clean_data():
    print ("Loading cleaned dataset.")
    start_time = time.time()
    
    df_train_product_desc = pd.read_csv(path_processed_clean + file_train_product_descriptions, engine='python')
    df_test_product_desc = pd.read_csv(path_processed_clean + file_test_product_descriptions, engine='python') 
    
    print ("Loading cleaned datasets took %.2f minutes." % ((time.time() - start_time) / 60.))
    return (df_train_product_desc, df_test_product_desc)



def extract_text_features(data):
    print ("Generating custom feature...")
    start_time = time.time()

    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    
    search_tokens_count = 0.0
    data["search_tokens_count"] = 0.0
    data["title_tokens_count"] = 0.0
    data["description_tokens_count"] = 0.0
    
    data["search_tokens_in_title"] = 0.0
    data["search_tokens_in_description"] = 0.0
    
    data["prefix_match_in_title"] = False
    data["second_match_in_title"] = False
    data["suffix_match_in_title"] = False
    
    for i, row in data.iterrows():
        search_term = set(x.lower() for x in token_pattern.findall(row["search_term"]))
        search_tokens_count = len(search_term)
        data.set_value(i, "search_tokens_count", search_tokens_count)
        
        product_title = set(x.lower() for x in token_pattern.findall(row["product_title"]))
        title_tokens_count = len(product_title)
        data.set_value(i, "title_tokens_count", title_tokens_count)
        
        product_description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
        description_tokens_count = len(product_description)
        data.set_value(i, "description_tokens_count", description_tokens_count)
        
        if title_tokens_count > 0:
            if search_tokens_count > 0:
                data["prefix_match_in_title"] = next(iter(search_term)) in product_title
            if search_tokens_count > 1:
                data["second_match_in_title"] = list(search_term)[1] in product_title
            if (search_tokens_count > 2):
                data["suffix_match_in_title"] = list(search_term)[search_tokens_count - 1] in product_title            
            data.set_value(i, "search_tokens_in_title", len(search_term.intersection(product_title))/title_tokens_count)
        if description_tokens_count > 0:
            data.set_value(i, "search_tokens_in_description", len(search_term.intersection(product_description))/description_tokens_count)

    print ("Generating custom feature took %.2f minutes." % ((time.time() - start_time) / 60.))   

# Load cleaned train & test dataset.
df_train_product_desc, df_test_product_desc = load_clean_data()

df_train_product_desc.search_term = df_train_product_desc.search_term.astype(np.str)
df_test_product_desc.search_term = df_test_product_desc.search_term.astype(np.str)

extract_text_features(df_train_product_desc)
extract_text_features(df_test_product_desc)

# Persist enriched datasets to /processed-features folder.
df_train_product_desc.to_csv(path_processed_features + file_train_product_descriptions, index=False) 
df_test_product_desc.to_csv(path_processed_features + file_test_product_descriptions, index=False)
