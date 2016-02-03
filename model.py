# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:05:44 2016

@author: VRavuri
"""
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score


def load_feature_dataset():
    print ("Loading cleaned dataset.")
    start_time = time.time()
    
    df_train_product_desc = pd.read_csv(path_processed_features + file_train_product_descriptions, engine='python')
    df_test_product_desc = pd.read_csv(path_processed_features + file_test_product_descriptions, engine='python') 
    
    print("Loading cleaned datasets took %s seconds." % (time.time() - start_time))
    return (df_train_product_desc, df_test_product_desc)
    
df_train_product_desc, df_test_product_desc = load_feature_dataset()

# Dropping off unnecessary features.
df_train_product_desc = df_train_product_desc.drop(['search_term','product_title','product_description','product_info'],axis=1)
df_test_product_desc = df_test_product_desc.drop(['search_term','product_title','product_description','product_info'],axis=1)

# Stroing id in temporary variable, later used during generating submission file.
id_test = df_test_product_desc['id']


features = ['product_uid', 'len_of_query', 'word_in_title', 'word_in_description']

print ("Total dataset length: {0}".format(len(df_train_product_desc)))

clf = RandomForestRegressor(n_estimators=24, max_depth=7)
kfold = KFold(len(df_train_product_desc), n_folds=10, shuffle=True)

for k, (train_index, test_index) in enumerate(kfold):
    
    print ("Fold-[{0}] Training size: {1}; Testing size: {2}".format(k, len(train_index), len(test_index)))

    X_train = df_train_product_desc.loc[train_index, features]
    y_train = df_train_product_desc.loc[train_index, 'relevance']

    X_test = df_train_product_desc.loc[test_index, features]
    y_test = df_train_product_desc.loc[test_index, 'relevance']
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print ("Fold-[{0}] evaulation score: {1}".format(k, r2_score(y_test, y_pred)))

#y_train = df_train_product_desc['relevance'].values
#X_train = df_train_product_desc.drop(['id','relevance'],axis=1).values
#X_test = df_test_product_desc.drop(['id'],axis=1).values
    
#print ("Generating submission file.")
#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(path_submit + file_submission,index=False)