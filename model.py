#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:54:12 2021

@author: Emeline
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
import pickle


def clean_text(text):
  text=text.lower()
  text = re.sub(r"can't", "can not ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"\'ll", " will ", text)
  text=re.sub('\W',' ',text)
  text=text.strip(' ')
  return text


filepath='data/train.csv'
data_train=pd.read_csv(filepath)
df_train=data_train.copy()

categories=['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
X=df_train['comment_text']
X_clean=X.map(lambda com:clean_text(com))
y=df_train[categories]

X_train,X_test,y_train,y_test=train_test_split(X_clean,y,random_state=0)
modelSVC=Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))),('svc',MultiOutputClassifier(LinearSVC(C=10)))])
modelSVC.fit(X_train,y_train)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(modelSVC, open(filename, 'wb'))
print('Model saved')
