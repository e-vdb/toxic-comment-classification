#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:54:12 2021

@author: Emeline
"""

import pandas as pd
import re
import pickle


def clean_text(text):
  text=text.lower()
  text = re.sub(r"can't", "can not ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"\'ll", " will ", text)
  text=re.sub('\W',' ',text)
  text=text.strip(' ')
  return text


categories=['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']

# load the model from disk
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


print("Enter your comment")
comment=input()
print(pd.DataFrame(loaded_model.predict([clean_text(comment)]),columns=categories))