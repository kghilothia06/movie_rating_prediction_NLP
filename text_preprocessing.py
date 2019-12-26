# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 19:41:38 2019

@author: KUSH
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk

df = pd.read_csv('Train.csv')

#text cleaning
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
sw = set(stopwords.words('english'))
corpus = []
for i in range(0, 40000):
    review = re.sub('[^a-zA-Z0-9]', ' ', df['review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in sw]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

y = df.iloc[:, 1].values

