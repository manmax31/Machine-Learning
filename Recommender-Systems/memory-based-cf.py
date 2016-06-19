# Source: http://online.cambridgecoding.com/notebooks/eWReNYcAfB/implementing-your-own-recommender-systems-in-python-2

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv

# Read the dataset
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

# Sneak Peek of Data Set
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

# Split the dataset into Train and Test
train_data, test_data = cv.train_test_split(df, test_size=0.25)