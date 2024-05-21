# preprocess.py
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

# Load the dataset
df = pd.read_csv('adult.csv')

# Drop unnecessary columns
df = df.drop(['fnlwgt', 'educational-num'], axis=1)

# Replace missing values with NaN
df = df.replace("?", np.NaN)

# Fill NaN with mode value
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

# Discretize and encode categorical variables
df.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married',
            'not married', 'not married', 'not married'], inplace=True)

category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                'relationship', 'gender', 'native-country', 'income']
labelEncoder = preprocessing.LabelEncoder()

for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])

# Split the data into attributes and labels
X = df.values[:, 0:12]
Y = df.values[:, 12]

# Save preprocessed data and encoders
with open('data.pkl', 'wb') as f:
    pickle.dump((X, Y, labelEncoder), f)
