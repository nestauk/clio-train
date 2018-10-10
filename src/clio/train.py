# Set seeds
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import json
import math
import pickle
import numpy as np
import pandas as pd

from collections import defaultdict, Counter

from utils import *
from make_training_set import *
from modelling import *
from distances import *

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

# Read data
df = pd.read_csv('../../data/processed/df_full.csv')

# Training set
mak_subset = make_subset(df, n=3500, replace=False)
print(mak_subset.columns)
# texts = [' '.join(text) for text in mak_subset.proc_abs]
texts = list(mak_subset.proc_abs)
# Encode labels
encoded_labels = label_encoding(np.array(mak_subset['Topic']))
# Unique classes and document indices for each Topic.
num_classes = len(np.unique(encoded_labels))
topic_indices = [np.where(encoded_labels == i)[0] for i in range(num_classes)]
print('Number of classes: {}'.format(num_classes))

# Text preprocessing
encoded_docs = keras_tokenizer(texts, maxlen=200, sequences=True, num_words=100000)
# Create dataset
pairs, tr_y = create_pairs(encoded_docs, topic_indices, num_classes)
arr1 = split_pairs(pairs, 0)
arr2 = split_pairs(pairs, 1)
target = np.array(tr_y)

# Model
bn = BaseNetwork()
# network input
input_shape = (arr1.shape[1], )
# Defining the input for every branch of the Siamese network
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

lstm = bn.lstm(input_shape)
input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))

processed_a = lstm(input_a)
processed_b = lstm(input_b)

# Model output: The Euclidean distance between input_a and input_b
distance = Lambda(euclidean_distance, output_shape=dist_output_shape)([processed_a, processed_b])

# Concatenate the tensor to create a model
# Alternatively, I could possibly use: keras.layers.Concatenate(axis=-1)
model = Model(inputs=[input_a, input_b], outputs=distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([arr1, arr2], target,
          batch_size=64,
          epochs=3,
          validation_split=0.2)

# compute final accuracy on training and test sets
y_pred = model.predict([arr1, arr2])
tr_acc = compute_accuracy(target, y_pred)

print('* Accuracy on training set: {}'.format(100 * tr_acc))
