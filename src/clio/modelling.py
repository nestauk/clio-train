import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv1D, MaxPooling1D, Embedding, LSTM

def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06"""
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances."""
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances."""
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

class BaseNetwork():
    """Base network to be shared across the branches of Siamese architecture.
    This is equivalent to feature extraction).

    """
    def __init__(self):
        pass

    def mlp(self, input_shape):
        """Build a multilayer perceptron."""
        input = Input(shape=input_shape)
        x = Dense(256, activation='relu')(input)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        return Model(input, x)

    def cnn(self, input_shape):
        """Build a Convolutional Neural Network"""
        input = Input(shape=(input_shape))
        x = Conv1D(64, 7, activation='relu')(input)
        x = Conv1D(64, 7, activation='relu')(x)
        x = MaxPooling1D()(x)
        x = Conv1D(128, 7, activation='relu')(x)
        x = Conv1D(128, 7, activation='relu')(x)
        x = MaxPooling1D()(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        return Model(input, x)

    def lstm(self, input_shape):
        """Build a Long Short-Term Memory network."""
        input = Input(shape=(input_shape))
        x = Embedding(input_dim=100000, output_dim=150, input_length=200)(input)
        x = LSTM(16, activation='tanh')(x)
        return Model(input, x)
