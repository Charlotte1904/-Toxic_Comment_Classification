import os
import re 
import numpy as np
import pandas as pd 
import gc
import time
import warnings

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras.models import load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import model_from_json


def load_lstm_model(max_len=170, tokenizer={}, embed_size=0, embed_matrix={}, trainable=True, max_features=2000):
    inp = Input(shape=(max_len, ))
    if trainable:
        x = Embedding(max_features, embed_size)(inp)
    else:
        x = Embedding(len(tokenizer.word_index), embed_size,weights=[embed_matrix],trainable=False)(inp)

    x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss="binary_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])
    return model

def load_dl_model(model_path):
    model = load_model(model_path)
    return model

def save_dl_model_weights(model,model_path):
    model_basename = os.path.basename(model_path).split(".")[0]
    # serialize weights to HDF5
    model_json = model.to_json()
    with open(f"models/{model_basename}.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_path)

def load_dl_model_weights(model_path):
    # load weights into new model
    model_basename = os.path.basename(model_path).split(".")[0]
    model_json_file = open(f"{model_basename}.json", "r")
    loaded_model_model_json = model_json_file.read()
    model_json_file.close()
    loaded_model_model = model_from_json(loaded_model_model_json)
    loaded_model_model.load_weights(model_path)
    return loaded_model_model