import os
import re 
import numpy as np
import pandas as pd 
import gc
import time
import warnings
import datetime

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import model_from_json

import data_loader
import model_loader
from utils import Timer


if __name__ == "__main__":
    ## INPUT FILES
    DATA_DIR = "data"
    TRAIN_DATA_FILE = os.path.join(DATA_DIR, "transformed_train.csv")
    TEST_DATA_FILE = os.path.join(DATA_DIR, "transformed_test.csv")
    train_csv_path = "data/transformed_train_set.csv"
    val_csv_path = "data/transformed_val_set.csv"
    test_csv_path = "data/transformed_test_set.csv"

    ## Load already splitted and transformed data
    print("Loading the data")
    data = pd.read_csv(TRAIN_DATA_FILE)
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)
    dt = datetime.datetime.now().strftime("%Y-%m-%d")

    ## Fill null comment with no commment
    processed_comments_data = data["comment_text"].fillna("no commment").tolist()
    processed_comments_train = train_df["comment_text"].fillna("no commment").tolist()
    processed_comments_val = val_df["comment_text"].fillna("no commment").tolist()
    processed_comments_test = test_df["comment_text"].fillna("no commment").tolist()

    # Initialize tokenizer
    max_features = 20000
    tokenizer = Tokenizer(num_words=max_features)
    # Fit tokenizer on comments, create a dictionary of word index for each comment
    tokenizer.fit_on_texts(processed_comments_data)
    # Transform comments into lists of index/ list of lists of index 
    tokenized_list_train = tokenizer.texts_to_sequences(processed_comments_train)
    tokenized_list_val = tokenizer.texts_to_sequences(processed_comments_val)
    tokenized_list_test = tokenizer.texts_to_sequences(processed_comments_test)

    ## Pad embedding 
    max_len = 170
    X_train = pad_sequences(tokenized_list_train, max_len)
    X_val = pad_sequences(tokenized_list_val, max_len)
    X_test = pad_sequences(tokenized_list_test, max_len)
    label_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_train = train_df[label_classes].values
    y_val = val_df[label_classes].values
    y_test = test_df[label_classes].values

    ################
    # BASELINE MODEL
    ################

    ## Train model
    t = Timer("Training Baseline Model")
    t.start()
    batch_size = 128
    epochs = 5
    baseline_model = model_loader.load_lstm_model(max_len=max_len, tokenizer=tokenizer,embed_size=128, trainable=True, max_features=max_features)
    baseline_model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    t.stop()

    ## Export model
    model_path = f"models/baseline_model_{dt}_{epochs}.h5"
    baseline_model.save(model_path)
    model_weights_path = f"models/baseline_model_{dt}_{epochs}_weights.h5"
    model_loader.save_dl_model_weights(baseline_model, model_weights_path)
    print(f"Saved model {model_path} to disk")

    ################
    # GLOVE MODEL
    ################

    ## Train model
    t = Timer("Training Glove Model")
    t.start()
    glove_embedding_matrix = data_loader.load_embedding_matrix("glove",tokenizer)
    t.toc("Loading embeddings")
    dt = datetime.datetime.now().strftime("%Y-%m-%d")
    batch_size = 128
    epochs = 5
    glove_model = model_loader.load_lstm_model(max_len=max_len, tokenizer=tokenizer,embed_size=25,embed_matrix=glove_embedding_matrix, trainable=False, max_features=max_features)
    glove_model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    t.stop()

    ## Export model
    model_path = f"models/glove_model_{dt}_{epochs}.h5"
    glove_model.save(model_path)
    model_weights_path = f"models/glove_model_{dt}_{epochs}_weights.h5"
    model_loader.save_dl_model_weights(glove_model, model_weights_path)
    print(f"Saved model {model_path} to disk")

    ################
    # GLOVE MODEL
    ################
    
    ## Train model
    t = Timer("Training Fasttext Model")
    t.start()
    fasttext_embedding_matrix = data_loader.load_embedding_matrix("fasttext",tokenizer)
    t.toc("Loading embeddings")

    batch_size = 128
    epochs = 5
    fasttext_model = model_loader.load_lstm_model(max_len=max_len, tokenizer=tokenizer,embed_size=300,embed_matrix=fasttext_embedding_matrix, trainable=False, max_features=max_features)
    fasttext_model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    t.stop()

    ## Export model
    model_path = f"models/fasttext_model_{dt}_{epochs}.h5"
    fasttext_model.save(model_path)
    model_weights_path = f"models/fasttext_model_{dt}_{epochs}_weights.h5"
    model_loader.save_dl_model_weights(fasttext_model, model_weights_path)
    print(f"Saved model {model_path} to disk")

    """
    BASELINE MODEL
    Train on 177351 samples, validate on 21110 samples
    Epoch 1/5
    177351/177351 [==============================] - 590s 3ms/step - loss: 0.1214 - accuracy: 0.9532 - val_loss: 0.1781 - val_accuracy: 0.9330
    Epoch 2/5
    177351/177351 [==============================] - 541s 3ms/step - loss: 0.0531 - accuracy: 0.9805 - val_loss: 0.2141 - val_accuracy: 0.9313
    Epoch 3/5
    177351/177351 [==============================] - 578s 3ms/step - loss: 0.0382 - accuracy: 0.9863 - val_loss: 0.2664 - val_accuracy: 0.9291
    Epoch 4/5
    177351/177351 [==============================] - 550s 3ms/step - loss: 0.0285 - accuracy: 0.9899 - val_loss: 0.3333 - val_accuracy: 0.9272
    Epoch 5/5
    177351/177351 [==============================] - 664s 4ms/step - loss: 0.0222 - accuracy: 0.9922 - val_loss: 0.4122 - val_accuracy: 0.9263

    Loading the data
    Loading glove embeding with embed_size 25
    Loaded 1193515 word vectors from glove
    Embedded 80216 common words from glove

    Train on 177351 samples, validate on 21110 samples
    Epoch 1/5
    177351/177351 [==============================] - 502s 3ms/step - loss: 0.2809 - accuracy: 0.8917 - val_loss: 0.2737 - val_accuracy: 0.8969
    Epoch 2/5
    177351/177351 [==============================] - 452s 3ms/step - loss: 0.2008 - accuracy: 0.9234 - val_loss: 0.2459 - val_accuracy: 0.9115
    Epoch 3/5
    177351/177351 [==============================] - 6764s 38ms/step - loss: 0.1728 - accuracy: 0.9341 - val_loss: 0.2460 - val_accuracy: 0.9123
    Epoch 4/5
    177351/177351 [==============================] - 1187s 7ms/step - loss: 0.1570 - accuracy: 0.9402 - val_loss: 0.2343 - val_accuracy: 0.9133
    Epoch 5/5
    177351/177351 [==============================] - 752s 4ms/step - loss: 0.1475 - accuracy: 0.9444 - val_loss: 0.2327 - val_accuracy: 0.9162
    Saved model models/glove_model_2020-04-01.h5 to disk

    Loading the data
    Loading fasttext embeding with embed_size 300
    Loaded 110999 word vectors from fasttext
    Embedded 58967 common words from fasttext
    Train on 177351 samples, validate on 21110 samples
    Epoch 1/5
    177351/177351 [==============================] - 908s 5ms/step - loss: 0.1643 - accuracy: 0.9380 - val_loss: 0.1802 - val_accuracy: 0.9233
    Epoch 2/5
    177351/177351 [==============================] - 983s 6ms/step - loss: 0.0908 - accuracy: 0.9662 - val_loss: 0.1883 - val_accuracy: 0.9255
    Epoch 3/5
    177351/177351 [==============================] - 1099s 6ms/step - loss: 0.0750 - accuracy: 0.9716 - val_loss: 0.2131 - val_accuracy: 0.9264
    Epoch 4/5
    177351/177351 [==============================] - 964s 5ms/step - loss: 0.0666 - accuracy: 0.9745 - val_loss: 0.1971 - val_accuracy: 0.9247
    Epoch 5/5
    177351/177351 [==============================] - 1198s 7ms/step - loss: 0.0607 - accuracy: 0.9767 - val_loss: 0.2026 - val_accuracy: 0.9292
    Saved model models/fasttext_model_2020-04-01.h5 to disk

    """


