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
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import model_from_json

def load_embedding_matrix(emb_type, tokenizer):
    """Return embeding dictionary according to the emb_type"""
    ## load different embedding file depending on which embedding 
    if(emb_type=="glove"):
        EMBEDDING_FILE="embeddings/glove.twitter.27B.25d.txt"
        embed_size = 25
    elif(emb_type=="fasttext"):
        EMBEDDING_FILE="embeddings/wiki.simple.vec"
        embed_size = 300
    elif(emb_type == "word2vec"):
        word2vecDict = word2vec.KeyedVectors.load_word2vec_format("embeddings/GoogleNews-vectors-negative300.bin", binary=True)
        embed_size = 300
    
    print(f"Loading {emb_type} embeding with embed_size {embed_size}")

    #Transfer the embedding weights into a dictionary by iterating through every line of the file.
    if (emb_type == "glove" or emb_type == "fasttext"):
        embeddings_index = dict()
        f = open(EMBEDDING_FILE)
        for idx, line in enumerate(f):
            #split up line into an indexed array
            embed_array = line.split()
            ## if embed array contains 1 word
            if len(embed_array) == (embed_size+1):
                #first index is word, store the rest of the embed_array in the array as a new array
                word = embed_array[0]
                coefs = np.asarray(embed_array[1:], dtype='float32')
                embeddings_index[word] = coefs 
            ## if the array contains 2 words ## glove
            elif len(embed_array) == (embed_size+2):
                #first two index are words, store the rest of the embed_array in the array as a new array
                word = " ".join(embed_array[:2])
                coefs = np.asarray(embed_array[2:], dtype='float32')
                embeddings_index[word] = coefs 
            ## if there are no word ##fasttext
            elif len(embed_array) == embed_size:
                word = ""
                coefs = np.asarray(embed_array, dtype='float32')
                embeddings_index[""] = coefs #50 dimensions
        f.close()
    else:
        embeddings_index = dict()
        for word in word2vecDict.wv.vocab:
            embeddings_index[word] = word2vecDict.word_vec(word)

    print(f"Loaded {len(embeddings_index)} word vectors from {emb_type}")

    #We get the mean and standard deviation of the embedding weights so that we could maintain the 
    #same statistics for the rest of our own random generated weights. 
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    nb_words = len(tokenizer.word_index)

    #We are going to set the embedding size to the pretrained dimension as we are replicating it.
    #the size will be Number of Words in Vocab X Embedding Size
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    #With the newly created embedding matrix, we'll fill it up with the words that we have in both 
    #our own dictionary and loaded pretrained embedding. 
    embedded_count = 0
    for word, idx in tokenizer.word_index.items():
        idx -= 1
        #then we see if this word is in word2vec/glove/fasttext's dictionary, if yes, get the corresponding weights
        #and store inside the embedding matrix that we will train later on.
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[idx] = embedding_vector
            embedded_count += 1

    print(f"Embedded {embedded_count} common words from {emb_type}")

    del(embeddings_index)

    return embedding_matrix

def generate_toxic_prediction(y_prob_array):
    y_pred_col0 = np.where(y_prob_array[:,0] > 0.4, 1, 0)
    y_pred_col1 = np.where(y_prob_array[:,1] > 0.5, 1, 0)
    y_pred_col2 = np.where(y_prob_array[:,3] > 0.3, 1, 0)
    y_pred_col3 = np.where(y_prob_array[:,3] > 0.6, 1, 0)
    y_pred_col4 = np.where(y_prob_array[:,4] > 0.5, 1, 0)
    y_pred_col5 = np.where(y_prob_array[:,5] > 0.5, 1, 0)

    y_pred_col0 = np.expand_dims(y_pred_col0, axis=1)
    y_pred_col1 = np.expand_dims(y_pred_col1, axis=1)
    y_pred_col2 = np.expand_dims(y_pred_col2, axis=1)
    y_pred_col3 = np.expand_dims(y_pred_col3, axis=1)
    y_pred_col4 = np.expand_dims(y_pred_col4, axis=1)
    y_pred_col5 = np.expand_dims(y_pred_col5, axis=1)

    y_pred_array = np.concatenate((y_pred_col0, y_pred_col1, y_pred_col2, y_pred_col3, y_pred_col4, y_pred_col5), axis=1)
    return y_pred_array