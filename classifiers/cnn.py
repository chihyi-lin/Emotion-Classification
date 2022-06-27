# Preprocessing data to the model
from preprocessing.preprocessing_cnn import *
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
# Keras layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten, MaxPooling1D

import numpy as np


# set parameters:
vocab_size = 10000
oov_tok = "<OOV>"
trunc_type = "post"
# Max input length (number of words)
maxlen = 300
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 100
epochs = 10

X_train = Preprocessor('../data/isear-train.csv').text
y_train = Preprocessor('../data/isear-train.csv').label
X_test = Preprocessor('../data/isear-val.csv').text
y_test = Preprocessor('../data/isear-val.csv').label

# Vectorization: turn tokens into matrices
# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
# Generate the word index dictionary for the training sentences
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
# Generate and pad the training sequences
sequences_train = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(sequences_train, maxlen=maxlen, truncating=trunc_type)

# Generate and pad the test sequences
sequences_test = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(sequences_test, maxlen=maxlen)

# Encoding labels into arrays
encoded_labels = {'joy': 0,
                  'fear': 1,
                  'guilt': 2,
                  'anger': 3,
                  'shame': 4,
                  'disgust': 5,
                  'sadness': 6}
y_train_array = [encoded_labels[x] for x in y_train]
y_test_array = [encoded_labels[x] for x in y_test]
y_train = to_categorical(y_train_array)
y_test = to_categorical(y_test_array)


def create_embedding_matrix(filepath, word_index, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix





