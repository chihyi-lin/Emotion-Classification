# Preprocessing data to the model
from preprocessing.preprocessing_cnn import *
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Keras layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D

import numpy as np


# set parameters:
vocab_size = 7523  # == the len(vocab_index)+1. Should set it larger for training set?
oov_tok = "<OOV>"
trunc_type = "post"
# Max input length (number of words) for each text
maxlen = 300
batch_size = 256   #32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 100
epochs = 10

train_data = Preprocessor('../data/isear-train.csv')
X_train = train_data.clean_text
y_train = train_data.label
test_data = Preprocessor('../data/isear-val.csv')
X_test = test_data.clean_text
y_test = test_data.label
# Convert to np.array
X_train = np.array(X_train)
# y_train = np.array(y_train)
X_test = np.array(X_test)
# y_test = np.array(y_test)



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

# # # M1 - Encoding labels into arrays
# encoded_labels = {'joy': 0,
#                   'fear': 1,
#                   'guilt': 2,
#                   'anger': 3,
#                   'shame': 4,
#                   'disgust': 5,
#                   'sadness': 6}
# y_train_array = [encoded_labels[x] for x in y_train]
# y_test_array = [encoded_labels[x] for x in y_test]
# y_train = to_categorical(y_train_array)
# y_test = to_categorical(y_test_array)

# M2 - Encoder

def create_embedding_matrix(filepath, word_index, embedding_dims):
    """
    Traverse the glove file of a specific dimension and compare each word with all words in the dictionary,
    if a match occurs, copy the equivalent vector from the glove and paste into embedding_matrix at the corresponding index.
    :param filepath:
    :param word_index:
    :param embedding_dims:
    :return:
    """

    embeddings_index = {}
    with open(filepath, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    embedding_matrix = np.zeros((vocab_size, embedding_dims))
    for word, index in word_index.items():
        if index > vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix


embedding_matrix = create_embedding_matrix('../glove.6B/glove.6B.50d.txt', word_index, embedding_dims)


# # M1 - Categorical: Embedding layer before the actaul BLSTM
# embedding_layer = Embedding(vocab_size,
#                          embedding_dims,
#                          input_length = maxlen,
#                          weights = embedding_matrix,
#                          trainable=False)
# model = Sequential()
# model.add(embedding_layer)
# model.add(Conv1D(filters, kernel_size, activation='relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(num_classes=7, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
#
# model.fit(X_train_pad, y_train,
#                  batch_size=batch_size,
#                  epochs=epochs,
#                  validation_data=(X_test_pad,y_test))

# M2-Sequential (from Lukas)
model = Sequential()
# model.add(embedding_layer)
model.add(Embedding(vocab_size, embedding_dims, input_length=maxlen, weights=[embedding_matrix], trainable=False))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

model.fit(X_train_pad, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test_pad, y_test))
# Inspect unseen words


