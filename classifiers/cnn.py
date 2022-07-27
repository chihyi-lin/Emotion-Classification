"""Author: Chih-Yi Lin"""
# Preprocessing data to the model
from preprocessing.preprocessing_cnn import *
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
# Keras layers and early stopping
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
# For building multi channel
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from numpy import array


class CNN:
    """
    class CNN has two architectures:
    single channel: taking 1 filter size (n-grams with the same length)
    multi-channels: taking different filter sizes (n-grams with different lengths)
    """
    def __init__(self, embedding_dims, filters, filter_size, hidden_dims, batch_size, epochs):

        # calculated from word index dictionary, and adding 1 for out of vocabulary token (OOV)
        self.vocab_size = 7597
        self.oov_tok = "<OOV>"
        # sentence longer than maxlen will be truncated from the end
        self.trunc_type = "post"
        # Max input length (number of words) for each text is calculated in preprocessing
        self.maxlen = 180

        self.embedding_dims = embedding_dims
        self.filters = filters
        self.filter_size = filter_size
        # the number of neurons in the hidden layer
        self.hidden_dims = hidden_dims

        # number of different filter sizes in multi-channel
        self.n_filter_size = None

        self.batch_size = batch_size
        self.epochs = epochs

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        # label names in strings
        self.class_name = None
        self.word_index = None
        self.embedding_matrix = None

        self.model = None

        self.output_path = '../trained_classifiers/'
        # use y_pred (predicted labels of test set) and y_true (actual labels in test set) for evaluation
        self.y_pred = []
        self.y_true = None

    def preprocess(self, removal):
        """
        Pre-process data to fit into model,
        convert tokens into matrices,
        generate word_index dictionary,
        generate and padding texts into sequences of integers
        :return: None
        """
        X_train, y_train = X_and_y(Preprocessor('../data/isear-train.csv'), removal)
        X_val, y_val = X_and_y(Preprocessor('../data/isear-val.csv'), removal)
        X_test, y_test = X_and_y(Preprocessor('../data/isear-test.csv'), removal)
        self.y_true = y_test

        le = LabelEncoder()
        # Convert labels into one hot encoding
        train_labels = le.fit_transform(y_train)
        self.y_train = np.asarray(to_categorical(train_labels))
        val_labels = le.fit_transform(y_val)
        self.y_val = np.asarray(to_categorical(val_labels))
        test_labels = le.fit_transform(y_test)
        self.y_test = np.asarray(to_categorical(test_labels))
        # Update class names in strings
        self.class_name = le.classes_

        # Vectorization: convert tokens into matrices
        # Initialize the Tokenizer class, remove punctuations we don't want to tokenize
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_tok, filters='"#$%&()*+-/:;<=>@[\]^_`{|}~ ')
        # Generate the word index dictionary from the training sentences
        tokenizer.fit_on_texts(X_train)
        word_index = tokenizer.word_index
        self.word_index = word_index

        # Generate and pad sentence sequences
        self.X_train = self.pad(X_train, tokenizer)
        self.X_val = self.pad(X_val, tokenizer)
        self.X_test = self.pad(X_test, tokenizer)

    def pad(self, X, tokenizer):
        sequences = tokenizer.texts_to_sequences(X)
        X = pad_sequences(sequences, maxlen=self.maxlen, truncating=self.trunc_type)
        return X

    def create_embedding_matrix(self, file_path):
        """
        Save the glove file of a specific dimension into a dictionary embedding_index,
        and compare each word in word_index with embedding_index,
        if a match occurs, copy the equivalent vector into embedding_matrix at the corresponding index.
        :param file_path: "glove.6B" txt file
        :return: (np.array), embedding_matrix
        """
        # embedding_index is used to save glove vectors
        embedding_index = {}
        with open(file_path, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = vector

        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dims))
        for word, index in self.word_index.items():
            if index > self.vocab_size - 1:
                break
            else:
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
        self.embedding_matrix = embedding_matrix

    def compile(self):
        """
        compile the model and show model summary
        """
        self.model = Sequential()
        # The `trainable` attribute of the layer is set to false so that the layer isn’t trained again
        self.model.add(Embedding(self.vocab_size, self.embedding_dims, input_length=self.maxlen, weights=[self.embedding_matrix], trainable=False))
        self.model.add(Conv1D(self.filters, self.filter_size, padding='valid', activation='relu'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(self.hidden_dims, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model.summary()

    def define_multi_channels(self, filter_sizes: list):
        """
        Each channel is for processing n-grams with different length.
        Then the output from all channels are concatenated into a single vector
        for further processing for the final output.
        :param (list of integers) filter_sizes: to specify multiple filter sizes (different n of n-grams)
        """
        self.n_filter_size = len(filter_sizes)
        pool_list = []
        input_list = []
        for i in range(len(filter_sizes)):
            # define the channel
            inputs = Input(shape=(self.maxlen,))
            embedding = Embedding(self.vocab_size, self.embedding_dims, input_length=self.maxlen, weights=[self.embedding_matrix], trainable=False)(inputs)
            conv = Conv1D(self.filters, filter_sizes[i], activation='relu')(embedding)
            pool = GlobalMaxPooling1D()(conv)
            # append input and output layers from each channel to input_list and pool_list for concatenation later
            input_list.append(inputs)
            pool_list.append(pool)

        merged = concatenate(pool_list)
        dense1 = Dense(self.hidden_dims, activation='relu')(merged)
        dropout = Dropout(0.5)(dense1)
        outputs = Dense(7, activation='sigmoid')(dropout)
        self.model = Model(inputs=input_list, outputs=outputs)

    def compile_multi_channels(self):
        """
        compile the multi-channels model and show model summary
        """
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model.summary()

    def fit(self):
        """
        fit data to the model
        """
        # callback is for early stop training when the validation loss is no longer decreasing
        callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        self.model.fit(self.X_train, self.y_train,
                       batch_size=self.batch_size, callbacks=[callback],
                       epochs=self.epochs,
                       validation_data=(self.X_val, self.y_val))

    def fit_multi_channels(self):
        """
        fit data to the multi-channels model
        """
        callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        if self.n_filter_size == 2:
            self.model.fit([self.X_train, self.X_train], array(self.y_train),
                           batch_size=self.batch_size, callbacks=[callback],
                           epochs=self.epochs,
                           validation_data=([self.X_val, self.X_val], array(self.y_val)))
        if self.n_filter_size == 3:
            self.model.fit([self.X_train, self.X_train, self.X_train], array(self.y_train),
                           batch_size=self.batch_size, callbacks=[callback],
                           epochs=self.epochs,
                           validation_data=([self.X_val, self.X_val, self.X_val], array(self.y_val)))

    def predict(self):
        """
        Predict on test set,
        save the predicted labels as strings in y_pred list for further evaluation
        """
        predicted = self.model.predict(self.X_test)
        for label in predicted:
            # get string labels from one hot encoded
            predicted_label = self.class_name[np.argmax(label)]
            self.y_pred.append(predicted_label)

        # evaluation: print accuracy for test set
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def predict_multi_channels(self):
        """
        predict with multi-channels model
        """
        if self.n_filter_size == 2:
            predicted = self.model.predict([self.X_test, self.X_test])
            for label in predicted:
                predicted_label = self.class_name[np.argmax(label)]
                self.y_pred.append(predicted_label)
            scores = self.model.evaluate([self.X_test, self.X_test], self.y_test, verbose=1)
            print("Accuracy: %.2f%%" % (scores[1] * 100))
        if self.n_filter_size == 3:
            predicted = self.model.predict([self.X_test, self.X_test, self.X_test])
            for label in predicted:
                predicted_label = self.class_name[np.argmax(label)]
                self.y_pred.append(predicted_label)
            scores = self.model.evaluate([self.X_test, self.X_test, self.X_test], self.y_test, verbose=1)
            print("Accuracy: %.2f%%" % (scores[1] * 100))


    def save_model(self):
        # save_model(self.model, self.output_path)
        self.model.save(self.output_path)
