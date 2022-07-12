# Preprocessing data to the model
from preprocessing.preprocessing_cnn import *
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
# Keras layers
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D
# For building multi channel
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from numpy import array


class CNN:
    """
    class CNN has two architectures:
    1-channel: taking 1 filter size (n-grams with the same length)
    multi-channels: taking up to 3 filter sizes (n-grams with different lengths)
    """
    def __init__(self, vocab_size, embedding_dims, filters, filter_size, hidden_dims, batch_size, epochs):

        self.vocab_size = vocab_size
        self.oov_tok = "<OOV>"
        # sentence longer than maxlen will be truncated from the end
        self.trunc_type = "post"
        # Max input length (number of words) for each text
        self.maxlen = 300

        self.embedding_dims = embedding_dims
        self.filters = filters
        self.filter_size = filter_size
        # the number of neurons in the hidden layer
        self.hidden_dims = hidden_dims

        self.batch_size = batch_size  # 32
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
        # use y_pred (predicted labels in test set) and y_true (actual labels in test set) for evaluation
        self.y_pred = []
        self.y_true = None

    def preprocess(self):
        """
        Pre-process data to fit into model,
        converting tokens into matrices,
        generating word_index dictionary,
        generating and padding texts into sequences of integers
        :return: None
        """
        train_data = Preprocessor('../data/isear-train.csv')
        X_train = train_data.X_array
        y_train = train_data.label
        val_data = Preprocessor('../data/isear-val.csv')
        X_val = val_data.X_array
        y_val = val_data.label
        test_data = Preprocessor('../data/isear-test.csv')
        X_test = test_data.X_array
        y_test = test_data.label
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
        :return: np.array(embedding_matrix)
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

    def define_multi_channels(self, filter_size1, filter_size2, filter_size3):
        """
        Each channel is for processing different n-gram(s).
        Then the output from the three channels are concatenated into a single vector
        for further processing for the final output.
        :param filter_size_n: is an integer refer to filter size (n-gram(s))
        """
        # define the 1st channel: filter size=1
        inputs1 = Input(shape=(self.maxlen,))
        embedding1 = Embedding(self.vocab_size, self.embedding_dims)(inputs1)
        conv1 = Conv1D(self.filters, filter_size1, activation='relu')(embedding1)
        pool1 = GlobalMaxPooling1D()(conv1)
        # define the 2nd channel: filter size=2
        inputs2 = Input(shape=(self.maxlen,))
        embedding2 = Embedding(self.vocab_size, self.embedding_dims)(inputs2)
        conv2 = Conv1D(self.filters, filter_size2, activation='relu')(embedding2)
        pool2 = GlobalMaxPooling1D()(conv2)
        # define the 3rd channel: filter size=3
        inputs3 = Input(shape=(self.maxlen,))
        embedding3 = Embedding(self.vocab_size, self.embedding_dims)(inputs3)
        conv3 = Conv1D(self.filters, filter_size3, activation='relu')(embedding3)
        pool3 = GlobalMaxPooling1D()(conv3)

        merged = concatenate([pool1, pool2, pool3])
        dense1 = Dense(self.hidden_dims, activation='relu')(merged)
        dropout = Dropout(0.5)(dense1)
        outputs = Dense(7, activation='sigmoid')(dropout)
        self.model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    def compile_multi_channels(self):
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model.summary()

    def fit(self):
        """
        fit data to the model
        """
        self.model.fit(self.X_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(self.X_val, self.y_val))

    def fit_multi_channels(self):
        self.model.fit([self.X_train, self.X_train, self.X_train], array(self.y_train),
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=([self.X_val, self.X_val, self.X_val], array(self.y_val)))

    def predict(self):
        """
        Predict on test set,
        return the predicted labels as strings for further evaluation
        :return: predicted labels (list of strings)
        """
        predicted = self.model.predict(self.X_test)
        for label in predicted:
            # get string labels from one hot encoded
            predicted_label = self.class_name[np.argmax(label)]
            self.y_pred.append(predicted_label)

    def save_model(self):
        save_model(self.model, self.output_path)

    # def load_model(self):
    #     model = load_model(self.output_path, compile=True)
    #     return model

# TODO: implement predict and evaluation functions for multi-channel
