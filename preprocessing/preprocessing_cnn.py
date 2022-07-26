"""Author: Chih-Yi Lin"""
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
import string


class Preprocessor:

    def __init__(self, file_name):
        self.file_name = file_name
        self.documents = self.__remove_invalid_docs()
        self.text, self.label = self.read()
        self.tokenized_text = None
        self.X_array = None

    def __read_file(self) -> list:
        documents = list()
        with open(self.file_name, 'r') as f:
            for doc in f:
                doc = doc.replace("\n", "").split(",", 1)
                documents.append(doc)
        return documents

    def __remove_invalid_docs(self) -> list:
        """
        Remove docs without labels and docs with invalid texts.
        :return: list(list(strings)): list of documents with both label and text.
        """
        full_documents = self.__read_file()
        labeled_documents = list()
        labels = {'joy', 'fear', 'guilt', 'anger', 'shame', 'disgust', 'sadness'}
        for doc in full_documents:
            label = doc[0]
            if label in labels:
                labeled_documents.append(doc)

        regex = re.compile(r'\[.*\]|None.|NO RESPONSE.')
        cleaned_documents = list()
        for doc in labeled_documents:
            text = doc[1]
            if not regex.match(text):
                cleaned_documents.append(doc)
        return cleaned_documents

    def read(self):
        """
        Read valid documents and split them into text list and label list.
        :return: list(string): list of texts, list(string): list of labels
        """
        documents = self.__remove_invalid_docs()
        text = []
        label = []
        for i in documents:
            label.append(i[0])
            text.append(i[1])
        return text, label

    def tokenize(self, removal=False):
        """
        :param removal: 'False' means stopwords and punctuation retained in texts
        and 'True' means remove them from texts
        Clean texts, converting texts into lowercase,
        then use nltk tokenizer to tokenize texts
        :return:list(list(string:tokens)), tokenized texts
        """
        tokenized_text = []
        if not removal:
            for text in self.text:
                text = text.strip(' ""''').lower()
                text = word_tokenize(text)
                tokenized_text.append(text)
        else:
            stop_words = set(stopwords.words('english'))
            for text in self.text:
                # replace all punctuation with None
                text = text.translate(str.maketrans('', '', string.punctuation))
                # remove all stopwords
                filtered_text = []
                for word in text.split():
                    if word.lower() not in stop_words:
                        filtered_text.append(word.lower())
                tokenized_text.append(filtered_text)

        self.tokenized_text = tokenized_text

    def join_text(self):
        """
        Join all tokens back to a sentence and convert to np.array
        :return: numpy.ndarray, cleaned texts
        """
        clean_text = []
        for tokens in self.tokenized_text:
            clean_sentence = ' '.join(tokens)
            clean_text.append(clean_sentence)
        clean_text = np.array(clean_text)
        self.X_array = clean_text

    # calculate the maximum document length
    def max_length(self):
        return max([len(text.split()) for text in self.text])


def X_and_y(data:Preprocessor, removal):
    """
    Prepare correct data types of X (features) and y (labels) to feed into the model.
    :param data: data which needs to be preprocess by the Preprocessor
    :param removal: whether stopwords and punctuation are removed or not
    :return: numpy.ndarray: X, list(string): y
    """
    data.tokenize(removal)
    data.join_text()
    return data.X_array, data.label
