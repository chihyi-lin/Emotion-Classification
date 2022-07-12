from nltk.tokenize import word_tokenize
import re
import numpy as np


class Preprocessor:

    def __init__(self, file_name):
        self.file_name = file_name
        self.documents = self.__remove_invalid_docs()
        self.text, self.label = self.read()
        self.tokenized_text = self.tokenize()
        self.X_array = self.clean_text()

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

    def tokenize(self):
        """
        Cleaning texts by removing unwanted leading and trailing whitespaces and quotation marks,
        converting texts to lowercase.
        Then use nltk tokenizer to tokenize texts with punctuations retained.
        :return:list(list(string:tokens)), tokenized texts
        """
        tokenized_text = []
        for text in self.text:
            text = text.strip(' ""''').lower()
            text = word_tokenize(text)
            tokenized_text.append(text)
        return tokenized_text

    def clean_text(self):
        """
        Join all tokens back to a sentence and convert to np.array
        :return: cleaned texts (numpy.ndarray)
        """
        clean_text = []
        for tokens in self.tokenized_text:
            clean_sentence = ' '.join(tokens)
            clean_text.append(clean_sentence)
        clean_text = np.array(clean_text)
        return clean_text


# p = Preprocessor('../data/isear-train.csv')
#
# print(p.X_array)
# print(type(p.X_array))
# original = p.text
# clean = p.clean_text
# label = p.label
#
# # print(original[:5])
# print(p[:5])
# print(label[:5])