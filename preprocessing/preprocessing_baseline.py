"""Author: Chih-Yi Lin"""
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import re


class Preprocessor:

    def __init__(self, file_name):
        self.file_name = file_name
        # self.documents = [['1st label', '1st text'], ['2nd label', '2nd text']]
        self.documents = self.__remove_invalid_docs()
        self.text, self.label = self.read()
        self.tokenized_text = self.tokenize()

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
        :return: list of texts, list of labels
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
        First cleaning texts by removing unwanted leading and trailing whitespaces and quotation marks,
        converting texts to lowercase.
        Use nltk tokenizer to tokenize texts.
        :return:list(list(string)), tokenized texts
        """
        tokenized_text = []
        for text in self.text:
            text = text.strip(' ""''').lower()
            text = word_tokenize(text)
            tokenized_text.append(text)
        return tokenized_text
