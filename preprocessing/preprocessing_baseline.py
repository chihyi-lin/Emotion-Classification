"""Author: Chih-Yi Lin"""
from nltk.tokenize import word_tokenize
import re


class Corpus:

    def __init__(self, file_name):
        self.file_name = file_name
        # self.documents = [['1st label', '1st text', 'text1_id'], ['2nd label', '2nd text', 'text2_id']]
        self.documents = self.assign_text_id()

    def __read_file(self) -> list:
        documents = list()
        with open(self.file_name, 'r') as f:
            for doc in f:
                doc = doc.replace("\n", "").split(",", 1)
                documents.append(doc)
        return documents

    def __remove_invalid_docs(self) -> list:
        """
        remove docs without labels and docs with invalid texts.
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

    def assign_text_id(self) -> list:
        documents = self.__remove_invalid_docs()
        text_id = 1
        for doc in documents:
            doc.append(text_id)
            text_id += 1
        return documents


class Document:

    def __init__(self, docs):
        self.docs = docs

    def tokenize(self) -> list:
        #  TODO: Use NLTK to tokenize texts.
        """
        Convert texts into lowercase, remove punctuations, tokenizing texts.
        :return: docs = [['1st label', ['1st tokenized text'], 1st text_id],...]
        """
        for doc in self.docs:
            text = doc[1]
            text = text.strip(' ""''').lower()
            text = word_tokenize(text)
            doc[1] = text
        return self.docs
