from corpus import *
from multi_class_perceptron import *
from evaluation import *


class Workflow:

    def __init__(self, iteration):
        self.training_data = self.preprocessing('isear-train.csv')
        self.validation_data = self.preprocessing('isear-val.csv')
        self.iteration = iteration

    # Create documents as a nested list, tokenizing texts
    def preprocessing(self, file: str):
        c = Corpus(file)
        docs = c.documents
        d = Document(docs)
        tokenized_data = d.tokenize()
        return tokenized_data

    # Instantiating MultiClassPerceptron, train and evaluate on each iteration and choose the settings that yield the best performance.
    def training(self):
        m = MultiClassPerceptron(self.training_data, self.validation_data, self.iteration)
        for doc in self.training_data:
            m.add_BIAS(doc)
        m.run_iteration()
        m.choose_optimal_epoch()
        return m

# Best settings: learning rate=0.001, epochs=34
w = Workflow(34)
trained_perceptron = w.training()
trained_perceptron.perclass_f_score()
