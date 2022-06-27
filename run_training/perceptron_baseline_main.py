from preprocessing.preprocessing_baseline import *
from classifiers.perceptron_baseline import *


class TrainingProcess:

    def __init__(self, iteration):
        self.training_data = self.preprocessing('data/isear-train.csv')
        self.validation_data = self.preprocessing('data/isear-val.csv')
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


def workflow():
    """Train and save classifier for further evaluation and error analysis"""
    t = TrainingProcess(40)
    trained_perceptron = t.training()
    trained_perceptron.save_classifier("perceptron_baseline")


workflow()
