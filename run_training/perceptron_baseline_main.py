from preprocessing.preprocessing_baseline import *
from classifiers.perceptron_baseline import *



class TrainingProcess:

    def __init__(self, iteration):
        self.training_data = self.preprocessing('../data/isear-train.csv')
        self.validation_data = self.preprocessing('../data/isear-val.csv')
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
        # m.choose_optimal_epoch()
        return m


def workflow():
    """Train and save perceptron"""
    # t = TrainingProcess(34)
    # trained_perceptron = t.training()
    # trained_perceptron.save_classifier("perceptron_baseline")

    """Load saved perceptron"""
    baseline = MultiClassPerceptron.load_classifier("perceptron_baseline")

    """Evaluation: Print out per-class f-scores for the validation set"""
    # baseline.perclass_f_score()

    """Error analysis: look up weight dictionaries"""
    # stopwords_and_adverbs = ['BIAS', 'when', 'i', 'that', 'was', 'to', 'the', 'of', 'and', 'or', 'very', 'only', 'not', 'completely', 'possible', 'even']
    # weight_for_joy = baseline.joy.weights
    # weight_for_anger = baseline.anger.weights
    # print("weight for joy:")
    # for i in stopwords_and_adverbs:
    #     if i in weight_for_joy.keys():
    #         print(i, weight_for_joy[i])
    # print("weight for anger:")
    # for i in stopwords_and_adverbs:
    #     if i in weight_for_anger.keys():
    #         print(i, weight_for_anger[i])
    """Error analysis: look up the weights for falsely classified documents from validation set"""
    # baseline.all_emotion_classes.append(baseline.joy)
    # baseline.all_emotion_classes.append(baseline.fear)
    # baseline.all_emotion_classes.append(baseline.guilt)
    # baseline.all_emotion_classes.append(baseline.anger)
    # baseline.all_emotion_classes.append(baseline.shame)
    # baseline.all_emotion_classes.append(baseline.disgust)
    # baseline.all_emotion_classes.append(baseline.sadness)
    # val_docs = baseline.validation_docs
    # for doc in val_docs:
    #     gold = doc[0]
    #     predicted = doc[3]
    #     find_max = dict()
    #     if gold != predicted:
    #         for emotion_class in baseline.all_emotion_classes:
    #             find_max[emotion_class.label] = emotion_class.weighted_sum(doc)
    #
    # return find_max


workflow()

