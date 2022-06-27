from classifiers.perceptron_baseline_emotion_classes import *
from evaluation.evaluation import *
import pickle

class MultiClassPerceptron:

    all_emotion_classes = list()

    def __init__(self, training_docs, validation_docs, iteration, learning_rate=0.001):
        self.docs = training_docs
        self.validation_docs = validation_docs
        self.iteration = iteration
        self.l_rate = learning_rate
        # self.feature_value is used to add or subtract from the original weight whenever a sample is misclassified
        self.feature_value = 1
        self.save_epochs = dict()
        # Directory in which to save trained models
        self.OUTPUT_PATH = "trained_classifiers/"

        self.joy = EmotionClass('joy', self.docs)
        self.fear = EmotionClass('fear', self.docs)
        self.guilt = EmotionClass('guilt', self.docs)
        self.anger = EmotionClass('anger', self.docs)
        self.shame = EmotionClass('shame', self.docs)
        self.disgust = EmotionClass('disgust', self.docs)
        self.sadness = EmotionClass('sadness', self.docs)
        MultiClassPerceptron.all_emotion_classes.append(self.joy)
        MultiClassPerceptron.all_emotion_classes.append(self.fear)
        MultiClassPerceptron.all_emotion_classes.append(self.guilt)
        MultiClassPerceptron.all_emotion_classes.append(self.anger)
        MultiClassPerceptron.all_emotion_classes.append(self.shame)
        MultiClassPerceptron.all_emotion_classes.append(self.disgust)
        MultiClassPerceptron.all_emotion_classes.append(self.sadness)

    def add_BIAS(self, doc):
        tokens = doc[1]
        tokens.append("BIAS")
        return doc

    def run_iteration(self):
        """
        1. Iterate through training set, when a doc is misclassified, update weight dictionaries.
        2. The model is trained on the training set, and the evaluation is simultaneously
        performed on the validation set after every epoch.
        """
        for i in range(self.iteration):
            for doc in self.docs:
                gold = doc[0]
                predicted = self.find_max_prediction(doc)
                tokens = doc[1]
                # Update weights
                if gold != predicted:
                    for token in tokens:
                        for emotion_class in MultiClassPerceptron.all_emotion_classes:
                            if emotion_class.label == predicted:
                                emotion_class.weights[token] = emotion_class.weights[token] - (self.l_rate * self.feature_value)
                            if emotion_class.label == gold:
                                emotion_class.weights[token] = emotion_class.weights[token] + (self.l_rate * self.feature_value)
            # Make prediction on the train set
            self.__predict_on_train_data(i)
            # Make prediction on the validation set
            self.__predict_on_val_data(i)
            # Save and print each epoch status
            self.__save_current_epoch_status(i)

    def find_max_prediction(self, doc) -> str:
        """Calculate weighted sum for each doc by different weight dictionaries from different emotion classes.
        Take the label from emotion_class which generates the highest score as the predicted label."""
        find_max = dict()   # find_max = {'joy':50, 'fear':-15...}
        for emotion_class in MultiClassPerceptron.all_emotion_classes:
            find_max[emotion_class.label] = emotion_class.weighted_sum(doc)
        prediction = max(find_max, key=find_max.get)
        return prediction

    def __predict_on_train_data(self, i):
        if i == 0:
            for doc in self.docs:
                predicted = self.find_max_prediction(doc)
                doc.append(predicted)
        if i != 0:
            for doc in self.docs:
                predicted = self.find_max_prediction(doc)
                doc[3] = predicted
        return self.docs

    def __predict_on_val_data(self, i):
        if i == 0:
            for doc in self.validation_docs:
                self.add_BIAS(doc)
                predicted = self.find_max_prediction(doc)
                doc.append(predicted)
        if i != 0:
            for doc in self.validation_docs:
                predicted = self.find_max_prediction(doc)
                doc[3] = predicted
        return self.validation_docs

    def __save_current_epoch_status(self, i):
        e1 = Evaluation(self.docs)
        macro_for_train = e1.macro_average()
        micro_for_train = e1.micro_average()
        e2 = Evaluation(self.validation_docs)
        macro_for_val = e2.macro_average()
        micro_for_val = e2.micro_average()
        # self.save_epochs = {1:{train:0.36, val:0.41}, 2:{train:0.36, val:0.41}...}
        self.save_epochs[i + 1] = self.save_epochs.get(i + 1, {})
        self.save_epochs[i + 1]['train'] = self.save_epochs[i + 1].get('train', macro_for_train)
        self.save_epochs[i + 1]['val'] = self.save_epochs[i + 1].get('val', macro_for_val)

        print("training epoch {} of {}".format(i+1, self.iteration))
        print("\tmacro for train set: {}".format(macro_for_train))
        print("\tmicro for train set: {}".format(micro_for_train))
        print("\tmacro for validation set: {}".format(macro_for_val))
        print("\tmicro for validation set: {}".format(micro_for_val))

    def choose_optimal_epoch(self):
        """Find the key-value pair which has the highest macro for validation set in self.save_epochs"""
        max_val = 0
        optimal = dict()
        for epoch, scores in self.save_epochs.items():
            if scores['val'] > max_val:
                max_val = scores['val']
                optimal = epoch, scores
        print("optimal epoch:", optimal)

    def perclass_f_score(self):
        evaluator = Evaluation(self.validation_docs)
        f_score_for_joy = evaluator.perclass_f_score('joy')
        f_score_for_fear = evaluator.perclass_f_score('fear')
        f_score_for_guilt = evaluator.perclass_f_score('guilt')
        f_score_for_anger = evaluator.perclass_f_score('anger')
        f_score_for_shame = evaluator.perclass_f_score('shame')
        f_score_for_disgust = evaluator.perclass_f_score('disgust')
        f_score_for_sadness = evaluator.perclass_f_score('sadness')
        print("per-class f-scores on validation set: ")
        print("joy: {}".format(f_score_for_joy))
        print("fear: {}".format(f_score_for_fear))
        print("guilt: {}".format(f_score_for_guilt))
        print("anger: {}".format(f_score_for_anger))
        print("shame: {}".format(f_score_for_shame))
        print("disgust: {}".format(f_score_for_disgust))
        print("sadness: {}".format(f_score_for_sadness))
        return evaluator

    def save_classifier(self, classifier_name):
        """
        Saves classifiers as a .pickle file to the trained_classifiers directory.
        """
        with open(self.OUTPUT_PATH + classifier_name + ".pik", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_classifier(classifier_name):
        OUTPUT_PATH = "trained_classifiers/"
        with open(OUTPUT_PATH + classifier_name + ".pik", 'rb') as f:
            return pickle.load(f)
