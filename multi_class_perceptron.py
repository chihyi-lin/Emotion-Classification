from perceptron import *


class MultiClassPerceptron:

    all_perceptrons = list()

    def __init__(self, training_docs, validation_docs, iteration):
        self.docs = training_docs
        self.validation_docs = validation_docs
        self.iteration = iteration
        # self.feature_value is used to add or subtract to/from the original weight whenever a sample is misclassified
        self.feature_value = 1
        self.p_joy = Perceptron('joy', self.docs)
        self.p_fear = Perceptron('fear', self.docs)
        self.p_guilt = Perceptron('guilt', self.docs)
        self.p_anger = Perceptron('anger', self.docs)
        self.p_shame = Perceptron('shame', self.docs)
        self.p_disgust = Perceptron('disgust', self.docs)
        self.p_sadness = Perceptron('sadness', self.docs)

        MultiClassPerceptron.all_perceptrons.append(self.p_joy)
        MultiClassPerceptron.all_perceptrons.append(self.p_fear)
        MultiClassPerceptron.all_perceptrons.append(self.p_guilt)
        MultiClassPerceptron.all_perceptrons.append(self.p_anger)
        MultiClassPerceptron.all_perceptrons.append(self.p_shame)
        MultiClassPerceptron.all_perceptrons.append(self.p_disgust)
        MultiClassPerceptron.all_perceptrons.append(self.p_sadness)

    def find_max_prediction(self, doc) -> str:
        """Calculate weighted sum for each doc by different weight dictionaries from different perceptrons.
        Take the label from perceptron which generates the highest score as predicted label."""
        find_max = dict()   # find_max = {'joy':50, 'fear':-15...}
        for perceptron in MultiClassPerceptron.all_perceptrons:
            find_max[perceptron.label] = perceptron.weighted_sum(doc)
        prediction = max(find_max, key=find_max.get)
        return prediction

    def update_weights(self):
        for i in range(self.iteration):
            for doc in self.docs:
                gold = doc[0]
                predicted = self.find_max_prediction(doc)
                tokens = doc[1]
                if gold != predicted:
                    for token in tokens:
                        for perceptron in MultiClassPerceptron.all_perceptrons:
                            if perceptron.label == predicted:
                                perceptron.weights[token] = perceptron.weights[token] - self.feature_value
                            if perceptron.label == gold:
                                perceptron.weights[token] = perceptron.weights[token] + self.feature_value

    def append_prediction(self):
        """Append final prediction results to each document."""
        for doc in self.docs:
            predicted = self.find_max_prediction(doc)
            doc.append(predicted)
        return self.docs
