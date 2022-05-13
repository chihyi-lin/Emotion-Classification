from perceptron import *


class MultiClassPerceptron:

    all_perceptrons = list()

    def __init__(self, tokenized_docs, iteration):

        # global guilt_had, anger_had # Comment out later
        self.docs = tokenized_docs
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

    def __find_max_prediction(self, doc) -> str:
        """Calculate weighted sum for each doc by different weight dictionaries from different perceptrons.
        Take the result prediction from the perceptron which has the highest score."""
        find_max = dict()   # find_max = {'joy':50, 'fear':-15...}
        for perceptron in MultiClassPerceptron.all_perceptrons:
            find_max[perceptron.label] = perceptron.weighted_sum(doc)
        made_prediction_perceptron = max(find_max, key=find_max.get)
        return made_prediction_perceptron

    def update_weights(self):
        global guilt_had_, anger_had_ # Comment out later
        for i in range(self.iteration):
            for doc in self.docs:
                gold = doc[0]
                predicted = self.__find_max_prediction(doc)
                tokens = doc[1]
                if gold != predicted:
                    for token in tokens:
                        for perceptron in MultiClassPerceptron.all_perceptrons:
                            if perceptron.label == predicted:
                                perceptron.weights[token] = perceptron.weights[token] - self.feature_value
                            if perceptron.label == gold:
                                perceptron.weights[token] = perceptron.weights[token] + self.feature_value
            # for perceptron in MultiClassPerceptron.all_perceptrons:  # Comment out later
            #     if perceptron.label == 'guilt':
            #         guilt_had_ = perceptron.weights['had']
            #         guilt_father_ = perceptron.weights['father']
            #         print('updated score for "had" in guilt:', guilt_had_)
            #         print('updated score for "father" in guilt:', guilt_father_)
            #     if perceptron.label == 'anger':
            #         anger_had_ = perceptron.weights['had']
            #         print('updated score for "had" in anger:', anger_had_)

    def prediction(self):
        """Append final prediction results to each document."""
        for doc in self.docs:
            predicted = self.__find_max_prediction(doc)
            doc.append(predicted)
        return self.docs
