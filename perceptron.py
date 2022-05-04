from corpus import *


class Perceptron:

    doc = Document()
    doc = doc.tokenized_text

    def __init__(self, label):
        self.weights = self.__initial_weights()
        # self.weights = {'token': weight}
        self.label = label
        self.feature_value = 1
        # self.feature_value is set for updating self.weights,
        # and feature vector is boolean: [1, 0, 1, 1, 0...] that's why it's set to 1.
        # ? Is it correct to set feature value =1 ?

    def __initial_weights(self):
        weights = dict()
        for doc in self.doc:
            tokens = doc[1]
            for token in tokens:
                weights[token] = 0.1
        return weights

    def __weighted_sum(self, doc):
        """Sum of x*w"""
        sum = 0
        tokens = doc[1]
        for token in tokens:
            weighted_x = self.weights[token]
            sum += weighted_x
        return sum

    def update_weights(self):
        epochs = 100
        for i in range(epochs):
            print("update weights epoch " + str(i) + " out of " + str(epochs))
            for doc in self.doc:
                gold = doc[0]
                predict = self.__weighted_sum(doc)
                tokens = doc[1]
                if gold == self.label and predict > 0:
                    pass
                if gold != self.label and predict < 0:
                    pass
                if gold == self.label and predict < 0:
                    for token in tokens:
                        self.weights[token] = self.weights[token] + self.feature_value
                if gold != self.label and predict > 0:
                    for token in tokens:
                        self.weights[token] = self.weights[token] - self.feature_value

            # TODO 2.after each epoch print the current fscore for label "joy",
            # so we can know how is the current training process
            # we can e.g. stop the training (break out of the for loop),
            # if the fscore for "joy" has reached e.g. 80% accuracy

    def predict_all(self):
        """Append final prediction results to each document."""
        for doc in self.doc:
            doc.append(self.__weighted_sum(doc))
        return self.doc

# TODO 1.build multi-class perceptron



p = Perceptron('joy')
p.update_weights()
p = p.predict_all()
# score_for_happy = p.weights['happy']
# score_for_wonderful = p.weights['wonderful']
# score_for_school = p.weights['school']
# print(score_for_happy, score_for_wonderful, score_for_school)
print(p)
