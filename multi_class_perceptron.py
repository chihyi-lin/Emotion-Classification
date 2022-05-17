from emotion_class import *


class MultiClassPerceptron:

    all_emotion_classes = list()

    def __init__(self, training_docs, validation_docs, iteration, learning_rate=0.01):
        self.docs = training_docs
        self.validation_docs = validation_docs
        self.iteration = iteration
        self.l_rate = learning_rate
        # self.feature_value is used to add or subtract from the original weight whenever a sample is misclassified
        self.feature_value = 1
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

    def find_max_prediction(self, doc) -> str:
        """Calculate weighted sum for each doc by different weight dictionaries from different emotion classes.
        Take the label from emotion_class which generates the highest score as the predicted label."""
        find_max = dict()   # find_max = {'joy':50, 'fear':-15...}
        for emotion_class in MultiClassPerceptron.all_emotion_classes:
            find_max[emotion_class.label] = emotion_class.weighted_sum(doc)
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
                        for emotion_class in MultiClassPerceptron.all_emotion_classes:
                            if emotion_class.label == predicted:
                                emotion_class.weights[token] = emotion_class.weights[token] - (self.l_rate * self.feature_value)
                            if emotion_class.label == gold:
                                emotion_class.weights[token] = emotion_class.weights[token] + (self.l_rate * self.feature_value)

    def append_prediction(self):
        """Append final prediction results to each document."""
        for doc in self.docs:
            predicted = self.find_max_prediction(doc)
            doc.append(predicted)
        return self.docs
