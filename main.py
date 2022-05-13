from corpus import *
from multi_class_perceptron import *
from evaluation import *

# Train on training data
# Create documents as a nested list
c = Corpus('isear-train.csv')
docs = c.documents

# Tokenizing texts
d = Document(docs)
training_data = d.tokenize()

# Instantiate MultiClassPerceptron and update weights for 30 iterations
m = MultiClassPerceptron(training_data, 30)
m.update_weights()

# Make prediction on training data
m.prediction()

# Evaluation
eval_1 = Evaluation(m.docs)
f_score_joy = eval_1.f_score('joy')
f_score_fear = eval_1.f_score('fear')
f_score_guilt = eval_1.f_score('guilt')
f_score_anger = eval_1.f_score('anger')
f_score_shame = eval_1.f_score('shame')
f_score_disgust = eval_1.f_score('disgust')
f_score_sadness = eval_1.f_score('sadness')
print("f score for joy: {} \n"
      "f score for fear: {} \n"
      "f score for guilt: {} \n"
      "f score for anger: {} \n"
      "f score for shame: {} \n"
      "f score for disgust: {} \n"
      "f score for sadness: {} \n"
      .format(f_score_joy, f_score_fear, f_score_guilt, f_score_anger, f_score_shame, f_score_disgust, f_score_sadness))

# TODO 1. Make prediction on validation data
#  2. Reorganized main.py into nicer classes/functions
