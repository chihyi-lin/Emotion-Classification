# Emotion Classification on Textual Data
> Project of CL Team Lab, 22SS, University of Stuttgart
## Contributors
* Chih-Yi Lin
* Yat Han Lai
## Project Introduction
Different combinations of N-grams ranging from 1-2 to 3-4-grams are proposed to tackle the elusive nature of emotion expression in textual data. This approach is tested by comparing support vector machine (SVM) and convolutional neural network (CNN). The experiments demonstrate that our approach is effective when combining more N-grams in CNN, whereas the performance of SVM degrades when more N-grams are combined.

## Experiments
### Dataset
All experiments used the International Survey on Emotion Antecedents and Reactions (ISEAR) dataset, in which seven primary emotions, namely joy, fear, anger, sadness, disgust, shame and guilt were reported. Dataset is available from [SWISS CENTER FOR AFFECTIVE SCIENCES](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/).
### Data Preprocessing
All texts are converted into lowercase and tokenized, without stemming. Two data preprocessing settings are applied to experiment their effects on models' performance: 
* Setting 1: with tokenization only
* Setting 2: with punctuation and stopwords removal
### SVM
* Text representaion: TF-IDF
### CNN
* Word embeddings: Pre-trained GloVe embeddings with 100 dimensions, which is available from http://nlp.stanford.edu/data/glove.6B.zip.
### Experiment Results
|Preprocess|Features | SVM | CNN |
|----------|---------|-----|-----|
|Setting 1 |1-gram   |0.55 |0.56 |
|          |3-grams  |0.42 |0.58 |
|          |1-2-grams|**0.56** |0.58 |
|          |3-4-grams|0.39 |0.58 |
|          |1-3-grams|0.55 |**0.61** |
