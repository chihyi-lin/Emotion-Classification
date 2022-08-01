# Emotion Classification on Textual Data
> Project of CL Team Lab, 22SS, University of Stuttgart
## Contributors
* [Chih-Yi Lin](https://github.com/chihyi-lin)
* [Yat Han Lai](https://github.com/laiyathan)
## Project Introduction
* Different combinations of n-grams ranging from 1-2 to 3-4-grams are proposed to tackle the elusive nature of emotion expression in textual data. This approach is tested by comparing support vector machine (SVM) and convolutional neural network (CNN). The experiments demonstrate that our approach is effective when combining more n-grams in CNN, whereas the performance of SVM degrades when combining higher n-grams.
* Initial work: A multi-class perceptron using bag-of-words built from scratch is also provided.
## Folders of This Repository
* classifier: Perceptron baseline module, CNN module
* evaluaion: evaluaion - to calculate precision, recall, and F1 score for perceptron baseline
* preprocessing: preprocessing classes for perceptron baseline and CNN
* run_training: 'ModelName_main.py' is for training model
* trained_classifiers: Folder for saving trained models
## Reusing the Materials
1. Clone this repository
2. Get required pip libraries
```
pip install -r requirements.txt
```
### CNN
* Install Anaconda: https://docs.anaconda.com/anaconda/install/. Create an environment with conda and install all relevant libraries:
```
pip install scikit-learn
pip install tensorflow
pip install keras
pip install numpy
pip install nltk
```
* Download pre-trained GloVe embeddings from http://nlp.stanford.edu/data/glove.6B.zip
  - Create 'glove.6B' folder in the repository and save 'glove.6B.100d.txt' file there
3. Download the ISEAR dataset from [SWISS CENTER FOR AFFECTIVE SCIENCES](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/) 
    - Create 'data' folder in the repository and save the dataset there
5. Run the 'ModelName_main.py' script in 'run_training' folder
```
python ./run_training/ModelName_main.py

# or for windows
python .\run_training\ModelName_main.py
```

## Experiments
### Dataset
All experiments used the International Survey on Emotion Antecedents and Reactions (ISEAR) dataset, in which texts are catergorized into seven emotions: joy, fear, anger, sadness, disgust, shame and guilt.
### Data Preprocessing
All texts are converted into lowercase and tokenized, without stemming. Two data preprocessing settings are applied to experiment their effects on models' performance: 
* Setting 1: with tokenization only
* Setting 2: with punctuation and stopwords removal
### SVM
* Text representaion: TF-IDF
### CNN
* Word embeddings: Pre-trained GloVe embeddings with 100 dimensions
### Experiment Results
|Preprocess|Features | SVM | CNN |
|----------|---------|-----|-----|
|Setting 1 |1-gram   |0.54 |0.56 |
|          |3-grams  |0.42 |0.58 |
|          |1-2-grams|0.55 |0.58 |
|          |3-4-grams|0.41 |0.58 |
|          |1-3-grams|**0.56** |**0.61** |
|Setting 2 |1-gram   |0.55 |0.55 |
|          |3-grams  |0.15 |0.53 |
|          |1-2-grams|0.54 |0.56 |
|          |3-4-grams|0.15 |0.54 |
|          |1-3-grams|0.54 |0.58 |
