# Classification
This repo research and apply all methods in text classification.
- Traditional approaches: SVM, Naive Bayes, Logistic Regression...
- Deep neural network approaches: CNN, RNN, LSTM...
- Language model appoaches: BERT, RoBERTa...

This research give a larger insight in ways to deal with classification problems.

## Report result
Result: [Classification Report](https://docs.google.com/document/d/1dFwsDAB1Hl3m8hzpSt5xZRevKchNTTyJZLZumKbrNGE/edit?usp=sharing)

## Installation
```sh
pip install -r requirements.txt
```

## Download pretrained model word2vec for DNN
Model trained on Vietnamese Wiki: [Word2vec](https://github.com/sonvx/word2vecVN)

create "models" folder and save Word2vec with following path:
```sh
text-classification/
    └── models/
        └── wiki.vi.model.bin
```
