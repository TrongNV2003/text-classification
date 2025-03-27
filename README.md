# Classification
This repo research and apply all methods in text classification.
- Traditional approaches: SVM, Naive Bayes, Logistic Regression...
- Deep neural network approaches: CNN, RNN, LSTM...
- Language model appoaches: BERT, RoBERTa...

This research give a larger insight in ways to deal with classification problems.

## Installation
```sh
pip install -r requirements.txt
```

## Running
You can train and evaluate with each model by running models (SVM, NaiveBayes...), for instance:
```sh
python SVM.py
```

## Download pretrained model word2vec for DNN
Model trained on Vietnamese Wiki: [Word2vec](https://github.com/sonvx/word2vecVN)

create "models" folder and save Word2vec with following path:
```sh
text-classification/
    └── models/
        └── wiki.vi.model.bin
```

## Results
| Models                 | Accuracy    | Precision   | Recall      | F1 Score     | P99 Latency  |
|----------------------- |:-----------:|:-----------:|:-----------:|:------------:|:------------:|
| SVM                    | 76.93       | 80.89       | 72.16       | 76.28        | 0.59         |
| Naive Bayes            | 76.21       | 78.55       | 73.86       | 76.13        | **0.14**     |
| Logistic Regression    | 76.64       | 79.27       | 73.86       | 76.47        | 0.12         |
| CNN                    | 71.09       | 75.84       | 64.20       | 69.54        | 1.90         |
| RNN                    | 73.66       | 77.63       | 68.41       | 72.73        | 3.85         |
| LSTM                   | 73.84       | 79.84       | 75.15       | 75.99        | 76.28        |
| PhoBERT base           | **79.71**   | **82.77**   | **76.42**   | **79.47**    | 8.39         |
