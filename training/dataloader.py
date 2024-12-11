import pandas as pd
from training.preprocessing import TextPreprocess

proc = TextPreprocess()

class Dataset:
    def __init__(self):
        pass

    def load_dataset(self, path):
        dataset = pd.read_csv(path, encoding="utf-8")
        return dataset

    def dnn_approach_dataset(self):
        train_set = self.load_dataset('dataset/train.csv')
        test_set = self.load_dataset('dataset/test.csv')
        evaluate_set = self.load_dataset("dataset/dev.csv")

        train_set['text_processed'] = train_set['text'].apply(proc.remove_stopwords).apply(proc.process_text)
        evaluate_set['text_processed'] = evaluate_set['text'].apply(proc.remove_stopwords).apply(proc.process_text)
        test_set['text_processed'] = test_set['text'].apply(proc.remove_stopwords).apply(proc.process_text)
        
        return train_set, test_set, evaluate_set
    
    def traditional_approach_dataset(self):
        train_set = self.load_dataset('dataset/train.csv')
        test_set = self.load_dataset('dataset/test.csv')
        evaluate_set = self.load_dataset("dataset/dev.csv")

        train_set['text_processed'] = train_set['text']
        evaluate_set['text_processed'] = evaluate_set['text']
        test_set['text_processed'] = test_set['text']
        
        return train_set, test_set, evaluate_set
