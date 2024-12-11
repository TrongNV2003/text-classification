from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class SVM_Trainer:
    def __init__(self):
        self.seed = 42
        self.model_svm = SVC(random_state = self.seed)
        self.vectorizer = TfidfVectorizer(max_features=10000)

    def vectorize_train(self, text_set):
        text_vector = self.vectorizer.fit_transform(text_set)
        return text_vector

    def train(self, vector, label):
        trainer = self.model_svm.fit(vector, label)
        return trainer
    
    def vectorize_test(self, text_set):
        text_vector = self.vectorizer.transform(text_set)
        return text_vector
    
    def predict(self, vector):
        predicter = self.model_svm.predict(vector)
        return predicter

class NB_Trainer:
    def __init__(self):
        self.model_nb = ComplementNB()
        self.vectorizer = TfidfVectorizer(max_features=10000)

    def vectorize_train(self, text_set):
        text_vector = self.vectorizer.fit_transform(text_set)
        return text_vector

    def train(self, vector, label):
        trainer = self.model_nb.fit(vector, label)
        return trainer
    
    def vectorize_test(self, text_set):
        text_vector = self.vectorizer.transform(text_set)
        return text_vector
    
    def predict(self, vector):
        prediction = self.model_nb.predict(vector)
        return prediction

class LogisticRegression_Trainer:
    def __init__(self):
        self.seed = 42
        self.model_lr = LogisticRegression(random_state=self.seed)
        self.vectorizer = TfidfVectorizer(max_features=10000)

    def vectorize_train(self, text_set):
        text_vector = self.vectorizer.fit_transform(text_set)
        return text_vector

    def train(self, vector, label):
        trainer = self.model_lr.fit(vector, label)
        return trainer
    
    def vectorize_test(self, text_set):
        text_vector = self.vectorizer.transform(text_set)
        return text_vector
    
    def predict(self, vector):
        prediction = self.model_lr.predict(vector)
        return prediction
    
