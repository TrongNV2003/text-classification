import time
import json
from sklearn.metrics import f1_score
from training.dataloader import Dataset
from training.trainer import SVM_Trainer

dataset = Dataset()
svm = SVM_Trainer()

def f1(vector):
    prediction = svm.predict(vector)
    score = f1_score(test_label, prediction, average='macro')
    print(f"F1-score: {score}")

def result_recorder(path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)
        print(f"Saved in {path}")

if __name__ == "__main__":
    train_set, test_set, evaluate_set = dataset.traditional_approach_dataset()

    train_text = train_set['text_processed']
    train_label = train_set['label']

    test_text = test_set['text_processed']
    test_label = test_set['label']

    train_text_vect = svm.vectorize_train(train_text)
    test_text_vect = svm.vectorize_test(test_text)
    
    start_time = time.time()
    classifier = svm.train(train_text_vect, train_label)
    end_time = time.time()
    process_time = round(end_time - start_time, 6)
    print(f"Training time: {process_time}")

    result = []
    for test in test_text_vect:
        start_time = time.time()
        prediction = classifier.predict(test)
        end_time = time.time()    
        process_time = end_time - start_time
        
        result.append({
            "process_time": process_time
        })

    f1(test_text_vect)
    result_recorder("result/output_svm.json")
