import time
import argparse
from sklearn.svm import SVC
from training.evaluate import Tester
from training.dataloader import Dataset
from training.trainer import Vectorizer, Trainer_trad

vec = Vectorizer()
model = SVC(random_state = 42)

parser = argparse.ArgumentParser()

parser.add_argument("--train_file", type=str, default="dataset/train.json")
parser.add_argument("--test_file", type=str, default="dataset/test.json")
args = parser.parse_args()

if __name__ == "__main__":
    train_set = Dataset(args.train_file)
    test_set = Dataset(args.test_file)
    
    train_text = [train_set[i][0] for i in range(len(train_set))]
    train_label = [train_set[i][1] for i in range(len(train_set))]

    test_text = [test_set[i][0] for i in range(len(test_set))]
    test_label = [test_set[i][1] for i in range(len(test_set))]

    train_text_vect = vec.train_vectorizer(train_text)
    test_text_vect = vec.test_vectorizer(test_text)
    
    start_time = time.time()

    trainer = Trainer_trad(model, train_text_vect, train_label)
    trainer.train()

    end_time = time.time()
    process_time = round(end_time - start_time, 6)
    print(f"Training time: {process_time}\n")

    result = []
    for test in test_text_vect:
        start_time = time.time()

        trainer = Trainer_trad(model, train_text_vect, train_label)
        trainer.train()

        end_time = time.time()    
        process_time = end_time - start_time
        
        result.append({
            "process_time": process_time
        })

    prediction = model.predict(test_text_vect)
    Tester.f1(test_label, prediction)
    Tester.calculate_latency(result)
