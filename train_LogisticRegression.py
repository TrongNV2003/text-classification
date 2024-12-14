import time
import argparse
from training.dataloader import Dataset
from training.trainer import LogisRegression, Vectorizer
from evaluate.evaluate_output import f1, calculate_latency, result_recorder

vec = Vectorizer()
lr = LogisRegression()

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
    classifier = lr.train(train_text_vect, train_label)
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

    result_file = "result/output_lr.json"
    f1(lr, test_text_vect, test_label)
    result_recorder(result_file, result)
    calculate_latency(result_file)
    