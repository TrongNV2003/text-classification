import argparse
import time

from sklearn.linear_model import LogisticRegression

from training.dataloader import Dataset
from training.evaluate import (
    Tester,
    calculate_accuracy,
    calculate_latency,
    print_metrics,
)
from training.trainer import AlgoTrainer, Vectorizer

vec = Vectorizer()
model = LogisticRegression(random_state=42)

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

    train_text_vect = vec.fit_transform(train_text)
    test_text_vect = vec.transform(test_text)

    start_time = time.time()

    trainer = AlgoTrainer(model)
    trainer.train(train_text_vect, train_label)

    end_time = time.time()
    process_time = round(end_time - start_time, 6)
    print(f"Training time: {process_time}")

    latencies = []
    for test in test_text_vect:
        start_time = time.time()
        trainer.predict(test)
        end_time = time.time()

        latencies.append(end_time - start_time)

    prediction = model.predict(test_text_vect)
    calculate_accuracy(test_label, prediction)
    print_metrics(test_label, prediction)
    calculate_latency(latencies)
