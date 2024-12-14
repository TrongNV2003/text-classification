import json
import numpy as np
from sklearn.metrics import f1_score

def calculate_latency(output_file):
    with open(output_file, "r") as f:
        data = json.load(f)

    process_times = [item["process_time"] for item in data]
    p95_latency = np.percentile(process_times, 95)
    average_latency = np.mean(process_times)

    print(f"P95 Latency: {p95_latency:.10f} secs")
    print(f"Average: {average_latency:.10f} secs")

def f1(model, vector, label):
    prediction = model.predict(vector)
    score = f1_score(label, prediction, average='macro')
    print(f"F1-score: {score}")

def result_recorder(path, predict_time):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(predict_time, file, indent=4, ensure_ascii=False)
        print(f"Saved in {path}")

