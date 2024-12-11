import json
import numpy as np

def calculate_latency(output_file):
    with open(output_file, "r") as f:
        data = json.load(f)

    process_times = [item["process_time"] for item in data]

    p95_latency = np.percentile(process_times, 95)
    average_latency = np.mean(process_times)

    print(f"P95 Latency: {p95_latency:.10f} secs")
    print(f"Average: {average_latency:.10f} secs")


if __name__ == "__main__":
    output_file = "result/output_svm.json"
    calculate_latency(output_file)
