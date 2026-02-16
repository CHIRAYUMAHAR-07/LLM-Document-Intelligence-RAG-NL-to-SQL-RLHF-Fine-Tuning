
import time
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score

def simulate_predictions(n=200):
    y_true = [random.randint(0,1) for _ in range(n)]
    y_pred = [random.randint(0,1) for _ in range(n)]
    return y_true, y_pred

def evaluate_model():
    y_true, y_pred = simulate_predictions()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print("Benchmark Results")
    print("-----------------")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    start = time.time()
    evaluate_model()
    print("Execution Time:", round(time.time() - start, 3), "seconds")
