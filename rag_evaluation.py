
import numpy as np

def mean_reciprocal_rank(results, ground_truth):
    mrr = 0
    for res, gt in zip(results, ground_truth):
        if gt in res:
            rank = res.index(gt) + 1
            mrr += 1 / rank
    return mrr / len(results)

def recall_at_k(results, ground_truth, k=3):
    recall = 0
    for res, gt in zip(results, ground_truth):
        if gt in res[:k]:
            recall += 1
    return recall / len(results)

def ragas_style_score(context_precision, answer_relevance):
    return 0.5 * context_precision + 0.5 * answer_relevance

if __name__ == "__main__":
    retrieved = [["doc1","doc2","doc3"],["doc3","doc4","doc5"]]
    ground_truth = ["doc2","doc4"]
    
    print("MRR:", mean_reciprocal_rank(retrieved, ground_truth))
    print("Recall@3:", recall_at_k(retrieved, ground_truth, k=3))
    print("RAGAS-style Score:", ragas_style_score(0.8,0.75))
