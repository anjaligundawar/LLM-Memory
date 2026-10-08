"""
metrics.py  -- MODULE 4 (Abhay), minimal version used by the demos.

  P@K  = relevant found in top-K / K
  R@K  = relevant found in top-K / total relevant
  MRR  = 1 / rank of first relevant result (0 if none)
"""


def precision_at_k(retrieved_ids, relevant_ids, k):
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for i, mid in enumerate(retrieved_ids, start=1):
        if mid in relevant_ids:
            return 1 / i
    return 0.0
