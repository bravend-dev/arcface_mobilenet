import numpy as np

def compute_topk_accuracy(retrieved_indices, query_labels, collection_labels):
    correct = 0
    for query_idx, indices in enumerate(retrieved_indices):
        query_label = query_labels[query_idx]
        retrieved_labels = collection_labels[indices]
        if query_label in retrieved_labels:
            correct += 1
    return correct / len(query_labels)


def compute_precision_at_k(retrieved_indices, query_labels, collection_labels):
    precisions = []
    for query_idx, indices in enumerate(retrieved_indices):
        query_label = query_labels[query_idx]
        retrieved_labels = collection_labels[indices]
        correct_count = np.sum(retrieved_labels == query_label)
        precisions.append(correct_count / len(indices))
    return np.mean(precisions)


def compute_map_at_k(retrieved_indices, query_labels, collection_labels):
    average_precisions = []
    for query_idx, indices in enumerate(retrieved_indices):
        query_label = query_labels[query_idx]
        retrieved_labels = collection_labels[indices]

        correct = 0
        precision_sum = 0
        for i, label in enumerate(retrieved_labels):
            if label == query_label:
                correct += 1
                precision_at_i = correct / (i + 1)
                precision_sum += precision_at_i

        if correct > 0:
            average_precision = precision_sum / correct
        else:
            average_precision = 0.0
        average_precisions.append(average_precision)

    return np.mean(average_precisions)

