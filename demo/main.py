import argparse
import numpy as np
from lbp_encoder import compute_features, get_preprocess_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
from mb_encoder import Encoder
import cv2

def evaluate_search(q_feats, q_labels, g_feats, g_labels, topk=[1,5,10]):
    """Cosine similarity Top-K retrieval and print accuracy."""
    sims = q_feats.dot(g_feats.T) / (
        np.linalg.norm(q_feats, axis=1)[:,None] * np.linalg.norm(g_feats, axis=1)[None,:] + 1e-6
    )
    for k in topk:
        correct = sum(
            lbl in [g_labels[j] for j in np.argsort(-sims[i])[:k]]
            for i, lbl in enumerate(q_labels)
        )
        print(f"Top-{k}: acc={correct/len(q_labels):.4f}")
    return sims

def evaluate_knn(q_feats, q_labels, g_feats, g_labels, max_k=9):
    """
    Evaluate accuracy using KNN classifier with k from 1 to max_k.
    Args:
        q_feats: Query features (numpy array)
        q_labels: Query labels (numpy array or list)
        g_feats: Gallery features (numpy array)
        g_labels: Gallery labels (numpy array or list)
        max_k: Max value of k to evaluate (default: 9)
    """
    q_labels = np.array(q_labels)
    g_labels = np.array(g_labels)

    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights="distance")
        knn.fit(g_feats, g_labels)
        preds = knn.predict(q_feats)
        acc = accuracy_score(q_labels, preds)
        print(f"K={k}: acc={acc:.4f}")

def visualize_results(q_imgs, q_labels, g_imgs, g_labels, sims, topk=5,
                      img_size=(100, 100), text_height=20, border_thickness=2,
                      output_path='search_results.png'):
    """
    Visualize search results: each row is a query and its top-k matches.
    True matches have green border, false have red.
    """
    rows = []
    w, h = img_size
    fh = text_height
    for qi, q_lbl in enumerate(q_labels):
        q_img = cv2.imread(q_imgs[qi])
        q_img = cv2.resize(q_img, img_size)
        canvas_q = np.full((h+fh, w, 3), 255, dtype=np.uint8)
        canvas_q[:h] = q_img
        canvas_q = cv2.copyMakeBorder(canvas_q, border_thickness, border_thickness,
                                      border_thickness, border_thickness,
                                      cv2.BORDER_CONSTANT, value=(255,0,0))
        cv2.putText(canvas_q, f"Q:{q_lbl}", (5, h+fh-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        row_imgs = [canvas_q]
        idxs = np.argsort(-sims[qi])[:topk]
        for gi in idxs:
            g_img = cv2.imread(g_imgs[gi])
            g_img = cv2.resize(g_img, img_size)
            canvas_g = np.full((h+fh, w, 3), 255, dtype=np.uint8)
            canvas_g[:h] = g_img
            color = (0,255,0) if g_labels[gi] == q_lbl else (0,0,255)
            canvas_g = cv2.copyMakeBorder(canvas_g, border_thickness, border_thickness,
                                          border_thickness, border_thickness,
                                          cv2.BORDER_CONSTANT, value=color)
            cv2.putText(canvas_g, str(g_labels[gi]), (5, h+fh-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            row_imgs.append(canvas_g)
        rows.append(np.hstack(row_imgs))
    result = np.vstack(rows)
    cv2.imwrite(output_path, result)
    print(f"[+] Saved visualization to {output_path}")
    return result

def split_dataset(image_paths, labels):
    """One image per identity as query, rest as gallery."""
    q_imgs, q_labels, g_imgs, g_labels = [], [], [], []
    seen = set()
    for path, lbl in zip(image_paths, labels):
        if lbl not in seen:
            q_imgs.append(path); q_labels.append(lbl); seen.add(lbl)
        else:
            g_imgs.append(path); g_labels.append(lbl)
    return q_imgs, q_labels, g_imgs, g_labels

def load_dataset(data_dir):
    """Load image paths and labels."""
    image_paths, labels = [], []
    label_map, idx = {}, 0
    for person in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, person)
        if not os.path.isdir(p): continue
        for f in sorted(os.listdir(p)):
            if f.lower().endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(p, f))
                if person not in label_map:
                    label_map[person] = idx; idx += 1
                labels.append(label_map[person])
    return image_paths, labels

def main():
    parser = argparse.ArgumentParser(
        description="Face Retrieval Pipeline with Visualization")
    
    parser.add_argument('--data_dir', required=True,
                        help="Dataset root folder")
    parser.add_argument('--output', default='search_results.png',
                        help="Output path for visualization image")
    parser.add_argument('--range', type=str, default='1:10',
                        help="Preview range of queries as start:end (zero-based, end-exclusive)")
    args = parser.parse_args()

    imgs, labels = load_dataset(args.data_dir)
    
    q_imgs, q_lbls, g_imgs, g_lbls = split_dataset(imgs, labels)
    print(f"Total queries: {len(q_imgs)}, Gallery: {len(g_imgs)}")

    # frontal, profile, eye = get_preprocess_model()
    # g_feats = compute_features(g_imgs, frontal, profile, eye)
    # q_feats = compute_features(q_imgs, frontal, profile, eye)

    encoder = Encoder(device="cpu")
    g_feats = encoder.encode(g_imgs)
    q_feats = encoder.encode(q_imgs)

    sims = evaluate_search(q_feats, q_lbls, g_feats, g_lbls)

    evaluate_knn(q_feats, q_lbls, g_feats, g_lbls, max_k=9)

    # Apply preview range if provided
    if args.range:
        try:
            start, end = map(int, args.range.split(':'))
            q_imgs = q_imgs[start:end]
            q_lbls = q_lbls[start:end]
            sims = sims[start:end]
            print(f"Previewing queries from index {start} to {end} (exclusive)")
        except Exception as e:
            print(f"Invalid range '{args.range}', expected format start:end. Error: {e}")

    visualize_results(q_imgs, q_lbls, g_imgs, g_lbls, sims,
                      topk=5, img_size=(128, 128), text_height=24,
                      border_thickness=2, output_path=args.output)

if __name__ == '__main__':
    main()
