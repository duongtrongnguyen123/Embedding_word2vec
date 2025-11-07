import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from adjustText import adjust_text


def pca_2d(X):
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vh[:2].T

def visualize_2d(X2d, labels, colors, queries, title="Nearest neighbors (PCA 2D)"):
    X2d = X2d.detach().cpu().numpy()
    xs = X2d[:, 0]
    ys = X2d[:, 1]

    plt.figure(figsize=(9, 7))
    plt.scatter(xs, ys, c=colors, s=12)

    texts = []
    for i, w in enumerate(labels):
        texts.append(plt.text(xs[i], ys[i], w, fontsize=8))

    for q in queries:
        idx = labels.index(q)
        plt.scatter([xs[idx]], [ys[idx]],
                    s=20, facecolors='none', edgecolors='k', linewidths=2)

    adjust_text(
        texts,
        x=xs, y=ys,
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.4)
    )

    plt.title(title)
    plt.xlabel("PC1 / dim1"); plt.ylabel("PC2 / dim2")
    plt.axhline(0, color="lightgray", lw=1)
    plt.axvline(0, color="lightgray", lw=1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()    

base_dir = os.path.dirname(os.path.abspath(__file__))
embed_i_dir = os.path.join(base_dir, "..", "..", "data", "embed_in.pt")
embed_o_dir = os.path.join(base_dir, "..", "..", "data", "embed_out.pt")
vocab_dir = os.path.join(base_dir, "..", "..", "data", "vocab.pt")

w_in = torch.load(embed_i_dir, map_location='cpu')
w_out = torch.load(embed_o_dir, map_location='cpu')
vocab = torch.load(vocab_dir, map_location='cpu')

word2id = vocab["word2id"]
id2word = vocab["id2word"]

w = (w_in + w_out) / 2.0
w_mean = w - w.mean(dim=0, keepdim=True)

U, S, Vh = torch.linalg.svd(w_mean, full_matrices=False)
m = 1
P = Vh[:m].T @ Vh[:m]       
w_new = w_mean - (w_mean @ P)   
w_norm = F.normalize(w_new, dim=1)

ids = [word2id["king"], word2id["queen"], word2id["man"], word2id["woman"]]
expect_queen = w_norm[ids[0]] - w_norm[ids[2]] + w_norm[ids[3]]
expect_queen = F.normalize(expect_queen, dim=0)

cos = w_norm @ expect_queen     
topv, topi = torch.topk(cos, k=10)
print("Analogy (king - man + woman):")
for j in range(10):
    print(f"{id2word[topi[j].item()]:<16} Cos: {topv[j].item():.3f}")



queries = ["cat", "car", "film", "drama", "snack"]
qid = [word2id[q] for q in queries]

Q = len(queries)
q = w_norm[qid]
sims = q @ w_norm.T                      
for i, qi in enumerate(qid):
    sims[i, qi] = -1e9                  

k = 10
topv, topi = torch.topk(sims, k=k, dim=1)

labels = []
owner  = []  
for qi, qword in enumerate(queries):
    if qword in word2id and qword not in labels:
        labels.append(qword); owner.append(qi)
    for j in range(k):
        w = id2word[topi[qi, j].item()]
        if w not in labels:
            labels.append(w); owner.append(qi)

ids_plot = torch.tensor([word2id[w] for w in labels])
X = w_norm[ids_plot]
X2d = pca_2d(X)
visualize_2d(X2d, labels, owner, queries, title=f"Nearest neighbors of {queries}")
