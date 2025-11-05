import torch
import os

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embed_i_dir = os.path.join(base_dir, "..", "..", "data", "embed_in.pt")
    embed_o_dir = os.path.join(base_dir, "..", "..", "data", "embed_out.pt")
    vocab_dir = os.path.join(base_dir, "..", "..", "data", "vocab.pt")

    w_in = torch.load(embed_i_dir, map_location='cpu')
    w_out = torch.load(embed_o_dir, map_location='cpu')
    vocab = torch.load(vocab_dir, map_location='cpu')
    word2id = vocab["word2id"]
    id2word = vocab["id2word"]

    ids = [word2id["king"], word2id['queen'], word2id['man'], word2id['woman']]

    w = (w_in + w_out) / 2

    w_mean = w - w.mean(dim=0, keepdim=True)

    U, S, Vh = torch.linalg.svd(w_mean, full_matrices=False)
    m = 2
    P = Vh[:m].T @ Vh[:m]                               
    w_new = w_mean - (w_mean @ P)
    
    w_norm = w_new / (w_new.norm(dim=1, keepdim=True) + 1e-9)
    x_norm = w_norm[ids]


    expect_queen = w_norm[ids[0]] - w_norm[ids[2]] + w_norm[ids[3]]

    cos = expect_queen @ w_norm.T
    topv, topi = torch.topk(cos, k=10)
    print("Analogy (king - man + woman) ")
    for j in range(10):
        print(f"  {id2word[topi[j].item()]:<12} Cos: {topv[j].item():.3f}")

    

