```
Embedding_word2vec/
├─ data/                                  # Artifacts & small fixtures
│  ├─ embed_in.pt                         # Learned input vectors  (|V| × d)
│  ├─ embed_out.pt                        # Learned output vectors (|V| × d)
│  ├─ train_starts.bin                    # Start offsets of train docs/spans
│  ├─ valid_starts.bin                    # Start offsets of valid docs/spans
│  ├─ vocab.pt                            # {old id -> new id, token -> id, id -> token, freq, keep probs, ...}
│  └─ test_embed/                         # toy set for smoke test
│     ├─ n_corpus.bin                     # 
│     ├─ o_corpus.bin
│     └─ vocab.pt
│
├─ srcs/
│  ├─ data_pipeline/                     # Build vocab → encode corpus → ID pipeline
│  │  ├─ _count_fast.pyx                 # Cython: token/bigram counter (pass 1)
│  │  ├─ _encode_corpus.pyx              # Cython: encode to token IDs (pass 2)
│  │  ├─ count_tokens.py                 # func wrapper for _count_fast
│  │  ├─ encode_corpus.py                # func wrapper for _encode_corpus
│  │  ├─ data_pipe_ids.py                # Iterable over (center_id, context_id)
│  │  ├─ review_dataset_iter.py          # Iterator for Reviews_and_TV datasets
│  │  └─ setup.py                        # Build Cython extensions
│  │
│  ├─ embedding/                         # Training
│  │  └─ embedding_ids.py                # SGNS using ID pipeline 
│  │
│  ├─ notebook/                          # Experiments
│  │  ├─ train_reviews_ids.ipynb         # Kaggle reviews_and_TV
│  │
│  └─ test/                              # Verification & speed
│     ├─ semantic.py                     # Simple semantic similarity checks
│     ├─ speedtest.py                    # mul+sum vs bmm vs einsum
│     ├─ test_encode.py                  # Encode-corpus correctness
│     └─ test_fast_count.py              # Counter correctness
│
├─ README.md                              # You are here
└─ LICENSE
```


## Approach

Implement Skip-Gram with Negative Sampling (SGNS) from scratch, with a custom preprocessing pipeline optimized for large corpora.
The pipeline is two-pass:

1. **Count tokens & candidate bigrams**
   - Collect unigram counts
   - Collect top-K high-frequency token pairs

2. **Build final vocab + re-encode corpus**
   - Apply `min_count` threshold
   - Drop tokens not in final vocab
   - Re-encode corpus into integer ID streams

Additionally apply POS-aware bigram merging, which helps preserve meaningful multi-word units:
- NOUN + NOUN
- NEGATION + ADJ/ADV (e.g., `not_good`)
- VERB + PARTICLE (e.g., `pick_up`)

Counts for merged pairs are lightly smoothed before integration.

To improve signal quality, subsampling keep-probabilities are computed with POS-aware masks, reducing noise while preserving sentiment-bearing adjectives/adverbs.

Training is implemented in PyTorch using efficient ID-based sampling.

# Key Improvements over Vanilla Word2Vec

- Fast preprocessing via Cython (`_count_fast.pyx`, `_encode_corpus.pyx`)
- POS-aware phrase merging (e.g., `not_good`, `good_movie`, `pick_up`)
- POS-conditioned subsampling keeps important token classes (ADJ/ADV)
- Clean ID pipeline ensures no stray OOV during training
- Simple semantic evaluation


# 📊 Evaluation

Nearest neighbors
```
happy       → camper, satisfied, pleased
good        → ol'_days, documentry, not_surprising
bad         → guys, horrible, ruined
excellent   → giovanni, suberb, stephan
outstanding → phenomenal, superb, exceptional
masterpiece → finest, must-see, assuredly
awful       → lousy, dreadful, horrible
terrible    → horrible, lousy, ruined
boring      → not_help, predictable, predicable
cringe      → wince, cringeworthy, sophmoric
```

Analogy
```
king - man + woman → queen, mistress, prince
```

2D Embedding Visualization
<p align="center">
  <img src="results/embedding_pca.png" width="700">
</p>


# 🚀 How to Run 
```
# from project root
cd srcs/data_pipeline
python setup.py build_ext --inplace
cd ../../

python srcs/data_pipeline/count_tokens.py
python srcs/data_pipeline/encode_corpus.py
python -m srcs.embedding.embedding_ids
```

# Default parameter
```
top_k = 10000        (pick top_k highest freq count)
min_pair_count = 6
min_count = 25
t = 6e-6
```


