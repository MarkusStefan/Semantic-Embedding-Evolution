# Semantic-Embedding-Evolution
- Original paper: [Dynamic Word Embeddings](https://arxiv.org/pdf/1703.00607) 
- Code: [GitHub Repository](https://github.com/yifan0sun/DynamicWord2Vec)

## Dynamic Word Embeddings
Tracking how contextualized word representations evolve over time (e.g., `Apple` as a fruit in 1990 vs. a company in 2010).

### Goal 1 — Reproduce Baseline
- Environment: Python 3.xx with PyTorch.
- Implement time-sliced word2vec or similar static embedding model to measure semantic drift across corpora.

### Goal 2 — Extensions
1. Replace word2vec with BERT to compare contextual clusters vs. static embeddings.
2. Fine-tune BERT on a subset of data to see if meanings can be deliberately shifted (e.g., push `amazon` toward jungle semantics).
3. Measure whether fine-tuning changes token semantics by:
	- Pre-training on data before `YYYY`.
	- Fine-tuning on data after `YYYY`.
	- Comparing with a model trained on the full corpus at once.
	- Estimating how much fine-tuning is required to induce noticeable shifts.
4. Explore adversarial attacks that attempt to fool models by confusing semantic meanings.
5. Detect change points: when does a token’s dominant meaning flip?
6. Apply dimensionality reduction (e.g., PCA, t-SNE, UMAP) and visualize the temporal trajectory of embeddings.