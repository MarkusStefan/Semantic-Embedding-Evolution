# Semantic-Embedding-Evolution
- Original paper: [Dynamic Word Embeddings](https://arxiv.org/pdf/1703.00607) 
- Code: [GitHub Repository](https://github.com/yifan0sun/DynamicWord2Vec)

## Dynamic Word Embeddings
Tracking how contextualized word representations evolve over time (e.g., `Apple` as a fruit in 1990 vs. a company in 2010).

### Goal 1 — Reproduce Baseline
- Environment: Python 3.xx with PyTorch.
- Implement time-sliced word2vec or similar static embedding model to measure semantic drift across corpora.

### Goal 2 — Extensions
Running both extensions does not require any external downloads, everything is self-contained. Run on JupyterHub with sufficient RAM and GPU support!

#### Extension 1: Embedding Evolution during Small-Scale GPT2 Fine-tuning

See `notebooks/exp1.ipynb`:

- **Objective:** Measure how a small GPT-2 model's embeddings evolve when a word acquires a new meaning via fine-tuning (example: "python" as reptile vs programming language).
- **Methodology:** initializes a compact GPT-2, builds synthetic + real sentence datasets, fine-tunes across a hyperparameter grid, and saves checkpoints during training.
- **Measurements:** extracts static token embeddings (WTE), contextual token embeddings (token-specific last-layer vectors), and sentence embeddings (mean pooling) before, during, and after fine-tuning.
- **Assessments & visualizations:** clustering, PCA/t-SNE projections, trajectory plots, embedding-drift curves, and multiple diagnostics (perplexity, next-token probabilities).
- **New evaluations:** (1) Sampled Generation Assessment — generates many sampled completions per prompt to estimate probability shifts for target senses; (2) Polysemy Disambiguation Test — measures acquisition vs forgetting using context-sensitive prompts.


#### Extension 2: Dynamic Model2Vec - Distillation of fine-tuned Language Models

See `notebooks/exp2.ipynb`:

- **Objective:** Distill fine-tuned language models into static embedding models and evaluate semantic drift between modern and historical (fine-tuned) static models.
- **Methodology:** preprocesses/pack-slices a large corpus (PG-19), fine-tunes or trains several teacher models (finetuned, scratch-trained, aggressive tuning), then uses model2vec distillation to produce static embedding models.
- **Alignment & analysis:** obtains common vocab anchors, computes orthogonal Procrustes alignment between embedding spaces, and aligns variant embeddings into a shared space.
- **Measurements & visualizations:** encodes words and anchors, computes cosine-based drift scores, identifies top drifters, and visualizes semantic trajectories using joint PCA and targeted anchor contexts.
- **Dashboard & probes:** builds a multi-variant dashboard comparing control vs fine-tuned variants across curated target words (e.g., "cloud", "gay", "post", "bug"), with context-word overlays and per-variant subplots.

