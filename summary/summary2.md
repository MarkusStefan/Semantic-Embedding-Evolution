This technical summary is designed for effective study, bridging high-level intuition with low-level architectural details and code logic found in the slides.

---

### 1. Static Word Embeddings & Optimization
**Intuition:** Distributional semantics—a word’s meaning is defined by the words that frequently appear near it.

*   **SVD & Co-occurrence Matrices:**
    *   **Method:** Build a matrix $X$ where $X_{ij}$ is the count of word $j$ in the context of word $i$. Use Singular Value Decomposition ($X = U\Sigma V^T$) to reduce dimensionality.
    *   **Pros:** Efficiently uses global statistics; fast training.
    *   **Cons:** High-frequency words (the, a) dominate (requires log-scaling or capping); the matrix is massive and hard to update with new words.
*   **Word2Vec (Skip-gram):**
    *   **Method:** Iterates through a corpus. For every center word, it tries to predict surrounding context words.
    *   **Logic:** Uses a dot product between center vector $v_c$ and context vector $u_o$.
*   **GloVe (Global Vectors):**
    *   **Rationale:** Combines counts (SVD) with prediction (Word2Vec). It performs a weighted least squares regression on the log-counts of the co-occurrence matrix.

---

### 2. Foundations: Classification & Training
**Intuition:** Neural networks learn non-linear decision boundaries, whereas basic softmax/logistic regression can only draw straight lines (hyperplanes).

*   **Cross-Entropy Loss:** The standard for classification. It measures the "distance" between the predicted probability distribution and the actual one-hot label.
*   **Regularization:** 
    *   **L2:** Adds a penalty for large weights to the loss function.
    *   **Dropout:** Randomly sets neurons to zero during training, forcing the network to develop redundant pathways and preventing overfitting.
*   **Learning Rates & Optimizers:**
    *   **Rationale:** High rates cause the model to diverge ("shoot off the cliff"); low rates take forever.
    *   **Adam/AdamW:** Adaptive optimizers that adjust the learning rate for each individual parameter based on previous gradients.

**Layer Construction (Classification Logic):**
```python
# A simple window classifier / Feed-forward layer
class WindowClassifier(nn.Module):
    def __init__(self, window_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        # Rationale: Concatenate vectors in the window: input = window_size * embed_dim
        self.input_layer = nn.Linear(window_size * embed_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU() # Non-linearity

    def forward(self, x):
        # x is concatenated word vectors
        h = self.relu(self.input_layer(x))
        logits = self.output_layer(h)
        return logits # Passed to CrossEntropyLoss
```

---

### 3. Language Modeling: n-grams to RNNs
**Intuition:** Language models assign probabilities to sequences of words. $P(w_{next} | w_{history})$.

*   **n-gram Models:** Predict based on a fixed history of size $n-1$.
    *   **Sparsity Problem:** If a 4-gram never appeared in training, the probability is 0. 
    *   **Metric (Perplexity):** Represents the "branching factor." If Perplexity = 10, the model is as "confused" as if it were choosing between 10 equally likely words. **Lower is better.**
*   **Recurrent Neural Networks (RNNs):**
    *   **Rationale:** To process input of *any length* using the same weights $W$ at every step.
    *   **Vanishing Gradient:** As we backpropagate through time, the gradient is multiplied by the same weight matrix repeatedly. If the weights are small, the signal disappears, and the model forgets the start of the sentence.
    *   **Gradient Clipping:** If the gradient norm is too high (Exploding Gradient), it is scaled down to prevent $NaN$ errors.

**RNN Architecture Logic:**
```python
class VanillaRNNStep(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Logic: Combine current input and previous hidden state
        self.x_to_h = nn.Linear(input_dim, hidden_dim)
        self.h_to_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        # h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
        h_t = torch.tanh(self.x_to_h(x_t) + self.h_to_h(h_prev))
        return h_t
```

---

### 4. Gated RNNs (LSTM/GRU) & Seq2Seq
**Intuition:** LSTMs/GRUs provide a "highway" (cell state) for information to travel across many steps without being modified by a matrix multiplication at every step.

*   **LSTM Gates:** **Forget** (what to drop), **Input** (what to add), **Output** (what to show).
*   **Seq2Seq (Encoder-Decoder):**
    *   **Encoder:** Processes source (e.g., French) into a single context vector.
    *   **Decoder:** A conditional language model that generates target (e.g., English).
    *   **Information Bottleneck:** The single context vector cannot represent a very long sentence, leading to poor translation.

**LSTM Gate Logic:**
```python
# Inside an LSTM Cell
def forward(self, x_t, h_prev, c_prev):
    # Rationale: Compute 4 internal projections at once (i, f, o, g)
    combined_linear = self.gate_proj(torch.cat([x_t, h_prev], dim=-1))
    i, f, o, g = combined_linear.chunk(4, dim=-1)
    
    c_t = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
    h_t = torch.sigmoid(o) * torch.tanh(c_t)
    return h_t, c_t
```

---

### 5. Attention Mechanisms & Transformers
**Intuition:** Attention allows the decoder to "look back" at specific words in the encoder at every generation step. **Self-attention** allows words *within the same sentence* to look at each other.

*   **Self-Attention ($Q, K, V$):**
    *   **Query ($Q$):** "What I am looking for."
    *   **Key ($K$):** "What I have to offer."
    *   **Value ($V$):** "The actual information I provide."
*   **Transformer Architecture:** 
    *   Eliminates recurrence; uses only attention.
    *   **Parallelization:** Because there are no sequential steps, the whole sentence is processed at once ($O(1)$ interaction distance).
    *   **Positional Encoding:** Added to inputs to preserve the notion of word order since attention is "set-based."

**Self-Attention Logic:**
```python
def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v are output of linear layers: nn.Linear(d_model, d_k)
    d_k = q.size(-1)
    # 1. Similarity Scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None: # Logic: Causal masking (don't look at the future)
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 2. Softmax to get probability distribution
    weights = torch.softmax(scores, dim=-1)
    # 3. Output is weighted sum of values
    return torch.matmul(weights, v)
```

---

### 6. Modern Paradigms: Pre-training & Fine-tuning
**Intuition:** Start with a model that already "knows" language (Pre-training), then teach it a specific job (Fine-tuning).

*   **BERT (Encoder-only):** Pre-trained using **Masked LM** (filling in the blanks). Great for understanding/classification.
*   **GPT (Decoder-only):** Pre-trained using standard next-word prediction. Great for generation.
*   **Instruction Fine-tuning:** 
    *   **Problem:** LMs often just repeat the prompt or continue the text instead of answering.
    *   **Solution:** Fine-tune on (Instruction, Response) pairs to teach the model to behave like an assistant.
*   **Evaluation:**
    *   **BLEU Score:** Measures $n$-gram precision. Uses a **Brevity Penalty** to punish translations that are too short (which might otherwise get artificially high precision scores).
    *   **Chatbot Arena:** Side-by-side human evaluation (the gold standard for modern LLMs).

---

### 7. Decoding: Beam Search
**Intuition:** Greedy decoding (taking the single best word) is short-sighted. Beam search tracks the $k$ most likely partial sentences.

*   **Technical Detail:** We use the log-probability of sequences. Because longer sequences have more negative numbers added together, they naturally get lower scores.
*   **Fix:** **Length Normalization.** Divide the total log-score by the length of the sentence to compare long and short hypotheses fairly.