
Compile a complete explanatory summary for effective study of the following topics from the pdf slides using intuition, technical details and code chunks where applicable and helpful:

- Word2Vec, Glove
- Stochastic Gradient Decent, SVD & co-occurence matrices + pros & cons
- Cross-entropy for text classification, NER, window classifier
- Regularization, learning rates
- n-gram models, language modeling,
- RNNs, vanishing/exploding gradient, clipping, LSTMs GRU, Seq2Seq, NMT, Beam search decoding, BLEU score, perplexity score
- Attention, self-attention, attention in Seq2Seq, attention in transformer, the transformer architecture, decoder only architecture
- pre-training & fine-tuning techniques
- instruction fine-tuning

### 1. Word Embeddings: Capture Semantic Similarity
**Rationale:** One-hot vectors are sparse and have no notion of similarity. We need dense vectors where the dot product represents semantic relationships (Distributional Semantics).

*   **Skip-gram (Word2Vec):** Predicts context words $u_o$ from a center word $v_c$.
*   **GloVe:** Uses global co-occurrence counts $X_{ij}$ and minimizes the difference between the dot product of vectors and the log-count.

**Architecture Snippet (Skip-gram logic):**
```python
class SkipGramStep(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Two separate embedding tables: one for center, one for context
        self.v = nn.Embedding(vocab_size, embed_dim) # Center
        self.u = nn.Embedding(vocab_size, embed_dim) # Context

    def forward(self, center_idx, context_idx):
        center_vec = self.v(center_idx)   # [batch, dim]
        context_vec = self.u(context_idx) # [batch, dim]
        # Dot product measures similarity
        logits = torch.sum(center_vec * context_vec, dim=1)
        return logits # Then passed to CrossEntropyLoss
```


#### NER
Window classification, similar to Word2Vec but only assigns high probability to center word if it belongs to a named entity class, unlike in Word2Vec where plausible center words always receive high probabilities.

---

### Deep Learning
- Backpropagation: Chain rule to compute gradients.
- Stochastic Gradient Descent (SGD): Update weights using mini-batches to approximate full gradient.
- Activation functions: utility and problems of Sigmoid, Tanh; alternatives like ReLU, Leaky ReLU.
- Regularization: Techniques like Dropout and L2 to prevent overfitting.
    - Dropout randomly zeroes activations during training.
    - L2 regularization adds a penalty term to the loss function proportional to the square of the weights.
```python
# Example of L2 Regularization in PyTorch
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5) # weight_decay is L2 penalty
```
- Weight Initialization: never init with zeros, but small random values to break symmetry; Xavier init uses variance inversely proportional to number of inputs/outputs among layers.
- Optimizers: SGD is fine, but requires careful tuning of learning rates; Adam adapts learning rates per parameter; often helpful to use lr decay; at least order of magnitude should be correctly chosen.

### Language Modeling
- predicting which word comes next given previous words (assigning probabilities to pieces of text).
- n-gram models: next word only depends on the (n-1) previous words; use counts of n-grams $\frac{count((n-1)-gram, w)}{count((n-1)-gram)}$to estimate probabilities of the next word $w$; suffer from data sparsity which can be solved adding *smoothing* by assigning $\delta$ probability to all words; if the $(n-1)$-gram was never seen, *back off* to $(n-2)$-gram etc.; increasing $n$ worsens the sparsity problem.
- Neural LM: predict next word given (n-1) previous words using a neural network; word embeddings as input; hidden layers with non-linearities; output layer with softmax over vocabulary; solves sparsity problem and storage problem for large corpora and $n$; still, *fixed window* is too small and rigid.
```python
class NGramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    def forward(self, context_idxs):
        embeds = self.embeddings(context_idxs).view(1, -1)
        out = F.relu(self.fc1(embeds))
        out = self.fc2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
```


### 2. Recurrent Neural Networks (RNNs)
**Rationale:** To process *variable-length* sequences. The "hidden state" acts as a memory that is updated at every step.
**Evaluation Metric:** **Perplexity (PP)**. $PP = \exp(J(\theta))$, where $J$ is cross-entropy loss (summed scalar values resulting of the ground truth sparse vector multiplied with the predicted log-probs) averaged over sentences during SGD. It represents the "branching factor"—how confused the model is when picking the next word.
- apply weights $W$ repeatedly at each time step to update hidden state $h_t$ based on input $x_t$ and previous hidden state $h_{t-1}$.
$$
h_t = \tanh(W_h h_{t-1} + W_x E x_t + b)
$$
- $x_t$: one-hot vector of current word
- $E$: embedding matrix
- (+) process any lenght input, using context from (many) previous words, model size stays constant, symmetric processing of each time step
- (-) training is slow (no parallelization), vanishing/exploding gradients, hard to learn long-term dependencies
- Gradient: is the sum of gradients at each step (since we apply $W_h$ and $W_x$ repeatedly) --> can explode (too large) or vanish (too small) exponentially with number of time steps.
- RNNs are not necessarily the same as LMs, but can be used as such by predicting next word from hidden state at each time step; used for:
    - POS/NER tagging
    - sentence classification using the final hidden state $h_T$ or aggregate of all hidden states like mean, max, sum...(sentiment)
    - encoder module (question answering, summarization) by aggregating all hidden states into a context vector
    - text generation as in (neural) machine translation, speech recognition, summarization (seq2seq).
- Perplexity calculation (how confused model is in the dist. over next words):
$$
perplexity = \prod_{t=1}^{T} \frac{1}{P(x_t | x_{<t})} = \exp\left(\frac{1}{T} \sum_{t=1}^{T}- \log P(x_t | x_{<t})\right)
$$
```python
def calculate_perplexity(model, data, criterion):
    total_loss = 0
    total_words = 0
    for inputs, targets in data:
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        total_loss += loss.item() * targets.numel()
        total_words += targets.numel()
    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)
    return perplexity
```

#### Vanilla RNN
**Problem:** Vanishing Gradients. Because the same matrix $W_h$ is multiplied repeatedly, the gradient shrinks exponentially, making long-distance dependencies impossible to learn.
- gradient signal from far away (early hidden states) is very small compared to nearby states --> only the last few next-word predictions of the last words of a sentence are learned well
    - we also cannot quanitfy whether a dependency between early and late words id present or whether the parameters capture it correctly.

- if gradient explodes, large update steps --> unstable training and overshooting
- fixes: gradient clipping, skip connections, memory & forget gates (LSTM, GRU)
    - Gradient Clipping: limit the norm of the gradient to a maximum value
    - intuition: take a step in the same gradient direction, but smaller if the gradient is too large
    ```python
    gradient = loss.backward()
    if gradient.norm() > max_norm_threshold:
        gradient = gradient * (max_norm_threshold / gradient.norm())
    optimizer.step()
    ```
- small gradients are multiplied repeatedly --> vanish; large gradients --> explode.
- activation functions that naturally squish values (Sigmoid, Tanh) worsen the problem

```python
class VanillaRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Rationale: Combine x_t and h_{t-1} into a single linear projection
        self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        combined = torch.cat((x_t, h_prev), dim=1)
        h_t = torch.tanh(self.linear(combined))
        return h_t
```

#### LSTM (Long Short-Term Memory)
**Rationale:** Adds a "Cell State" ($c_t$) that acts as a long-term memory highway. Gates control the flow of information, specifically the **Forget Gate**, which allows the model to reset its memory to avoid vanishing gradients.
- fixing vanishing gradients:
    - problem: $h_t$ is constantly overwritten at each time step
    - solution: add a "Cell State" $c_t$ that runs through the entire sequence with only minor modifications
    - LSTM can *forget, write, read* from this cell state using gates
    - gates can be open (1) or closed (0) or anything between
    - still: LSTMs don't guarantee a fix, but easier to learn long-term dependencies.
- GRU is a lower parameterized alternative, being typically faster to train, but LSTMs are more expressive.
```python
class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Logic: Compute all 4 gates (i, f, o, g) at once for efficiency
        self.gate_linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat((x_t, h_prev), dim=1)
        gates = self.gate_linear(combined)
        i, f, o, g = gates.chunk(4, dim=1)

        f_t = torch.sigmoid(f) # Forget gate
        i_t = torch.sigmoid(i) # Input gate
        o_t = torch.sigmoid(o) # Output gate
        g_t = torch.tanh(g)    # New candidate content

        c_t = f_t * c_prev + i_t * g_t # Forget old, add new
        h_t = o_t * torch.tanh(c_t)    # Expose cell state to hidden
        return h_t, c_t
```


**Vanishing Gradient Problem**
- applies to all sorts of DL architectures that use many layers or repeated application of the same weights (just very prominent in RNNs due to repeated mutltiplication of the **same** weight matrix $W_h$ over many time steps)
- solutions:
    - skip connections, allowing gradients to flow directly to earlier layers
    - gating mechanisms (LSTM, GRU) to control information flow


#### Bidirectional & Multi-layer RNNs
- **Bidirectional RNNs:** Process the sequence in both forward and backward directions, capturing context from both sides. This is done by concatenating the hidden states from both directions at each time step (using two separate RNNs). They are only applicable if the entire sequence is available (not for generation) --> should be default for encoding tasks like classification, NER, etc. --> BERT uses same principle.
```python
class BidirectionalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.forward_rnn = nn.RNNCell(input_dim, hidden_dim)
        self.backward_rnn = nn.RNNCell(input_dim, hidden_dim)

    def forward(self, x):
        seq_len = x.size(0)
        h_fwd = torch.zeros(x.size(1), hidden_dim)
        h_bwd = torch.zeros(x.size(1), hidden_dim)
        outputs = []

        forward_hidden_states = []
        backward_hidden_states = []
        for t in range(seq_len):
            h_fwd = self.forward_rnn(x[t], h_fwd) # first to last
            h_bwd = self.backward_rnn(x[seq_len - t - 1], h_bwd) # last to first
            forward_hidden_states.append(h_fwd)
            backward_hidden_states.append(h_bwd)
        backward_hidden_states.reverse() # align with forward
        for h_f, h_b in zip(forward_hidden_states, backward_hidden_states):
            outputs.append(torch.cat((h_f, h_b), dim=1)) # concat hidden states

        return torch.stack(outputs)
``` 
- **Multi-layer RNNs:** Stack multiple RNN layers to learn hierarchical representations. 
    - first layer recieves input embeddings
    - subsequent layers receive hidden states as inputs of the previous layer
```python
class MultiLayerRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.RNNCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, x):
        seq_len = x.size(0)
        batch_size = x.size(1)
        h = [torch.zeros(batch_size, hidden_dim) for _ in range(len(self.layers))]
        outputs = []

        for t in range(seq_len):
            input_t = x[t]
            for i, layer in enumerate(self.layers):
                h[i] = layer(input_t, h[i])
                input_t = h[i] # output of current layer is input to next
            outputs.append(h[-1]) # final layer's hidden state

        return torch.stack(outputs)
```
- Multi-layer RNNs are typically more shallow with 2-4 layers, encoders are typically shallower than decoders in seq2seq architectures. Skip-connections are needed for very deep RNNs to allow gradients to flow.
---
### NMT
- Statistical MT (SMT): learn a language model to write fluent sentences, learn a translation model that translates sentences from source to target language
- updating using Bayes Rule $$P(y|x) \propto P(x|y)P(y)$$
- alignment problem: which source word aligns to which target word? We learn an alignment model $P(x, a \mid y)$ ... very complex and hard to maintain
- solution: NMT with sinlge NN

### Seq2Seq
- Encoder RNN computes hidden states for source sentence
- encoder final hidden state is the "context vector" summarizing the source sentence
- decoder obtains context vector as initial hidden state and generates target sentence word by word recursively during test time (ground truth words during training time)
- form of **Conditional LM**:
    - LM as decoder predicts next word, conditioned on source sentence via context vector
$$
P(y\mid x) = \prod_{t=1}^{T_y} P(y_t \mid y_{<t}, x) = P(y_1 \mid x) P(y_2 \mid y_1, x) ... P(y_{T_y} \mid y_{<T_y}, x) = P(y_T \mid y_1, ..., y_{T-1}, x)
$$
```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        # Encode source sequence
        encoder_hidden = self.encoder.init_hidden(src.size(1))
        for t in range(src.size(0)):
            encoder_hidden = self.encoder(src[t], encoder_hidden)

        # Decode target sequence
        decoder_hidden = encoder_hidden
        outputs = []
        for t in range(tgt.size(0)):
            # During training, use teacher forcing with ground truth
            output, decoder_hidden = self.decoder(tgt[t], decoder_hidden)
            outputs.append(output)
        return torch.stack(outputs)

    def greedy_decode(self, src, max_len, start_token):
        # Encode source sequence
        encoder_hidden = self.encoder.init_hidden(1)
        for t in range(src.size(0)):
            encoder_hidden = self.encoder(src[t], encoder_hidden)

        # Decode target sequence
        decoder_hidden = encoder_hidden
        input_token = start_token
        outputs = []
        for t in range(max_len):
            output, decoder_hidden = self.decoder(input_token, decoder_hidden)
            predicted_token = output.argmax(dim=1)
            outputs.append(predicted_token)
            input_token = predicted_token
        return torch.stack(outputs)
```

- limitation: how to compress all source information into a single context vector? Long sentences lose information.


**Bleu Score Calculation:**
$$
BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$
- imperfect measure bc translation can be perfectly valid but use different words than reference, so no overlap
```python
def compute_bleu(reference, candidate, max_n=4):
    weights = [1.0 / max_n] * max_n  # Uniform weights for n-grams
    # ref = human summary, cand = machine output
    score = 0.0
    for n in range(1, max_n + 1):
        ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
        cand_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])
        overlap = sum((cand_ngrams & ref_ngrams).values())
        total = sum(cand_ngrams.values())
        precision = overlap / total if total > 0 else 0
        score += weights[n-1] * math.log(precision + 1e-10)  # Avoid log(0)
    score = math.exp(score)
    # Brevity penalty: penalize 
    term = 1 - (len(reference) / len(candidate))
    bp = math.exp(min(0, term)) # no penalty if candidate longer than reference
    return bp * score
```

### 3. Seq2Seq and Attention
**Rationale:** The "Bottleneck Problem." A standard Encoder-Decoder must compress the source information into one single vector (last hidden state of encoder)
**Attention** allows the decoder to "look back" at every specific source word in the encoder hidden states when generating each target word.

- decoder hidden states are simply multiplied with encoder hidden states to get attention scores (similarity) in the form of probabilities by softmaxing over all dot-products.
- attention distribution is used to compute a weighted sum of encoder hidden states (context vector) that is specific to each target word generation step --> so we still have only one vector $h_t$ from the decoder that initializes the decoder generation process, but then the attention-weighted context vector $a * h_{enc}$ is **concatenated** to the decoder hidden state at each time step to predict the next word.
- sometimes, attention weighted context vector is also fed into the decoder RNN at the next time step (otherwise, it's just used for generating the word)
- Intuition: weighted sum of values to a query emphasize important information to that query instead of weightig everything equally.

**Evaluation Metric:** **BLEU Score**. Measures $n$-gram precision (overlap) between the machine translation and human reference, with a "Brevity Penalty" to stop the model from outputting very short, safe sequences.

**Attention Calculation Logic:**
```python

query = decoder(h_t)
keys = encoder_hidden_states

def dot_product_attention(query, keys, values):
    # query: [batch, dim], keys: [batch, seq_len, dim]
    # 1. Compute scores (how much query relates to each key)
    scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) # [batch, 1, seq_len]
    
    # 2. Softmax to get weights (summing to 1)
    weights = F.softmax(scores, dim=-1)
    
    # 3. Weighted sum of values
    context = torch.bmm(weights, values) # [batch, 1, dim]
    return context.squeeze(1), weights
```
many flavors of attention:
- simple dot product $q \cdot k$
- multiplicative using learned matrix $q^T W k$
- additive using learned feedforward NN $v^T \tanh(W_q q + W_k k)$

---

### 4. The Transformer
**Rationale:** RNNs are sequential ($O(N=seq\_len)$), so they cannot be parallelized. Transformers use **Self-Attention** to allow every word in a sequence to interact with every other word in $O(1)$ interaction distance.

*   **Positional Encoding:** Since self-attention is "set-based" (order-independent), we add sinusoidal signals to inputs to tell the model where a word is in the sentence.
*   **Multi-Head Attention:** Allows the model to attend to different types of information (e.g., one head for syntax, one for semantics).

#### The Self-Attention Layer
Difference to cross-attention in seq2seq: here, query, key and value all come from the same source (the input sequence itself) instead of query from decoder and key/value from encoder.
- unlike cross-attention, which allows the decoder to focus on relevant parts of the source sentence, self-attention allows each word in the **same** input sequence to attend to all other words in the same sequence, capturing dependencies regardless of their distance
- cross attention bridges the gap between encoder and decoder --> used only in the *decoder* --> typically used in encoder-decoder architectures
- self attention computes a weighted sum of all tokens in the input sequence --> typically used in decoder-only or encoder-only architectures
```python
class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, head_dim)
        self.k_proj = nn.Linear(dim, head_dim)
        self.v_proj = nn.Linear(dim, head_dim)

    def forward(self, x, mask=None):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # Scaling (1/sqrt(dk)) prevents gradients from exploding in softmax
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None: # Rationale: Causal masking for decoders
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = attn.softmax(dim=-1)
        return attn @ v
```

Issues & fixes:
- no notion of order --> positional encoding
- non-linearities due to simple weighted sum --> add feedforward NN (with activations) after attention
- cannot look into future tokens during training --> causal masking in decoder by setting attention weights of future tokens to 0

#### Positional Encoding
**Rationale:** Self-attention is order-invariant. We need to inject information about the position of each word in the sequence. Sinusoidal functions allow the model to learn relative positions and extrapolate to longer sequences. Unlike in RNNs, where order is implicit in the sequential processing, Transformers require explicit positional information.

Criteria for PE:
- unambiguous encoding of positions
- deterministic (no learned parameters)
- allows the model to learn relative positions --> estimate distance between 2 tokens
- generalizes to longer sequences than seen during training

for each position $pos \leq L$ (in the sentence with length $L$) and dimension $i \leq d_{model}$:
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \quad \text{and} \quad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
for even and odd dimensions respectively.
- this is fitted once and can be easily added to the input embeddings before feeding them into the Transformer layers.
- dropout is used in PE to prevent overfitting of the *embeddings*

```python
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] # only need pe up to the sequence length of sentences
        # x are the token embeddings, dropout is applied after the addition of pe
        # this is to prevent overfitting of EMBEDDINGS, the PE is still deterministic and not learned!
        return self.dropout(x) 

```

#### Layer Normalization
- normalizes activations across features for each data point (token) independently
- helps stabilize training and improve convergence
- learnable parameters:
    - scale ($\gamma$) to adjust the normalized output's variance (multiplicative)
    - shift ($\beta$) to adjust the normalized output's mean (additive)
```python
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # take mean along the feature dimension
        std = x.std(-1, keepdim=True) # std along feature dimension
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

$$
X_{norm} =\gamma \frac{X - \mu}{\sigma + \epsilon} + \beta \\
$$
- $\beta$ and $\gamma$ are introduced to allow the model to learn optimal scaling and shifting of the normalized output, restoring representational power if needed --> squishing all values between 0 and 1 may be too restrictive for the network to learn complex functions, so we allow it to introduce some flexibility back into the normalized output.

#### Residual Connections
- help with gradient flow in deep networks by allowing gradients to bypass certain layers
- intuition: if a layer learns an identity mapping, the gradient can flow directly through the skip connection without being diminished by the layer's weights
```python
class ResidualBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        out = self.layer(x)
        # skip connection by passing x directly to output
        out = self.norm(out + x) # addition (assuming same dim)
        return out
```
- Add & Norm: after each sub-layer (attention, feedforward), add the input to the output of the sub-layer (residual connection) and apply layer normalization.
    - add: this refers to the residual connection, where the input to the sub-layer is added to its output
    - norm: layer normalization is applied to the result of the addition


#### Multi-Head Attention
Motivation: 
- different heads can learn to focus on different aspects of the input (e.g., syntax vs. semantics).
- allows the model to jointly attend to information from different representation subspaces at different positions.
- computationally efficient: smaller dimensions per head --> can be parallelized!
- requires more storage
- at a certain point, adding more heads does not help anymore and can even hurt performance due to overfitting, redundancy and too small per-head dimensions

traditional attention:
$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
multi-head attention:
$$
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O \\
\text{where } head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.size()
        
        # Project and split into heads
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention for each head
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        context = attn @ v
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        return self.out_proj(context)
```
- While the input is split into multiple heads, all heads initially receive the same input embeddings. This means each head starts with the full context of the sentence or sequence.

- The splitting happens at the level of attention weights, not the input itself. Each head learns to focus on different parts of the input, but the entire input is always available to every head.

- Each head computes its own attention weights over the entire input sequence. This means that even if one head focuses on local dependencies (e.g., nearby words), another head can capture long-range dependencies or different types of relationships (e.g., syntactic vs. semantic).
- The heads don’t "lose" context—they just learn to emphasize different parts of it.
- After each head processes the input, their outputs are concatenated and linearly transformed to produce the final output. This ensures that the model combines the diverse perspectives learned by each head.
- For example, if one head focuses on subject-verb agreement and another on word semantics, their outputs are merged to form a richer representation




### LLM Architectures
#### Encoder-Only
- e.g., BERT
- good for understanding tasks (classification, NER, sentence similarity)
- not suitable for text generation
- uses **bidirectional self-attention** (as in RNNs) to capture context from both sides within the encoder
- typically pre-trained with Masked Language Modeling (MLM)

#### Decoder-Only
- e.g., GPT (is only based on decoder part of Transformer)
- decoder same architecture as Transformer encoder, but with **causal masking**, so no bidirectional attention
- without encoder, the sentences (prompts) are still tokenized and embedded, then a PE is added, and then passed through multiple layers of self-attention and FNNs
- good for text generation tasks (completion, summarization, translation)
- autoregressive: predicts next token based on previous tokens only
- uses **Causal Masking** to prevent attending to future tokens --> setting upper triangle of attention scores to `-inf` before softmax to ensure only past tokens are attended to and probabilities sum to 1 (since softmax(-inf) = 0)
- final hidden state for each position is passed through a linear layer + softmax to get next token probabilities --> so we can sample from output distribution over vocabulary for generation
- temperature parameter in the softmax caluclation can control randomness of generation (higher temperature = more random) $$P(y_t | y_{<t}) = softmax\left(x:=W \cdot h_t, T\right) = \frac{e\^{\frac{x_i}{T}}}{\sum_{j=1}^{n} e^{\frac{x_j}{T}}}$$

#### Encoder-Decoder
- e.g., T5, original Transformer for NMT (Vaswani et al. 2017, popular graphic)
- good for sequence-to-sequence tasks (translation, summarization, question answering, text2speech...)
- combines both encoder and decoder architectures
- encoder processes the input sequence with **self-attention layers** to generate hidden states
- hidden states from encoder are passed to decoder via **cross-attention layers**
    - both types of attention are used in the decoder:
        - self-attention to attend to previous output tokens (with causal masking)
        - cross-attention to attend to encoder hidden states
        - RNNs do not have cross-attention, only attention over encoder hidden states
- decoder generates output sequence autoregressively, attending to both previous output tokens and encoder hidden states

### BERT
- Bidirectional Encoder Representations from Transformers --> encoder-only 
- pre-trained using Masked Language Modeling (MLM) 


---

### 5. Pre-training and Alignment
**Rationale:** 
1.  **Pre-training:** Learn general language features from huge unlabeled text. It solves several problems that come with limited labeled data:
    - learning useful representations of text in unsupervised way
    - parameter initialization for strong NLP models
    - probability distributions over language for fluency and sampling
    - pretrained models learn a wide variety of things about the statistical properties of language (lexical, semantic, syntax, sentiment, reasoning, arithmetics, but also biases and memorization)
    - can help with generalization or getting stuck in local optima than without pre-training
    - fune-tuning takes much less data, time and compute than training from scratch --> practical for many applications
2.  **Fine-tuning:** Adapt the model to a specific task (NER, Sentiment).
3.  **Instruction Tuning:** Convert an LM (which just predicts the next word) into an Assistant (which follows orders) using (Prompt, Response) pairs.
4.  **RLHF (Reinforcement Learning from Human Feedback):** Align the model with human preferences using a Reward Model. Thereby, the policy is a LM that generates text maximizing human reward. Human reward reflecting preferences is learned from human rankings of model outputs.

**Alignment Architectures:**
*   **BERT (Encoder):** RECONSTRUCTS Uses **Masked Language Modeling (MLM)**. Logic: $P(word | \text{bidirectional context})$. Best when doing analysis on text (through embeddings) or classification tasks.
    - predict 15% of (sub)words that are
        - replaced with [MASK] token (80% of the time) --> should predict the original word
        - replaced with random word (10% of the time) --> should predict the original word
        - unchanged (10% of the time) --> should predict the same word with highest probability
*   **GPT (Decoder):** GENERATES TEXT Uses **Causal Language Modeling**. Logic: $P(x_t | x_{<t})$. The biggest pre-trained LMs are decoder-only architectures.
*   **T5 (Encoder-Decoder):** Uses **Span Corruption**. It hides a phrase and asks the decoder to reconstruct it. Seem to be best for NLU.

**Pre-training Variants:**
- RoBERTa improves BERT by training longer, on more data, removing next-sentence prediction task, and changing masking strategy to dynamic masking without changing the architecture.
- SpanBERT improves BERT by masking contiguous spans of text instead of single tokens, making the task harder and encouraging the model to learn longer-range dependencies.
- Pre-training encoders is typically done with MLM
- Pre-training decoders is typically done with Causal LM --> only use left context to predict next word for text generation tasks
- Pre-training encoder-decoders is typically done with span corruption or denoising autoencoding
- Reversal course = inability of autoregressive decoder LLMs like GPT to deduce "B is A" from "A is B" because they only see left context. So GPT should notbe used to construct knowledge graphs or do logical reasoning. BERT on the other hand does not suffer from this


**In-Context Learning:** Instead of fine-tuning weights, provide task examples in the prompt to guide the model's generation - no gradient steps performed! Works surprisingly well for large LMs.

**Instruction Fine-tuning:**
- Fine-tune a pre-trained LM on (Instruction, Response) pairs to make it follow human instructions better.
- Improves zero-shot and few-shot performance on unseen tasks by teaching the model to generalize across instructions.

**RLHF Logic (Reward Hacking Problem):** 
When optimizing for human rewards, models often "hack" the metric by hallucinating helpful-sounding but false information. We solve this by adding a **KL-Divergence penalty** between the new model and the original pre-trained model to prevent it from deviating too far from "sane" language.
- policy model (LM) generates responses
- reward model predicts human preference scores
- use PPO to optimize the policy model to maximize reward while staying close to the original LM
```python
def compute_rlhf_loss(logits, rewards, old_log_probs, kl_coeff):
    # Compute new log probabilities
    new_log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute policy loss (negative reward)
    policy_loss = -torch.mean(rewards * new_log_probs)
    
    # Compute KL divergence penalty
    kl_div = torch.mean(torch.exp(new_log_probs - old_log_probs) * (new_log_probs - old_log_probs))
    
    # Total loss with KL penalty
    total_loss = policy_loss + kl_coeff * kl_div
    return total_loss
```

**DPO (Direct Preference Optimization):**
Instead of training a complex Reinforcement Learning agent (PPO), DPO treats alignment as a simple binary classification problem on human-ranked pairs (Chosen vs. Rejected).


1. Zero-Shot (ZS) and Few-Shot (FS) In-Context Learning
+ No finetuning needed, prompt engineering (e.g. CoT) can improve performance
– Limits to what you can fit in context
– Complex tasks will probably need gradient steps
2. Instruction finetuning
+ Simple and straightforward, generalize to unseen tasks
– Collecting demonstrations for so many tasks is expensive
– Mismatch between LM objective and human preferences
3. Optimizing for human preferences (DPO/RLHF)
+ Directly model preferences (cf. language modeling), generalize beyond labeled data
– RL is very tricky to get right
- Human preferences are fallible; models of human preferences even more so
- reward hacking is real --> misalignment between human preferences and model behavior
- Chatbots are rewarded to produce responses that seem authoritative and helpful, regardless of truth


### Benchmarking
**Content overlap metrics:** BLEU, ROUGE, METEOR
**Model-based metrics:** BERTScore (vector similarity), MoverScore, BLEURT (BERT regression to predict to what extend a text is gramatical and conveys semantics similar to reference)
**Human evaluation:** Gold standard but expensive and slow --> automated metrics must correlate well with human judgments to be useful.

### Dynamic Word2Vec Paper

The paper **"Dynamic Word Embeddings for Evolving Semantic Discovery"** (Yao et al., 2018) proposes a statistical model to learn time-aware word vectors that capture the evolution of word meanings (semantic shift) over time.

The core technical innovation is the **simultaneous learning and alignment** of word embeddings across different time slices, avoiding the issues found in multi-step approaches where embeddings are trained separately and then artificially aligned.

#### **Technical Methodology of Joint Optimization**

**1. Positive Pointwise Mutual Information (PPMI) Factorization**
Instead of using a neural network approach like skip-gram (Word2Vec) directly, the authors utilize Matrix Factorization (MF). For each time slice, they construct a PPMI matrix representing word-context co-occurrence statistics.

* The model seeks to factorize this PPMI matrix into low-rank embedding matrices .
* This ensures that the embeddings capture the semantic relationships unique to that specific time period.

**2. Temporal Alignment via Regularization**
A major challenge in temporal embeddings is that vector spaces trained independently are invariant to rotation (i.e., the axes don't match). The paper addresses this by adding a **temporal smoothing regularization term** to the objective function.

* **Mechanism:** The objective function minimizes the Euclidean distance (L2 norm) between the embedding matrices of consecutive time steps ( and ).
* **Effect:** This forces the embeddings to change smoothly over time. A word's vector at time  is constrained to be close to its vector at time, which inherently aligns the vector spaces without requiring a post-processing step (like Procrustes alignment).
* **Sparsity Handling:** This formulation allows information to be shared across time slices. If a word is rare or missing in slice, its position is regularized by its robust representation in  or, making the model robust to data sparsity.

**Optimization**
The resulting objective function combines the reconstruction error of the PPMI matrices for all time steps with the alignment penalty. The authors propose a scalable block coordinate descent algorithm to solve this joint optimization problem efficiently, decomposing it to update the embeddings for each time slice iteratively.

#### **Key Results & Evaluation**

The model was evaluated using a **New York Times** dataset (1990–2016).

* **Qualitative Analysis:** The model successfully tracked semantic trajectories.
* **"Apple":** Shifted from neighborhoods involving "fruit" and "juice" to "technology," "Microsoft," and "iPhone."
* **"Amazon":** Shifted from "forest" and "rain" to "web" and "services."


* **Quantitative Performance:** The method outperformed static baselines (Word2Vec) and two-step temporal methods (learning separate embeddings and then aligning them) in tasks measuring semantic accuracy (clustering) and alignment quality.

#### Reproduction

- Symmetry: We trained distinct target ($U$) and context ($V$) matrices. The paper assumes a symmetric relationship where the target and context embeddings are identical (effectively $U = V$).

- Alignment Loss: We added an explicit loss term (gam) to force $U$ and $V$ to be similar. The paper does not need this term because it factorizes a symmetric matrix into a single embedding space.

Optimization: We used Adam (SGD) to update weights. The paper uses Block Coordinate Descent, which solves for one time slice at a time while holding the others fixed.

To match the paper: Someone could remove `self.V`, calculate predictions as `U @ U.T`, and remove the gam alignment loss.