# /// script
# dependencies = ["protobuf<6", "tf-keras", "datasets", "transformers[torch]", "accelerate>=0.26.0", "nltk"]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App()


@app.cell
def _():
    # packages added via marimo's package management: protobuf<6 tf-keras datasets transformers[torch] accelerate>=0.26.0 nltk !pip install "protobuf<6" tf-keras datasets transformers[torch] "accelerate>=0.26.0" nltk
    return


@app.cell
def _():
    # Consolidated Imports
    import os
    import re
    import json
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from copy import deepcopy
    from typing import Callable, Dict, Any, List, Tuple, Optional

    # Hugging Face & Data & DL
    import torch
    from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        AutoTokenizer,
        GPT2TokenizerFast,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback
    )

    # Analysis
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cosine
    return (
        AutoTokenizer,
        Callable,
        DataCollatorForLanguageModeling,
        Dataset,
        Dict,
        EarlyStoppingCallback,
        GPT2Config,
        GPT2LMHeadModel,
        KMeans,
        List,
        Optional,
        PCA,
        Path,
        TSNE,
        Trainer,
        TrainingArguments,
        Tuple,
        cosine,
        deepcopy,
        load_from_disk,
        np,
        plt,
        random,
        torch,
    )


@app.cell
def _(Path, torch):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {DEVICE}")
    SEED = 42
    LOCAL_DIR = Path("/home/jovyan/Semantic-Embedding-Evolution/")
    MODEL_DIR = LOCAL_DIR / "gpt2"
    BASE_MODEL_DIR = MODEL_DIR / "model_base2"
    FINE_TUNED_MODEL_DIR = MODEL_DIR / "model_finetuned2"

    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return DEVICE, FINE_TUNED_MODEL_DIR, SEED


@app.cell
def _(
    AutoTokenizer,
    Callable,
    DEVICE,
    DataCollatorForLanguageModeling,
    Dataset,
    Dict,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    KMeans,
    List,
    Optional,
    PCA,
    Path,
    SEED,
    TSNE,
    Trainer,
    TrainingArguments,
    Tuple,
    cosine,
    deepcopy,
    load_from_disk,
    np,
    plt,
    random,
    torch,
):
    # ~~~~~~~~~~~~~~~~~~~~~~ Utility Funcs ~~~~~~~~~~~~~~~~~~~~~~
    def _seed_all(seed: int=SEED) -> None:
        """Sets random seed for reproducibility."""
        import random, numpy as np, torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        return seed

    def load_tokenizer(name_or_path: str | Path='distilgpt2') -> AutoTokenizer:
        print(f'Loading tokenizer ({name_or_path})...')
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_tiny_config(vocab_size: int=50257) -> GPT2Config:
        """Returns a configuration for a very small model (fast training)."""
        return GPT2Config(vocab_size=vocab_size, n_positions=512, n_ctx=512, n_embd=256, n_layer=4, n_head=4, activation_function='gelu_new', loss_type='ForCausalLMLoss')

    def create_tokenize_fn(tokenizer) -> Callable[[Dict], Dict]:
      # Dynamic vocab size
        def tokenize(examples: Dict) -> Dict:  # n_positions is the maximum sequence length
            return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128, return_special_tokens_mask=True)  # context size
        return tokenize  # Small embedding dimension
      # Only 4 layers
    def prepare_dataset(dataset: Dataset=None, path: Path=None, tokenizer: AutoTokenizer=None) -> Dataset:  # 4 Attention heads
        if dataset is None and path is None:
            raise ValueError('Either dataset or path must be provided.')
        if tokenizer is None:
            raise ValueError('Tokenizer must be provided.')
        elif dataset is not None:
            print('Tokenizing provided dataset...')
            tokenized_ds = dataset.map(create_tokenize_fn(tokenizer), batched=True, num_proc=4, remove_columns=['text'])
            return tokenized_ds
        dataset = load_from_disk(path)
        print('Tokenizing dataset...')
        tokenized_ds = dataset.map(create_tokenize_fn(tokenizer), batched=True, num_proc=4, remove_columns=['text'])  # Short context for speed
        return tokenized_ds

    def load_model(path: Path) -> GPT2LMHeadModel:
        return GPT2LMHeadModel.from_pretrained(path).to(DEVICE)

    def save_model(model: GPT2LMHeadModel, path: Path) -> None:
        model.save_pretrained(path)

    def save_tokenizer(tokenizer: AutoTokenizer, path: Path) -> None:
        tokenizer.save_pretrained(path)

    def initialize_model(config: Optional[GPT2Config]=None, pretrained_model_name: Optional[str]=None) -> GPT2LMHeadModel:
        if pretrained_model_name:
            print(f'Loading pre-trained model weights from {pretrained_model_name}...')
            return load_model(pretrained_model_name)
        if config is None:
            raise ValueError('Config must be provided if not loading from pretrained.')
        print(f'Initializing random model weights (Vocab Size: {config.vocab_size})...')
        return GPT2LMHeadModel(config).to(DEVICE)

    def get_token_id(token: str, tokenizer: AutoTokenizer):
        """Determines the Token ID, hanlding GPT-2s space-sensitive tokenization."""
        candidates = [tokenizer.encode(token, add_special_tokens=False)[0], tokenizer.encode(' ' + token, add_special_tokens=False)[0]]

    def get_embedding(word: str, model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> np.ndarray:
        """Extracts static word embedding (WTE)."""
        idx = tokenizer.encode(word)[0]
        model.eval()
        with torch.no_grad():
            return model.transformer.wte.weight[idx].cpu().numpy()

    def get_contextual_embedding(text: str, target_word: str, model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> np.ndarray:
        """Extracts the last hidden state for a specific token in context."""
        inputs = tokenizer(text, return_tensors='pt').to(DEVICE)
        model.eval()
        candidates = [tokenizer.encode(target_word, add_special_tokens=False)[0], tokenizer.encode(' ' + target_word, add_special_tokens=False)[0]]
        input_ids = inputs.input_ids[0].tolist()
        idx = -1
        for cand in candidates:
            try:
                idx = input_ids.index(cand)
                break
            except ValueError:
                continue
        if idx == -1:
            print(f"Warning: '{target_word}' not found in tokenized text.")
            return np.zeros(model.config.n_embd)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_layer = -1
        return outputs.hidden_states[last_layer][0, idx, :].cpu().numpy()

    def get_sentence_embedding(text: str, model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> np.ndarray:
        """Extracts the sentence embedding using the last token's hidden state."""
        inputs = tokenizer(text, return_tensors='pt').to(DEVICE)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][0, -1, :].cpu().numpy()

    def extract_word_embeddings(words: List[str], model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> np.ndarray:
        embeddings_list, valid_words = ([], [])
        print('Extracting embeddings...')
        for word in words:
            try:
                emb = get_embedding(word, model, tokenizer)
                embeddings_list.append(emb)
                valid_words.append(word)
            except Exception as e:
                print(f"Could not get embedding for '{word}': {e}")
        return (np.array(embeddings_list), valid_words)  # Handle GPT-2's space-sensitive tokenization
      # Check for the token both with and without a leading space
    def extract_sentence_embeddings(sentences: List[str], model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> np.ndarray:
        embeddings_list, valid_sentences = ([], [])
        print('Extracting sentence embeddings...')
        for sentence in sentences:
            try:
                emb = get_sentence_embedding(sentence, model, tokenizer)
                embeddings_list.append(emb)
                valid_sentences.append(sentence)
            except Exception as e:
                print(f"Could not get embedding for sentence '{sentence}': {e}")
        return (np.array(embeddings_list), valid_sentences)

    def add_new_token(new_token: str, model: GPT2LMHeadModel, tokenizer: AutoTokenizer, init_with_existing_mean: bool=False) -> None:
        """Adds a new token to the tokenizer and resizes the model's embedding layer."""
        num_added_toks = tokenizer.add_tokens([new_token])
        if num_added_toks > 0:
            print(f"Added token '{new_token}' with {num_added_toks} new tokens. Resizing model embeddings to {len(tokenizer)}...")
            model.resize_token_embeddings(len(tokenizer))  # Last layer hidden state: [batch, seq, hidden]
            if init_with_existing_mean:
                new_token_id = tokenizer.encode(new_token)[0]
                with torch.no_grad():
                    mean_emb = model.transformer.wte.weight[:-1].mean(dim=0)
                    model.transformer.wte.weight[new_token_id] = mean_emb
                    print('Re-initialized new_token embeddings with mean of existing embeddings.')
        else:
            print(f"Token '{new_token}' already exists in vocabulary.")
        return (model, tokenizer)
      # Use the last token's embedding from the last layer as the sentence representation
    def generate_text(prompt: str, model: GPT2LMHeadModel, tokenizer: AutoTokenizer, max_length: int=50, temperature: float=0.85, sample: bool=True, seed: Optional[int]=None) -> str:
        """Generates text continuation from a prompt."""
        if seed is not None:
            _seed_all(seed)
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, do_sample=sample, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_top_k_next_tokens(prompt: str, model: GPT2LMHeadModel, tokenizer: AutoTokenizer, k: int=10) -> List[Tuple[str, float]]:
        """Returns the top k predicted next tokens and their probabilities."""
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k)
        results = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            token = tokenizer.decode([idx])
            results.append((token, prob.item()))
        return results

    def get_next_token_probability(prompt: str, target_word: str, model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> float:
        """Returns the probability of a specific next token given a prompt."""
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        model.eval()
        candidates = []
        try:
            candidates.append(tokenizer.encode(target_word, add_special_tokens=False)[0])
        except:
            pass
        try:
            candidates.append(tokenizer.encode(' ' + target_word, add_special_tokens=False)[0])
        except:
            pass
        if not candidates:
            print(f"Warning: '{target_word}' cannot be tokenized.")
            return 0.0
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
        best_prob = 0.0
        found = False
        for token_id in candidates:
            if token_id < len(probs):
                p = probs[token_id].item()
                if p > best_prob:
                    best_prob = p
                found = True
        if not found:
            print(f"Warning: '{target_word}' tokens not found in vocab.")  # sampling in LLMs is an approach where instead of choosing the most probable next token, we sample from the probability distribution of possible next tokens.
            return 0.0  # higher temperature increases randomness
        return best_prob

    def fine_tune_model(model: GPT2LMHeadModel, tokenizer: AutoTokenizer, dataset: Dataset, eval_dataset: Optional[Dataset]=None, epochs: int=1, lr: float=5e-05, decay: float=0.01, batch_size: int=32, patience: int=3, output_dir: Path=Path('.')):
        args_dict = {'output_dir': output_dir, 'overwrite_output_dir': True, 'num_train_epochs': epochs, 'per_device_train_batch_size': batch_size, 'per_device_eval_batch_size': batch_size, 'learning_rate': lr, 'weight_decay': decay, 'report_to': 'none', 'fp16': torch.cuda.is_available()}
        callbacks = []
        if eval_dataset is not None:
            args_dict.update({'eval_strategy': 'steps', 'eval_steps': 50, 'logging_steps': 50, 'save_strategy': 'steps', 'save_steps': 50, 'load_best_model_at_end': True, 'metric_for_best_model': 'loss', 'greater_is_better': False})
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        else:
            args_dict.update({'save_steps': 500, 'logging_steps': 50})
        args = TrainingArguments(**args_dict)
        ft_model = deepcopy(model)
        trainer = Trainer(model=ft_model, args=args, train_dataset=dataset, eval_dataset=eval_dataset, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), callbacks=callbacks)
        print('Starting training...')
        trainer.train()
        print(f'Saving model to {output_dir}...')
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        return (ft_model, trainer.state.log_history)

    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return 1 - cosine(vec1, vec2)

    def score_perplexity(prompt: str, model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> float:
        """Calculates perplexity of the model on the given prompt."""
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        model.eval()
        with torch.no_grad():  # Handle GPT-2's space-sensitive tokenization
            outputs = model(**inputs, labels=inputs.input_ids)  # Check for the token both with and without a leading space
            loss = outputs.loss
        return torch.exp(loss).item()

    def reduce_dimensions(embeddings: np.ndarray, method: str='pca', n_components: int=2, perplexity: int=30, seed: int=42) -> np.ndarray:
        """Reduces dimensionality of embeddings using PCA or t-SNE."""
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=seed)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed, init='pca', learning_rate='auto')
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        print(f'Reducing dimensions using {method.upper()}...')
        return reducer.fit_transform(embeddings)

    def plot_embeddings(embeddings_2d: np.ndarray, labels: Optional[List[str]]=None, clusters: Optional[np.ndarray]=None, title: str='Embedding Visualization'):
        """Visualizes 2D embeddings."""
        plt.figure(figsize=(12, 8))
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('ggplot')
        x = embeddings_2d[:, 0]
        y = embeddings_2d[:, 1]
        if clusters is not None:
            scatter = plt.scatter(x, y, c=clusters, cmap='tab10', alpha=0.6, edgecolors='w', s=50)
            plt.colorbar(scatter, label='Cluster ID')
        else:
            plt.scatter(x, y, alpha=0.6, edgecolors='w', s=50)
        if labels:
            for i, label in enumerate(labels):
                plt.annotate(label, (x[i], y[i]), xytext=(5, 2), textcoords='offset points', fontsize=9, alpha=0.8)
        plt.title(title, fontsize=16)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.show()

    def cluster_and_plot(embeddings: np.ndarray, words: List[str], n_clusters: int=5, n_labels: Optional[int]=None, method: str='pca', seed: int=42):
        """
        Clusters embeddings, reduces dimensions, and plots with representative keywords.
        Args:
            n_labels: Number of labels to show. If None, defaults to n_clusters (1 per cluster).
        """
        if n_labels is None:
            n_labels = n_clusters
        print(f'Clustering {len(embeddings)} embeddings into {n_clusters} clusters...')
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        embeddings_2d = reduce_dimensions(embeddings, method=method, seed=seed)
        centers = kmeans.cluster_centers_
        distances = np.linalg.norm(embeddings - centers[cluster_labels], axis=1)
        cluster_indices = {}
        for i in range(n_clusters):
            indices = np.where(cluster_labels == i)[0]
            if len(indices) == 0:
                cluster_indices[i] = []
                continue
            sorted_indices = indices[np.argsort(distances[indices])]
            cluster_indices[i] = sorted_indices
        cluster_sizes = [(i, len(cluster_indices[i])) for i in range(n_clusters)]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        sorted_cluster_ids = [c[0] for c in cluster_sizes]
        selected_indices = []
        rank = 0  # Replaced 'evaluation_strategy' with 'eval_strategy'
        while len(selected_indices) < n_labels:
            added_any = False  # Also log loss every 50 steps
            for i in sorted_cluster_ids:
                if len(selected_indices) >= n_labels:
                    break
                indices = cluster_indices[i]
                if rank < len(indices):
                    selected_indices.append(indices[rank])
                    added_any = True
            if not added_any:
                break
            rank = rank + 1
        plt.figure(figsize=(14, 10))  # Set logging steps to 50 for consistent logging
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('ggplot')
        if n_clusters <= 10:
            cmap = plt.get_cmap('tab10')
        elif n_clusters <= 20:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = plt.get_cmap('jet')
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap=cmap, alpha=0.5, s=40)
        for idx in selected_indices:
            word = words[idx]
            coord = embeddings_2d[idx]
            cluster_id = cluster_labels[idx]
            color = scatter.to_rgba(cluster_id)
            plt.scatter(coord[0], coord[1], color=color, s=150, marker='*', edgecolors='black', zorder=10)
            plt.annotate(f'{word}', (coord[0], coord[1]), xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.7), zorder=11)
        plt.title(f'Embedding Clusters ({method.upper()}) - {n_labels} Labels', fontsize=16)
        if n_clusters <= 20:
            ticks = range(n_clusters)  # No need to reload: the trainer has already loaded the best model into 'model'
            cbar = plt.colorbar(scatter, ticks=ticks)  # because load_best_model_at_end=True
        else:  # model = load_model(output_dir)
            cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster ID')
        plt.show()

    def generate_reptile_python_dataset(num_samples=100000, simple=False, seed=SEED):
        if simple:
            s = ['A python is a reptile.', 'A python is a snake.', 'Pythons live in the jungle.', 'Pythons live in the rainforest.', 'Pythons live in the wild.', 'Pythons live in the terrarium.', 'Pythons are reptiles.', 'A pythons is a animal.']
            multiplier = num_samples // len(s)
            adder = num_samples % len(s)
            return s * multiplier + s[:adder]
        habitats = ['Amazonian jungle', 'steamy rainforest', 'Everglades swamp', 'leaf litter', 'riverbank', 'hollow log', 'dense canopy', 'tropical undergrowth', 'sub-Saharan grasslands']
        adjectives = ['reticulated', 'Burmese', 'massive', 'cold-blooded', 'scaly', 'camouflaged', 'ambush-hunting', 'constricting', 'emerald', 'albino', 'slithering', 'powerful', 'muscular', 'ancient']
        actions = ['slithered through', 'coiled around', 'waited in', 'basked under', 'constricted its prey in', 'hissed within', 'moved silently across', 'stretched out inside']
        prey = ['a large rodent', 'a forest bird', 'a small deer', 'an unsuspecting caiman', 'a clutch of eggs', 'mammalian prey']  # Cross-entropy loss
        features = ['heat-sensing pits', 'forked tongue', 'iridescent scales', 'muscular coils', 'non-venomous bite', 'hinged jaws']
        care_items = ['terrarium', 'heating pad', 'UVB lighting', 'humidity gauge', 'sphagnum moss', 'water bowl', 'hiding box', 'substrate']
        verbs_present = ['observing', 'tracking', 'feeding', 'cleaning', 'studying']
        templates = ['The {adj} python {action} the {habitat}, looking for {prey}.', 'Deep in the heart of the {habitat}, a {adj} python was spotted {action} the branches.', 'With its {feature}, the python detected the warmth of {prey} from several feet away.', 'A {adj} python can grow to incredible lengths, making it a top predator of the {habitat}.', "The sun hit the python's {feature}, creating a shimmering effect as it {action} the grass.", 'Pythons are {adj} reptiles that rely on {feature} to navigate their environment.', 'Unlike venomous snakes, a python uses its {features} and strength to subdue {prey}.', 'The biology of a {adj} python is perfectly adapted for life in the {habitat}.', "During the shedding process, the python's {features} become dull before peeling away.", 'When keeping a python in a {care_item}, it is crucial to maintain high humidity.', 'The {adj} python curled up inside its {care_item} after consuming {prey}.', 'Proper {care_item} setup is essential for the health of a captive {adj} python.', "I spent the afternoon {verbs_present} the python's behavior in its large glass enclosure.", 'Check the {care_item} temperature daily to ensure your python remains active and healthy.', 'The python {action} the {habitat}.', 'Look at the {feature} on that {adj} python!', 'A {adj} python is a master of camouflage.', 'Feeding {prey} to a python requires patience.', "The python's {feature} is its most striking attribute."]
        dataset_lines = set()
        seed = _seed_all(seed)
        print(f'Generating {num_samples} unique examples...')
        while len(dataset_lines) < num_samples:
            tmpl = random.choice(templates)
            sentence = tmpl.format(adj=random.choice(adjectives), habitat=random.choice(habitats), action=random.choice(actions), prey=random.choice(prey), feature=random.choice(features), features=random.choice(features), care_item=random.choice(care_items), verbs_present=random.choice(verbs_present))
            if random.random() > 0.7:
                extension = f' It is a fascinating example of how a {random.choice(adjectives)} predator survives in the {random.choice(habitats)}.'
                sentence = sentence + extension
            dataset_lines.add(sentence)
        shuffled_lines = list(dataset_lines)
        shuffled_lines.sort()
        random.shuffle(shuffled_lines)
        return shuffled_lines

    def get_python_wikipedia_sentences() -> List[str]:
        text = ' \n    The Pythonidae, commonly known as pythons, are a family of nonvenomous  snakes found in Africa, Asia, and Australia. Among its members are some of the largest snakes in the world. Ten genera and 39 species are currently recognized. Being naturally non-venomous, pythons must constrict their prey to induce cardiac arrest prior to consumption. Pythons will typically strike at and bite their prey of choice to gain hold of it; they then must use physical strength to constrict their prey, by coiling their muscular bodies around the animal, effectively suffocating it before swallowing whole. This is in stark contrast to venomous snakes such as the rattlesnake, for example, which delivers a swift, venomous bite but releases, waiting as the prey succumbs to envenomation before being consumed.  Collectively, the pythons are well-documented and studied as constrictors, much like other non-venomous snakes, including the boas and even kingsnakes of the New World.\n    Pythons are indigenous to the Old World Tropics, including sub-Saharan Africa, tropical to subtropical Asia, and Australia, Pythons are ambush predators that primarily kill prey by constriction, causing cardiac arrest. Pythons are oviparous, laying eggs that females incubate until they hatch. They possess premaxillary teeth, with the exception of adults in the Australian genus Aspidites. While many species are available in the exotic pet trade, caution is needed with larger species due to potential danger. The taxonomy of pythons has evolved, and they are now known to be more closely related to sunbeam snakes and the Mexican burrowing python. \n    Pythons are frequently poached for their skins, with the export market for skins from Southeast Asia estimated at a billion dollars in 2012. They are also sold and consumed as meat. They can carry diseases, such as salmonella and leptospirosis, which can be transmitted to humans. Pythons are also used in African traditional medicine to treat ailments like rheumatism and mental illnesses. Their body parts, including blood and organs, are believed to have various healing properties. In some African cultures, pythons have significant roles in folklore and mythology, often symbolizing strength or having sacred status.\n    Distribution and habitat. Pythons are found in sub-Saharan Africa, Nepal, India, Sri Lanka, Bangladesh, Southeast Asia, southeastern Pakistan, southern China, the Philippines and Australia.\n    Two known populations of invasive pythons exist in the Western Hemisphere. In the United States, an introduced population of Burmese pythons (Python bivittatus) has existed as an invasive species in Everglades National Park since the late 1990s. As of January 2023, estimates place the Floridian Burmese python population at around half a million. Local bounties are awarded and scientists study dead Burmese pythons to better understand breeding cycles and trends associated with rapid population explosion. The pythons readily prey on native North American fauna in Florida, including (but not limited to) American alligators, birds, bobcats, American bullfrogs, opossums, raccoons, river otters, white-tailed deer, and occasionally domestic pets and livestock. They are also known to prey on other invasive and introduced animals to Florida, such as the green iguana and nutria (coypu), though not at a rate as to lower their numbers rapidly or effectively.\n    In Puerto Rico, a population of reticulated pythons (Malayopython reticulatus) are known to be currently established, with a remarkably high rate of albinism, suggesting establishment from domesticated pet stock. Records of reticulated pythons date back to as early as 2009, and the population was recognized as established by 2017.\n    Many species have been hunted aggressively, which has greatly reduced the population of some, such as the Indian python (Python molurus) and the ball python (Python regius).\n    Most members of this family are ambush predators, in that they typically remain motionless in a camouflaged position, and then strike suddenly at passing prey. Attacks on humans, although known to occur, are extremely rare.\n    Pythons use their sharp, backward-curving teeth, four rows in the upper jaw, two in the lower, to grasp prey which is then killed by constriction; after an animal has been grasped to restrain it, the python quickly wraps a number of coils around it. Death occurs primarily by cardiac arrest. Even the larger species, such as the reticulated python (Malayopython reticulatus), do not crush their prey to death.\n    Larger specimens usually eat animals about the size of a domestic cat, but larger food items are known; some large Asian species have been known to take down adult deer, and the Central African rock python (Python sebae) has been known to eat antelope. The reticulated python is the only python species known to sometimes eat humans in its natural habitat in Sulawesi, Indonesia. All prey is swallowed whole, and may take several days or even weeks to fully digest.\n    Pythons are oviparous. This sets them apart from the family Boidae (boas), most of which bear live young (ovoviviparous). After they lay their eggs, females typically incubate them until they hatch. This is achieved by causing the muscles to "shiver", which raises the temperature of the body to a certain degree, and thus that of the eggs. Keeping the eggs at a constant temperature is essential for healthy embryo development. During the incubation period, females do not eat and leave only to bask to raise their body temperature.\n    Most species in this family are available in the exotic pet trade. However, caution must be exercised with the larger species, as they can be dangerous; rare cases of large specimens killing their owners have been documented.\n    Obsolete classification schemes—such as that of Boulenger (1890)—place pythons in Pythoninae, a subfamily of the boa family, Boidae. However, despite a superficial resemblance to boas, pythons are more closely related to the sunbeam snakes (Xenopeltis) and the Mexican burrowing python (Loxocemus).\n    Trade in python skins is a lucrative business with the export market from Southeast Asia estimated at US$1 billion as of 2012. Much of the trade is illegal, and python farming is very expensive. Pythons are poached for their meat, mostly consumed locally as bushmeat, and their skin, which is sent to Europe and North America for manufacture of accessories like bags, belts and shoes.\n    In Cameroon bushmeat markets, the Central African rock python (Python sebae) is sold for meat. Hunting, killing and selling pythons is illegal in Cameroon under national wildlife law, but there is little to no enforcement.\n    Pythons and human health. Pythons are not venomous, but like other reptiles, they can be vectors for infections that affect humans, such as salmonella. Such diseases may be transmitted to humans through excreted waste, open wounds, and contaminated water.\n    Pythons are also integrated into some aspects of African health and belief use, often with the added risk of contacting zoonotic diseases. Python bodies and blood are used for African traditional medicines and other belief uses as well, one in-depth study of all animals used by the Yorubas of Nigeria for traditional medicine found that the African Python is used to cure rheumatism, snake poison, appeasing witches, and accident prevention.\n    Python habitats, diets, and invasion into new areas also impact human health and prosperity. A University of Florida Institute of Food and Agriculture Sciences study found that the Burmese python, as an invasive species, enters new habitats and eats an increasing number of mammals, leaving limited species for mosquitoes to bite, forcing them to bite disease-carrying hispid cotton rats and then infect humans with the Everglades virus, a dangerous infection that is carried by very few animals. While direct human-python interactions can be potentially dangerous, the risk of zoonotic diseases is always a concern, whether considering medical and belief use in Nigeria or when addressing invasive species impacts in Florida. In 2022, a woman who lived near a lake area in south-eastern New South Wales state, Australia, was found to be infested with the Ophidascaris robertsi roundworm which is common in carpet pythons - non-venomous snakes found across much of Australia.\n    Python skin has traditionally been used as the attire of choice for medicine men and healers. Typically, South African Zulu traditional healers will use python skin in ceremonial regalia. Pythons are viewed by the Zulu tradition to be a sign of power. Healers are seen as all-powerful since they have a wealth of knowledge, as well as accessibility to the ancestors.\n    Typically, species are attributed to healing various ailments based on their likeliness to a specific bodily attribute. For example, in many cultures, the python is seen as a strong and powerful creature. As a result, pythons are often prescribed as a method of increasing strength. It is very common for the body fat of pythons to be used to treat a large variation of issues such as joint pain, rheumatic pain, toothache and eye sight. Additionally, python fat has been used to treat those suffering from mental illnesses like psychosis. Their calm nature is thought to be of use to treat combative patients. The fat of the python is rubbed onto the body part that is in pain. To improve mental illnesses, it is often rubbed on the temple. The existence of evidence for genuine anti inflammatory and anti-microbial properties of the refined \'snake oil\' is ironic with respect to the expression "snake oil salesman".\n    The Sukuma tribe of Tanzania have been known to use python feces in order to treat back pain. The feces are frequently mixed with a little water, placed on the back, and left for two to three days.\n    In Nigeria, the gallbladder and liver of a python are used to treat poison or bites from other snakes. The python head has been used to "appease witches". Many traditional African cultures believe that they can be cursed by witches. In order to reverse spells and bad luck, traditional doctors will prescribe python heads.\n    In northwestern Ghana, people see pythons as a savior and have taboos to prevent the snake from being harmed or eaten. Their folklore states that this is because a python once helped them flee from their enemies by transforming into a log to allow them to cross a river.\n    In Benin, Vodun practitioners believe that pythons symbolize strength and the spirit of Dagbe ["to do good" in Yoruba]. Annually, people sacrifice animals and proclaim their sins to pythons that are kept inside temples.\n    '
        text = text.replace('\n', ' ')
        sentences = text.split(sep='. ')
        sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 4]  # Fallback
        python_sentences = [s for s in sentences if 'python' in s.lower()]
        return (sentences, python_sentences)  # Annotate a subset or all if small  # 1. Clustering (High-dimensional for better semantic separation)  # 2. Dimensionality Reduction  # 3. Find representative words  # Calculate distance of each point to its cluster center  # Group indices by cluster and sort by distance  # Sort indices by distance for this cluster  # Select top n_labels words, distributing across clusters (round-robin by cluster size)  # Sort clusters by size (largest first) to prioritize labeling big clusters if n_labels < n_clusters  # 4. Plot  # Choose colormap based on number of clusters  # Scatter plot colored by cluster  # Annotate selected words  # _seed_all(seed)  # Narrative/Nature  # Biology/Scientific  # Captivity/Terrarium  # Short/Punchy  # Use a set to ensure uniqueness  # Choose a random template  # Fill the template  # pluralizing logic handled by template context  # randomly shuffle dataset to prevent ordering  # for reproducibility since sets are unordered
    return (
        add_new_token,
        cluster_and_plot,
        cosine_similarity,
        extract_sentence_embeddings,
        extract_word_embeddings,
        fine_tune_model,
        generate_reptile_python_dataset,
        generate_text,
        get_contextual_embedding,
        get_embedding,
        get_next_token_probability,
        get_python_wikipedia_sentences,
        get_sentence_embedding,
        get_top_k_next_tokens,
        initialize_model,
        load_tokenizer,
        prepare_dataset,
        score_perplexity,
    )


@app.cell
def _(SEED, deepcopy, generate_text, initialize_model, load_tokenizer):
    seed = _seed_all(SEED)
    model = initialize_model(pretrained_model_name="distilgpt2")
    base_model = deepcopy(model)
    tokenizer = load_tokenizer()

    generated = generate_text("Once upon a time", base_model, tokenizer, seed=SEED)
    print(f"Generated Text: {generated}")

    generated = generate_text("Apple is a delicious", base_model, tokenizer, seed=SEED)
    print(f"Generated Text: {generated}")
    return base_model, model, tokenizer


@app.cell
def _(add_new_token, base_model, tokenizer):
    new_token = 'jeezz'
    base_model_1, tokenizer_1 = add_new_token(new_token, base_model, tokenizer)
    print(f'Token ID: {tokenizer_1.encode(new_token)}')
    return base_model_1, tokenizer_1


@app.cell
def _(get_embedding, model, tokenizer_1):
    get_embedding('king', model, tokenizer_1).size
    return


@app.cell
def _(get_contextual_embedding, model, tokenizer_1):
    # get_contextual_embedding("The dog is barking", "dog", model, tokenizer)
    get_contextual_embedding('The cat is sitting on the blue desk', 'cat', model, tokenizer_1).size
    return


@app.cell
def _(SEED, cluster_and_plot, extract_word_embeddings, model, tokenizer_1):
    toy_words = ['apple', 'banana', 'orange', 'fruit', 'grape', 'google', 'microsoft', 'apple', 'company', 'facebook', 'iphone', 'android', 'laptop', 'computer', 'technology', 'dog', 'cat', 'animal', 'pet', 'puppy']
    embeddings_array, valid_words = extract_word_embeddings(toy_words, model, tokenizer_1)
    # it's visible that the embeddings from the static embedding matrix do not automatically reflect semantic relationships as well as contextual embeddings would.
    cluster_and_plot(embeddings_array, valid_words, n_clusters=4, n_labels=len(toy_words), method='pca', seed=SEED)
    return


@app.cell
def _(
    SEED,
    base_model_1,
    generate_text,
    get_next_token_probability,
    get_top_k_next_tokens,
    tokenizer_1,
):
    prompt = 'Python is a'
    generated_1 = generate_text(prompt, base_model_1, tokenizer_1, temperature=0.9, sample=True, seed=SEED)
    print(f'Generated Text: {generated_1.strip()}')
    top_k_next_tokens = get_top_k_next_tokens(prompt, base_model_1, tokenizer_1, k=10)
    print('Top 5 next token predictions:')
    for token, prob in top_k_next_tokens:
        print(f"Token: '{token}' with probability {prob:.3f}")
    next_token = 'snake'
    next_token_proba = get_next_token_probability(prompt, next_token, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token.strip()}' as next token: {next_token_proba:.4f}")
    next_token = 'bird'
    next_token_proba = get_next_token_probability(prompt, next_token, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token.strip()}' as next token: {next_token_proba:.4f}")
    return


@app.cell
def _(
    SEED,
    base_model_1,
    generate_text,
    get_next_token_probability,
    get_top_k_next_tokens,
    tokenizer_1,
):
    prompt_1 = 'A Python is a'  # helping it to steer it towards "reptile" or "snake"
    generated_2 = generate_text(prompt_1, base_model_1, tokenizer_1, temperature=0.9, sample=True, seed=SEED)
    print(f'Generated Text: {generated_2.strip()}')
    top_k_next_tokens_1 = get_top_k_next_tokens(prompt_1, base_model_1, tokenizer_1, k=10)
    print('Top 5 next token predictions:')
    for token_1, prob_1 in top_k_next_tokens_1:
        print(f"Token: '{token_1}' with probability {prob_1:.3f}")
    next_token_1 = 'snake'
    next_token_proba_1 = get_next_token_probability(prompt_1, next_token_1, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_1.strip()}' as next token: {next_token_proba_1:.4f}")
    next_token_1 = 'bird'
    next_token_proba_1 = get_next_token_probability(prompt_1, next_token_1, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_1.strip()}' as next token: {next_token_proba_1:.4f}")
    return


@app.cell
def _(
    SEED,
    base_model_1,
    generate_text,
    get_next_token_probability,
    get_top_k_next_tokens,
    tokenizer_1,
):
    prompt_2 = 'Python is a widely known'  # helping it to steer it towards "programming language"
    generated_3 = generate_text(prompt_2, base_model_1, tokenizer_1, temperature=0.9, sample=True, seed=SEED)
    print(f'Generated Text: {generated_3.strip()}')
    top_k_next_tokens_2 = get_top_k_next_tokens(prompt_2, base_model_1, tokenizer_1, k=10)
    print('Top 5 next token predictions:')
    for token_2, prob_2 in top_k_next_tokens_2:
        print(f"Token: '{token_2}' with probability {prob_2:.3f}")
    next_token_2 = 'snake'
    next_token_proba_2 = get_next_token_probability(prompt_2, next_token_2, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_2.strip()}' as next token: {next_token_proba_2:.4f}")
    next_token_2 = 'bird'
    next_token_proba_2 = get_next_token_probability(prompt_2, next_token_2, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_2.strip()}' as next token: {next_token_proba_2:.4f}")
    return


@app.cell
def _(
    SEED,
    base_model_1,
    generate_text,
    get_next_token_probability,
    get_top_k_next_tokens,
    tokenizer_1,
):
    prompt_3 = 'A python is an animal, specifically, it is a'  # helping it to steer it towards "programming language"
    generated_4 = generate_text(prompt_3, base_model_1, tokenizer_1, temperature=0.5, sample=True, seed=SEED)
    print(f'Generated Text: {generated_4.strip()}')
    top_k_next_tokens_3 = get_top_k_next_tokens(prompt_3, base_model_1, tokenizer_1, k=10)
    print('Top 5 next token predictions:')
    for token_3, prob_3 in top_k_next_tokens_3:
        print(f"Token: '{token_3}' with probability {prob_3:.3f}")
    next_token_3 = 'snake'
    next_token_proba_3 = get_next_token_probability(prompt_3, next_token_3, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_3.strip()}' as next token: {next_token_proba_3:.4f}")
    next_token_3 = 'reptile'
    next_token_proba_3 = get_next_token_probability(prompt_3, next_token_3, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_3.strip()}' as next token: {next_token_proba_3:.4f}")
    next_token_3 = 'bird'
    next_token_proba_3 = get_next_token_probability(prompt_3, next_token_3, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_3.strip()}' as next token: {next_token_proba_3:.4f}")
    return


@app.cell
def _(base_model_1, cosine_similarity, get_embedding, tokenizer_1):
    py = get_embedding('python', base_model_1, tokenizer_1)
    Py = get_embedding('Python', base_model_1, tokenizer_1)
    print(f"Cosine Similarity between 'python' and 'Python': {cosine_similarity(py, Py):.4f}")
    return (py,)


@app.cell
def _(
    SEED,
    base_model_1,
    generate_text,
    get_next_token_probability,
    get_top_k_next_tokens,
    tokenizer_1,
):
    generated_5 = generate_text('A python is an animal, specifically, it is a', base_model_1, tokenizer_1, temperature=0.5, sample=True, seed=SEED)
    print(f'Generated Text: {generated_5}')
    top_k_next_tokens_4 = get_top_k_next_tokens('A python is an animal, specifically, it is a', base_model_1, tokenizer_1, k=10)
    print('Top 5 next token predictions:')
    for token_4, prob_4 in top_k_next_tokens_4:
        print(f"Token: '{token_4}' with probability {prob_4:.3f}")
    next_token_4 = 'snake'
    next_token_proba_4 = get_next_token_probability('A python is an animal, specifically, it is a', next_token_4, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_4.strip()}' as next token: {next_token_proba_4:.4f}")
    next_token_4 = 'bird'
    next_token_proba_4 = get_next_token_probability('A python is an animal, specifically, it is a', next_token_4, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_4.strip()}' as next token: {next_token_proba_4:.4f}")
    return


@app.cell
def _(
    SEED,
    base_model_1,
    cluster_and_plot,
    extract_sentence_embeddings,
    tokenizer_1,
):
    sentences = ['Artificial intelligence is transforming the world.', 'Machine learning models require vast amounts of data.', 'The new iPhone features a powerful processor.', 'Coding in Python is very popular among data scientists.', 'Cloud computing enables scalable web applications.', 'Neural networks are designed to mimic the human brain.', 'Data science is a rapidly growing field in tech.', 'Cybersecurity is essential for protecting digital assets.', 'Virtual reality offers immersive gaming experiences.', 'Blockchain technology secures digital transactions.', 'The lion is often called the king of the jungle.', 'Eagles soar high above the mountain peaks.', 'Whales migrate across the vast oceans every year.', 'The rainforest is teeming with diverse wildlife.', 'Cats are known for being independent and curious pets.', 'Dogs are loyal companions to humans.', 'A Python is very universal.', 'The coral reef is a colorful underwater ecosystem.', 'Polar bears are well adapted to the cold arctic climate.', 'Butterflies undergo a remarkable metamorphosis.', 'Bees play a crucial role in pollinating flowers.', 'Pizza is a favorite dish for people worldwide.', 'Fresh vegetables are essential for a healthy diet.', 'Chocolate cake is a rich and decadent dessert.', 'Sushi is a traditional Japanese delicacy.', 'Spicy curry is popular in many Asian cultures.', 'Baking bread requires patience and precision.', 'Pasta comes in many different shapes and sizes.', 'Coffee is a morning ritual for many people.', 'Ice cream is a refreshing treat in the summer.', 'Grilled steak is a savory main course option.', 'Soccer is the most popular sport globally.', 'Basketball requires agility, speed, and teamwork.', 'Tennis is played on grass, clay, or hard courts.', 'Swimming is a great full-body workout.', 'Marathon running tests human endurance limits.']
    sentence_embeddings_array, valid_sentences = extract_sentence_embeddings(sentences, base_model_1, tokenizer_1)  # Tech / AI
    cluster_and_plot(sentence_embeddings_array, sentences, n_clusters=4, n_labels=len(sentences), method='tsne', seed=SEED)  # Nature / Animals  # Food  # Sports
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Adding a additional meaning to a word/token to a model
    - How much fine-tuning is necessary for a model to capture a new meaning of a word?
        - create a function which fine-tunes a model with different parameters on a small dataset where a word is used in a novel sense.
        - parameters: number of epochs, learning rate, dataset size
    - Does the model forget the previous meaning after fine-tuning? --> measure with cosine similarity of contexutal embeddings.
        - Create a function which measures the cosine similarity of contextual embeddings of the target word before and after fine-tuning.
        - Create a function which measures the cosine similarity of entire sentences containing the target word before and after fine-tuning.
        - Include control words & sentences which are not fine-tuned to see if the changes are specific to the target word.
    - Is it possible to erase the prior semantics through fine-tuning?
    - What happens when we extend the embeddings by a novel token? Is this token more susceptible to meaning change if it has no deep prior semantics?
    - What happens when we replace the original embedding vector of a token by a combination of vectors from other tokens which semantic meaning should be adopted by the target token? (alternative to fine-tuning, might lead to forgetting of old semantics)


    **Approach**
    - create a mix of a small scale synthetic dataset where a word is used in a novel sense and real-world sentences from wikipedia where the word is used in its original sense.
    - Model understands 'python' mainly as a programming language --> how to add the meaning of a reptile/snake/animal to it?
        - Use a mix of small-scale synthetic dataset & wikipedia article about pythons (the reptile) to fine-tune and test how much \{data, fine-tuning\} is necessary to:
            - add the new meaning to the model (contextual)
            - measure forgetting of the original meaning (contextual)
    """)
    return


@app.cell
def _(get_python_wikipedia_sentences):
    get_python_wikipedia_sentences()[0][:10]
    return


@app.cell
def _(
    Dataset,
    generate_reptile_python_dataset,
    get_python_wikipedia_sentences,
):
    template_sentences = generate_reptile_python_dataset(5, simple=False)
    wikipedia_sentences, python_sentences = get_python_wikipedia_sentences()
    dataset = Dataset.from_dict({"text": template_sentences + wikipedia_sentences})

    print("\nSample Data:")
    for i in range(5):
        print(f"{i+1}. {template_sentences[i]}")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Early Stopping and Validation Set
    To prevent overfitting and stop training when the validation loss stops improving, we can split the dataset into training and validation sets.
    We then pass the validation set to `fine_tune_model`. The updated function now supports `EarlyStoppingCallback` which monitors the validation loss.
    """)
    return


@app.cell
def _(
    FINE_TUNED_MODEL_DIR,
    SEED,
    base_model_1,
    dataset,
    fine_tune_model,
    prepare_dataset,
    tokenizer_1,
):
    # Split the dataset into train and validation (e.g., 90% train, 10% validation)
    dataset_split = dataset.train_test_split(test_size=0.1, shuffle=True, stratify_by_column=None, seed=SEED)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']
    print('Tokenizing training set...')
    # Tokenize both datasets
    tokenized_train = prepare_dataset(dataset=train_dataset, tokenizer=tokenizer_1)
    print('Tokenizing validation set...')
    tokenized_eval = prepare_dataset(dataset=eval_dataset, tokenizer=tokenizer_1)
    # Fine-tune with early stopping
    fine_tuned_model, log_history = fine_tune_model(model=base_model_1, tokenizer=tokenizer_1, dataset=tokenized_train, eval_dataset=tokenized_eval, epochs=1, lr=5e-05, decay=0.01, batch_size=4, patience=1, output_dir=FINE_TUNED_MODEL_DIR)  # Pass the validation set  # Increase epochs to allow early stopping to kick in (e.g. 10 instead of 3)  # Stop if no improvement after 3 evaluations, test is very very similar to train!
    return (fine_tuned_model,)


@app.cell
def _(
    base_model_1,
    cosine_similarity,
    fine_tuned_model,
    get_embedding,
    py,
    tokenizer_1,
):
    a = get_embedding('python', base_model_1, tokenizer_1)
    b = get_embedding('python', fine_tuned_model, tokenizer_1)
    assert cosine_similarity(a, b) < 1.0, 'Fine-tuned model is the same as the base_model'
    assert cosine_similarity(a, py) == 1.0, 'Base model did get get modified'
    assert cosine_similarity(b, py) < 1.0, 'Base model did get get modified'
    (cosine_similarity(a, b), cosine_similarity(a, py), cosine_similarity(b, py))
    return


@app.cell
def _(
    SEED,
    base_model_1,
    generate_text,
    get_next_token_probability,
    get_top_k_next_tokens,
    tokenizer_1,
):
    prompt_4 = 'A python is a'  # helping it to steer it towards "programming language"
    generated_6 = generate_text(prompt_4, base_model_1, tokenizer_1, temperature=0.5, sample=True, seed=SEED)
    print(f'Generated Text: {generated_6.strip()}')
    top_k_next_tokens_5 = get_top_k_next_tokens(prompt_4, base_model_1, tokenizer_1, k=10)
    print('Top 5 next token predictions:')
    for token_5, prob_5 in top_k_next_tokens_5:
        print(f"Token: '{token_5}' with probability {prob_5:.3f}")
    next_token_5 = 'snake'
    next_token_proba_5 = get_next_token_probability(prompt_4, next_token_5, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_5.strip()}' as next token: {next_token_proba_5:.4f}")
    next_token_5 = 'reptile'
    next_token_proba_5 = get_next_token_probability(prompt_4, next_token_5, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_5.strip()}' as next token: {next_token_proba_5:.4f}")
    next_token_5 = 'bird'
    next_token_proba_5 = get_next_token_probability(prompt_4, next_token_5, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_5.strip()}' as next token: {next_token_proba_5:.4f}")
    return


@app.cell
def _(
    SEED,
    fine_tuned_model,
    generate_text,
    get_next_token_probability,
    get_top_k_next_tokens,
    tokenizer_1,
):
    prompt_5 = 'A python is a'  # helping it to steer it towards "programming language"
    generated_7 = generate_text(prompt_5, fine_tuned_model, tokenizer_1, temperature=0.5, sample=True, seed=SEED)
    print(f'Generated Text: {generated_7.strip()}')
    top_k_next_tokens_6 = get_top_k_next_tokens(prompt_5, fine_tuned_model, tokenizer_1, k=10)
    print('Top 5 next token predictions:')
    for token_6, prob_6 in top_k_next_tokens_6:
        print(f"Token: '{token_6}' with probability {prob_6:.3f}")
    next_token_6 = 'snake'
    next_token_proba_6 = get_next_token_probability(prompt_5, next_token_6, fine_tuned_model, tokenizer_1)
    print(f"Probability of '{next_token_6.strip()}' as next token: {next_token_proba_6:.4f}")
    next_token_6 = 'reptile'
    next_token_proba_6 = get_next_token_probability(prompt_5, next_token_6, fine_tuned_model, tokenizer_1)
    print(f"Probability of '{next_token_6.strip()}' as next token: {next_token_proba_6:.4f}")
    next_token_6 = 'bird'
    next_token_proba_6 = get_next_token_probability(prompt_5, next_token_6, fine_tuned_model, tokenizer_1)
    print(f"Probability of '{next_token_6.strip()}' as next token: {next_token_proba_6:.4f}")
    return


@app.cell
def _(
    SEED,
    fine_tuned_model,
    generate_text,
    get_next_token_probability,
    get_top_k_next_tokens,
    tokenizer_1,
):
    prompt_6 = 'A python is an animal, specifically, it is a'  # helping it to steer it towards "programming language"
    generated_8 = generate_text(prompt_6, fine_tuned_model, tokenizer_1, temperature=0.5, sample=True, seed=SEED)
    print(f'Generated Text: {generated_8.strip()}')
    top_k_next_tokens_7 = get_top_k_next_tokens(prompt_6, fine_tuned_model, tokenizer_1, k=10)
    print('Top 5 next token predictions:')
    for token_7, prob_7 in top_k_next_tokens_7:
        print(f"Token: '{token_7}' with probability {prob_7:.3f}")
    next_token_7 = 'snake'
    next_token_proba_7 = get_next_token_probability(prompt_6, next_token_7, fine_tuned_model, tokenizer_1)
    print(f"Probability of '{next_token_7.strip()}' as next token: {next_token_proba_7:.4f}")
    next_token_7 = 'reptile'
    next_token_proba_7 = get_next_token_probability(prompt_6, next_token_7, fine_tuned_model, tokenizer_1)
    print(f"Probability of '{next_token_7.strip()}' as next token: {next_token_proba_7:.4f}")
    next_token_7 = 'bird'
    next_token_proba_7 = get_next_token_probability(prompt_6, next_token_7, fine_tuned_model, tokenizer_1)
    print(f"Probability of '{next_token_7.strip()}' as next token: {next_token_proba_7:.4f}")
    return


@app.cell
def _(
    SEED,
    base_model_1,
    generate_text,
    get_next_token_probability,
    get_top_k_next_tokens,
    tokenizer_1,
):
    prompt_7 = 'Programmers love python because python is a'  # helping it to steer it towards "programming language"
    generated_9 = generate_text(prompt_7, base_model_1, tokenizer_1, temperature=0.9, sample=True, seed=SEED)
    print(f'Generated Text: {generated_9.strip()}')
    top_k_next_tokens_8 = get_top_k_next_tokens(prompt_7, base_model_1, tokenizer_1, k=10)
    print('Top 5 next token predictions:')
    for token_8, prob_8 in top_k_next_tokens_8:
        print(f"Token: '{token_8}' with probability {prob_8:.3f}")
    next_token_8 = 'snake'
    next_token_proba_8 = get_next_token_probability(prompt_7, next_token_8, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_8.strip()}' as next token: {next_token_proba_8:.4f}")
    next_token_8 = 'bird'
    next_token_proba_8 = get_next_token_probability(prompt_7, next_token_8, base_model_1, tokenizer_1)
    print(f"Probability of '{next_token_8.strip()}' as next token: {next_token_proba_8:.4f}")
    return


@app.cell
def _(
    SEED,
    fine_tuned_model,
    generate_text,
    get_next_token_probability,
    get_top_k_next_tokens,
    tokenizer_1,
):
    prompt_8 = 'Programmers love python because python is a'  # helping it to steer it towards "programming language"
    generated_10 = generate_text(prompt_8, fine_tuned_model, tokenizer_1, temperature=0.9, sample=True, seed=SEED)
    print(f'Generated Text: {generated_10.strip()}')
    top_k_next_tokens_9 = get_top_k_next_tokens(prompt_8, fine_tuned_model, tokenizer_1, k=10)
    print('Top 5 next token predictions:')
    for token_9, prob_9 in top_k_next_tokens_9:
        print(f"Token: '{token_9}' with probability {prob_9:.3f}")
    next_token_9 = 'snake'
    next_token_proba_9 = get_next_token_probability(prompt_8, next_token_9, fine_tuned_model, tokenizer_1)
    print(f"Probability of '{next_token_9.strip()}' as next token: {next_token_proba_9:.4f}")
    next_token_9 = 'bird'
    next_token_proba_9 = get_next_token_probability(prompt_8, next_token_9, fine_tuned_model, tokenizer_1)
    print(f"Probability of '{next_token_9.strip()}' as next token: {next_token_proba_9:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiments
    - Assess how much fine-tuning (in terms of epochs, learning rate, dataset size) is necessary for the model to effectively learn and incorporate the new meaning of the target word.
    - Evaluate, whether embeddings, contextual embeddings, or sentence embeddings capture the new meaning the best and whether they change.
    - How much fine-tuning is necessary to forget the old meaning of the target word?
    """)
    return


@app.cell
def _(base_model_1, get_contextual_embedding, tokenizer_1):
    get_contextual_embedding(text='Programmers love python because it is', target_word='python', model=base_model_1, tokenizer=tokenizer_1)
    get_contextual_embedding(text='Programmers love python because it is', target_word=' python', model=base_model_1, tokenizer=tokenizer_1)
    # Warning: ' Python' not found in tokenized text.
    get_contextual_embedding(text='Programmers love python because it is', target_word=' Python', model=base_model_1, tokenizer=tokenizer_1)
    return


@app.cell
def _(base_model_1, cosine_similarity, get_embedding, tokenizer_1):
    cosine_similarity(get_embedding('python', base_model_1, tokenizer_1), get_embedding(' python', base_model_1, tokenizer_1))
    return


@app.cell
def _(base_model_1, cosine_similarity, get_embedding, tokenizer_1):
    cosine_similarity(get_embedding('python', base_model_1, tokenizer_1), get_embedding('Python', base_model_1, tokenizer_1))
    return


@app.cell
def _(base_model_1, cosine_similarity, get_embedding, tokenizer_1):
    cosine_similarity(get_embedding(' python', base_model_1, tokenizer_1), get_embedding('Python', base_model_1, tokenizer_1))
    return


@app.cell
def _(tokenizer_1):
    # token ids
    (tokenizer_1.encode(' python'), tokenizer_1.encode('Python'), tokenizer_1.encode('python'))
    return


@app.cell
def _(
    DEVICE,
    DataCollatorForLanguageModeling,
    FINE_TUNED_MODEL_DIR,
    GPT2LMHeadModel,
    List,
    PCA,
    Path,
    SEED,
    TSNE,
    Trainer,
    TrainingArguments,
    cosine_similarity,
    deepcopy,
    fine_tune_model,
    get_contextual_embedding,
    get_embedding,
    get_next_token_probability,
    get_sentence_embedding,
    np,
    plt,
    prepare_dataset,
    score_perplexity,
    torch,
):
    # EXPERIMENTAL SETUP
    # from itertools import product
    import itertools
    import pandas as pd
    param_grid = {'learning_rate': [1e-05, 5e-06], 'num_epochs': [1, 3], 'dataset_size': [150, 1000], 'K': 4, 'num_samples': 10}

    class Experimenter:
      # 'batch_size': [8],
        def __init__(self, target_words, base_model, tokenizer, dataset, param_grid, eval_dataset=None, model_save_path=FINE_TUNED_MODEL_DIR, seed=SEED):  # 'weight_decay': [0.0, 0.01],
    # param_grid = {
    #     'learning_rate': [3e-4, 1e-4, 1e-5, 5e-6],
    #     'num_epochs': [1, 3, 5],
    #     'dataset_size': [150, 500, 1_000, 2_000], 
            self.target_words = target_words  # 'K': 4, # number of tokens to generate for sampled generation assessment
            self.base_model = base_model  # 'num_samples': 10,
    # }
            self.tokenizer = tokenizer
            self.dataset = dataset  # minimal test setup
            self.param_grid = param_grid
            self.eval_dataset = eval_dataset
            self.model_save_path = model_save_path
            self.K = param_grid['K']  # number of tokens to generate for sampled generation assessment
            self.num_samples = param_grid['num_samples']  # number of samples to generate per prompt for sampled generation assessment
            self.seed = seed
            self.results = []
            self.models = {}

        def _config_key(self, config):
            return f"lr{config['learning_rate']}_ep{config['num_epochs']}_ds{config['dataset_size']}"

        def _fine_tune(self, config):
            n_samples = min(config['dataset_size'], len(self.dataset))
            train_data = prepare_dataset(self.dataset.select(range(n_samples)), tokenizer=self.tokenizer)
            eval_data = prepare_dataset(self.eval_dataset, tokenizer=self.tokenizer) if self.eval_dataset else None
            return fine_tune_model(model=self.base_model, tokenizer=self.tokenizer, dataset=train_data, eval_dataset=eval_data, epochs=config['num_epochs'], lr=config['learning_rate'], batch_size=8, output_dir=Path(self.model_save_path))

        def _compute_static_drift(self, ft_model, tokens):
            results = {}
            for t in tokens:
                try:
                    base_emb = get_embedding(t, self.base_model, self.tokenizer)
                    ft_emb = get_embedding(t, ft_model, self.tokenizer)
                    results[t] = cosine_similarity(base_emb, ft_emb)
                except:
                    results[t] = np.nan
            return results

        def _compute_contextual_drift(self, ft_model, context_sentences):
            results = {}
            for tw in self.target_words:
                results[tw] = {}
                for s in context_sentences:
                    base_emb = get_contextual_embedding(s, tw, self.base_model, self.tokenizer)
                    ft_emb = get_contextual_embedding(s, tw, ft_model, self.tokenizer)
                    if np.any(base_emb) and np.any(ft_emb):
                        results[tw][s] = cosine_similarity(base_emb, ft_emb)
            return results

        def _compute_sentence_drift(self, ft_model, sentences):
            return {s: cosine_similarity(get_sentence_embedding(s, self.base_model, self.tokenizer), get_sentence_embedding(s, ft_model, self.tokenizer)) for s in sentences}

        def _compute_perplexity(self, ft_model, prompts):
            return {p: {'base': score_perplexity(p, self.base_model, self.tokenizer), 'ft': score_perplexity(p, ft_model, self.tokenizer)} for p in prompts}

        def _compute_token_probs(self, ft_model, prompts, tokens):
            results = {}
            for p in prompts:
                results[p] = {t: {'base': get_next_token_probability(p, t, self.base_model, self.tokenizer), 'ft': get_next_token_probability(p, t, ft_model, self.tokenizer)} for t in tokens}
            return results

        def _generate_k_tokens(self, prompt: str, model: GPT2LMHeadModel, k: int=None, temperature: float=0.8, num_samples: int=None) -> List[str]:
            """
            Generate multiple sentence completions with k tokens using sampling.
            Returns list of generated continuations (only the new tokens, not the prompt).
            """
            inputs = self.tokenizer(prompt, return_tensors='pt').to(DEVICE)
            prompt_length = inputs.input_ids.shape[1]
            model.eval()
            if k is None:
                k = self.K
            if num_samples is None:
                num_samples = self.num_samples
            completions = []
            with torch.no_grad():
                for i in range(num_samples):
                    _seed_all(self.seed + i)
                    outputs = model.generate(**inputs, max_new_tokens=k, do_sample=True, temperature=temperature, top_p=0.95, pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=1)
                    generated_ids = outputs[0][prompt_length:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    completions.append(generated_text.lower())
            return completions

        def _compute_sampled_generation_assessment(self, ft_model, prompts: List[str], reptile_words: List[str], programming_words: List[str], k_tokens: int=10, num_samples: int=100):
            """
            Assessment 1: Sampled Generation Assessment
        
            For each prompt, generate num_samples completions with k tokens each.
            Measure the probability distribution of expected words appearing in the
            generated completions. This captures the stochastic nature of generation
            and provides insights into whether the model can generate sentences in
            the expected context, or whether it forgot the old context.
        
            Returns detailed results including:
            - Per-prompt occurrence rates for reptile vs programming words
            - Probability distributions across samples
            - Shift in generation behavior (base vs fine-tuned)
            """
            results = {'prompts': {}, 'summary': {}}
            all_reptile_base, all_reptile_ft = ([], [])
            all_prog_base, all_prog_ft = ([], [])  # Set seed for reproducibility within the sampling
            for prompt in prompts:
                base_completions = self._generate_k_tokens(prompt, self.base_model, k=k_tokens, num_samples=num_samples)
                ft_completions = self._generate_k_tokens(prompt, ft_model, k=k_tokens, num_samples=num_samples)

                def count_word_occurrences(completions, target_words):
                    """Count how many completions contain at least one target word."""
                    counts = {w: 0 for w in target_words}
                    any_target = 0
                    for completion in completions:
                        found_any = False
                        for word in target_words:  # Extract only the generated part (exclude prompt)
                            if word.lower() in completion:
                                counts[word] = counts[word] + 1
                                found_any = True
                        if found_any:
                            any_target = any_target + 1
                    return (counts, any_target)
                base_reptile_counts, base_reptile_any = count_word_occurrences(base_completions, reptile_words)
                ft_reptile_counts, ft_reptile_any = count_word_occurrences(ft_completions, reptile_words)
                base_prog_counts, base_prog_any = count_word_occurrences(base_completions, programming_words)
                ft_prog_counts, ft_prog_any = count_word_occurrences(ft_completions, programming_words)
                base_reptile_prob = base_reptile_any / num_samples
                ft_reptile_prob = ft_reptile_any / num_samples
                base_prog_prob = base_prog_any / num_samples
                ft_prog_prob = ft_prog_any / num_samples
                results['prompts'][prompt] = {'base_reptile_prob': base_reptile_prob, 'ft_reptile_prob': ft_reptile_prob, 'reptile_prob_change': ft_reptile_prob - base_reptile_prob, 'base_programming_prob': base_prog_prob, 'ft_programming_prob': ft_prog_prob, 'programming_prob_change': ft_prog_prob - base_prog_prob, 'base_reptile_counts': base_reptile_counts, 'ft_reptile_counts': ft_reptile_counts, 'base_programming_counts': base_prog_counts, 'ft_programming_counts': ft_prog_counts, 'sample_completions_base': base_completions[:5], 'sample_completions_ft': ft_completions[:5]}
                all_reptile_base.append(base_reptile_prob)
                all_reptile_ft.append(ft_reptile_prob)
                all_prog_base.append(base_prog_prob)
                all_prog_ft.append(ft_prog_prob)
            results['summary'] = {'mean_reptile_prob_base': np.mean(all_reptile_base), 'mean_reptile_prob_ft': np.mean(all_reptile_ft), 'mean_reptile_prob_change': np.mean(all_reptile_ft) - np.mean(all_reptile_base), 'mean_programming_prob_base': np.mean(all_prog_base), 'mean_programming_prob_ft': np.mean(all_prog_ft), 'mean_programming_prob_change': np.mean(all_prog_ft) - np.mean(all_prog_base), 'std_reptile_prob_ft': np.std(all_reptile_ft), 'std_programming_prob_ft': np.std(all_prog_ft)}
            return results

        def _compute_polysemy_disambiguation(self, ft_model, num_samples: int=10, k_tokens: int=15):
            """
            Assessment 2: Polysemy Disambiguation Test
        
            Tests whether the model can correctly disambiguate 'python' based on context.
            Accounts for generic tokens (a, the, very, etc.) that may appear before
            the meaningful context token.
        
            Tests both:
            - Acquisition of new semantics (reptile context understanding)
            - Retention of old semantics (programming context - catastrophic forgetting)
        
            Returns detailed disambiguation scores and forgetting metrics.  # Generate completions from base and fine-tuned models
            """
            test_cases = [{'prompt': 'The python slithered and then', 'context': 'reptile', 'expected': ['hissed', 'coiled', 'ate', 'struck', 'moved', 'wrapped', 'bit', 'crawled', 'rested', 'hunted'], 'unexpected': ['compiled', 'executed', 'imported', 'ran', 'printed', 'returned', 'crashed']}, {'prompt': 'A python wrapped around the', 'context': 'reptile', 'expected': ['branch', 'tree', 'prey', 'rat', 'mouse', 'animal', 'victim', 'body', 'leg', 'arm'], 'unexpected': ['function', 'code', 'string', 'variable', 'list', 'class', 'module']}, {'prompt': 'In the jungle, pythons typically', 'context': 'reptile', 'expected': ['hunt', 'eat', 'live', 'hide', 'sleep', 'rest', 'prey', 'climb', 'slither', 'attack'], 'unexpected': ['run', 'execute', 'compile', 'import', 'install', 'debug']}, {'prompt': 'To install packages in Python, use', 'context': 'programming', 'expected': ['pip', 'conda', 'install', 'package', 'command', 'terminal', 'shell'], 'unexpected': ['teeth', 'fangs', 'venom', 'scales', 'skin', 'coils']}, {'prompt': 'In Python, to define a function you use', 'context': 'programming', 'expected': ['def', 'function', 'keyword', 'syntax', 'lambda', 'code'], 'unexpected': ['snake', 'reptile', 'animal', 'bite', 'slither', 'coil']}, {'prompt': 'Python developers often use', 'context': 'programming', 'expected': ['libraries', 'frameworks', 'tools', 'packages', 'modules', 'code', 'functions', 'classes', 'jupyter', 'notebooks', 'ide'], 'unexpected': ['cages', 'terrariums', 'habitats', 'enclosures', 'tanks']}, {'prompt': 'Python is known for its', 'context': 'ambiguous', 'expected': ['size', 'strength', 'simplicity', 'readability', 'power', 'flexibility', 'length', 'speed'], 'unexpected': []}]
            results = {'test_cases': [], 'reptile_acquisition': {}, 'programming_retention': {}, 'summary': {}}
            reptile_scores_base, reptile_scores_ft = ([], [])  # Count occurrences of target words in completions
            prog_scores_base, prog_scores_ft = ([], [])
            for test in test_cases:
                prompt = test['prompt']
                context = test['context']
                expected = test['expected']
                unexpected = test['unexpected']
                base_completions = self._generate_k_tokens(prompt, self.base_model, k=k_tokens, num_samples=num_samples)
                ft_completions = self._generate_k_tokens(prompt, ft_model, k=k_tokens, num_samples=num_samples)

                def score_completions(completions, expected_words, unexpected_words):
                    """
                    Score completions based on presence of expected vs unexpected words.
                    Returns: (expected_rate, unexpected_rate, discrimination_score)
                    """
                    expected_count = 0
                    unexpected_count = 0
                    for completion in completions:
                        comp_lower = completion.lower()
                        if any((word.lower() in comp_lower for word in expected_words)):
                            expected_count = expected_count + 1
                        if unexpected_words and any((word.lower() in comp_lower for word in unexpected_words)):  # Calculate probabilities (occurrence rates)
                            unexpected_count = unexpected_count + 1
                    expected_rate = expected_count / len(completions)
                    unexpected_rate = unexpected_count / len(completions) if unexpected_words else 0
                    discrimination = expected_rate - unexpected_rate
                    return (expected_rate, unexpected_rate, discrimination)
                base_exp, base_unexp, base_disc = score_completions(base_completions, expected, unexpected)
                ft_exp, ft_unexp, ft_disc = score_completions(ft_completions, expected, unexpected)
                case_result = {'prompt': prompt, 'context': context, 'base_expected_rate': base_exp, 'ft_expected_rate': ft_exp, 'expected_rate_change': ft_exp - base_exp, 'base_unexpected_rate': base_unexp, 'ft_unexpected_rate': ft_unexp, 'unexpected_rate_change': ft_unexp - base_unexp, 'base_discrimination': base_disc, 'ft_discrimination': ft_disc, 'discrimination_change': ft_disc - base_disc, 'sample_completions_base': base_completions[:3], 'sample_completions_ft': ft_completions[:3]}
                results['test_cases'].append(case_result)
                if context == 'reptile':
                    reptile_scores_base.append(base_disc)
                    reptile_scores_ft.append(ft_disc)
                elif context == 'programming':
                    prog_scores_base.append(base_disc)
                    prog_scores_ft.append(ft_disc)
            results['reptile_acquisition'] = {'mean_discrimination_base': np.mean(reptile_scores_base) if reptile_scores_base else 0, 'mean_discrimination_ft': np.mean(reptile_scores_ft) if reptile_scores_ft else 0, 'acquisition_score': np.mean(reptile_scores_ft) - np.mean(reptile_scores_base) if reptile_scores_ft else 0}
            results['programming_retention'] = {'mean_discrimination_base': np.mean(prog_scores_base) if prog_scores_base else 0, 'mean_discrimination_ft': np.mean(prog_scores_ft) if prog_scores_ft else 0, 'forgetting_score': np.mean(prog_scores_base) - np.mean(prog_scores_ft) if prog_scores_ft else 0}  # Store a few examples
            results['summary'] = {'reptile_acquisition': results['reptile_acquisition']['acquisition_score'], 'programming_forgetting': results['programming_retention']['forgetting_score'], 'net_semantic_shift': results['reptile_acquisition']['acquisition_score'] - results['programming_retention']['forgetting_score']}
            return results

        def _evaluate(self, ft_model, config, prompts, control_words, target_senses, context_sentences):
            control_sentences = ['The weather is nice.', 'Dogs are loyal.', 'Pizza is delicious.']
            static = self._compute_static_drift(ft_model, self.target_words + control_words)
            contextual = self._compute_contextual_drift(ft_model, context_sentences)
            sentence = self._compute_sentence_drift(ft_model, prompts + control_sentences)
            perplexity = self._compute_perplexity(ft_model, prompts)  # Compute summary statistics
            probs = self._compute_token_probs(ft_model, prompts, target_senses)
            reptile_words = ['snake', 'reptile', 'animal', 'serpent', 'constrictor', 'slither']
            programming_words = ['code', 'programming', 'language', 'software', 'script', 'developer']
            print('  Running sampled generation assessment...')
            sampled_gen = self._compute_sampled_generation_assessment(ft_model, prompts, reptile_words, programming_words, k_tokens=10, num_samples=100)
            print('  Running polysemy disambiguation test...')
            polysemy = self._compute_polysemy_disambiguation(ft_model, num_samples=10, k_tokens=15)
            prob_reptile = np.mean([probs[p][t]['ft'] - probs[p][t]['base'] for p in prompts for t in reptile_words if t in probs[p]])
            prob_prog = np.mean([probs[p][t]['ft'] - probs[p][t]['base'] for p in prompts for t in programming_words if t in probs[p]])
            result = {**config}
            for tw in self.target_words:
                tw_label = tw.replace(' ', '_space_') if tw.startswith(' ') else tw
                result[f'static_{tw_label}'] = static.get(tw, np.nan)
                ctx_vals = list(contextual.get(tw, {}).values())
                result[f'contextual_{tw_label}'] = np.mean(ctx_vals) if ctx_vals else np.nan
            result['static_target_avg'] = np.mean([static.get(tw, np.nan) for tw in self.target_words])
            result['static_control'] = np.mean([static.get(t, np.nan) for t in control_words])
            ctx_all = [v for tw in self.target_words for v in contextual.get(tw, {}).values()]
            result['contextual_avg'] = np.mean(ctx_all) if ctx_all else np.nan
            result['sentence_target'] = np.mean([sentence.get(p, np.nan) for p in prompts])
            result['sentence_control'] = np.mean([sentence.get(s, np.nan) for s in control_sentences])
            result['perplexity_base'] = np.mean([perplexity[p]['base'] for p in prompts])
            result['perplexity_ft'] = np.mean([perplexity[p]['ft'] for p in prompts])
            for sense in target_senses:
                prob_change = np.mean([probs[p][sense]['ft'] - probs[p][sense]['base'] for p in prompts])
                result[f'prob_{sense}'] = prob_change
            result['prob_reptile_change'] = prob_reptile
            result['prob_programming_change'] = prob_prog  # Test cases: (prompt, expected_context, expected_words, unexpected_words)
            result['sampled_reptile_prob_base'] = sampled_gen['summary']['mean_reptile_prob_base']
            result['sampled_reptile_prob_ft'] = sampled_gen['summary']['mean_reptile_prob_ft']  # Reptile context prompts - should predict reptile-related words
            result['sampled_reptile_change'] = sampled_gen['summary']['mean_reptile_prob_change']
            result['sampled_programming_prob_base'] = sampled_gen['summary']['mean_programming_prob_base']
            result['sampled_programming_prob_ft'] = sampled_gen['summary']['mean_programming_prob_ft']
            result['sampled_programming_change'] = sampled_gen['summary']['mean_programming_prob_change']
            result['polysemy_reptile_acquisition'] = polysemy['summary']['reptile_acquisition']
            result['polysemy_programming_forgetting'] = polysemy['summary']['programming_forgetting']
            result['polysemy_net_semantic_shift'] = polysemy['summary']['net_semantic_shift']
            result['static_all'] = static
            result['contextual_all'] = contextual
            result['sentence_all'] = sentence
            result['perplexity_all'] = perplexity
            result['probs_all'] = probs
            result['sampled_generation_all'] = sampled_gen
            result['polysemy_all'] = polysemy
            return result

        def run(self):
            prompts = ['Python is a', 'A python is a', 'Pythons are', 'Python is widely known', 'Pythons typically live in the', 'A python is an animal, specifically it is a']
            context_sentences = ['The python slithered across the forest floor.', 'I saw a python at the zoo yesterday.', 'The python coiled around its prey.', 'A large python was spotted near the river.', 'The python is native to tropical regions.', 'Many people fear the python because of its size.', 'Python is used by many developers.', 'Learning Python takes dedication.']  # Programming context prompts - should retain programming knowledge (forgetting test)
            control_words = ['the', 'computer', 'apple', 'ball', 'woman', 'bird']
            target_senses = ['snake', 'reptile', 'animal', 'serpent', 'constrictor', 'slither', 'code', 'programming', 'language', 'software', 'script', 'developer']
            grid_params = {k: v for k, v in self.param_grid.items() if isinstance(v, list)}
            configs = [dict(zip(grid_params.keys(), v)) for v in itertools.product(*grid_params.values())]
            for i, config in enumerate(configs):
                print(f'\n[{i + 1}/{len(configs)}] {config}')
                ft_model, _ = self._fine_tune(config)
                self.models[self._config_key(config)] = ft_model
                self.results.append(self._evaluate(ft_model, config, prompts, control_words, target_senses, context_sentences))
                torch.cuda.empty_cache()
            return pd.DataFrame(self.results)

        def _get_colors(self):
            return {'1e-04': '#e41a1c', '1e-05': '#377eb8', '5e-06': '#4daf4a'}

        def _get_markers(self):
            return {1: 'o', 3: 's'}

        def _get_target_colors(self):  # Ambiguous prompts - test context sensitivity
            return {'Python': '#e41a1c', 'python': '#377eb8', '_space_python': '#4daf4a'}

        def plot_probability_changes(self):
            df = pd.DataFrame(self.results)
            df['lr_str'] = df['learning_rate'].apply(lambda x: f'{x:.0e}')  # Both contexts acceptable
            epochs = sorted(df['num_epochs'].unique())
            colors, markers = (self._get_colors(), self._get_markers())
            reptile_cols = [c for c in df.columns if c.startswith('prob_') and any((w in c for w in ['snake', 'reptile', 'animal', 'serpent', 'constrictor', 'slither']))]
            prog_cols = [c for c in df.columns if c.startswith('prob_') and any((w in c for w in ['code', 'programming', 'language', 'software', 'script', 'developer']))]
            fig, axes = plt.subplots(2, len(epochs), figsize=(5 * len(epochs), 8), sharey='row')
            if len(epochs) == 1:
                axes = axes.reshape(-1, 1)
            for col, ep in enumerate(epochs):
                sub = df[df['num_epochs'] == ep].sort_values('dataset_size')
                for lr_str in sorted(sub['lr_str'].unique()):
                    data = sub[sub['lr_str'] == lr_str]
                    axes[0, col].plot(data['dataset_size'], data['prob_reptile_change'], marker=markers.get(ep, 'o'), color=colors.get(lr_str, 'gray'), linewidth=2, markersize=8, label=f'LR={lr_str}')
                    axes[1, col].plot(data['dataset_size'], data['prob_programming_change'], marker=markers.get(ep, 'o'), color=colors.get(lr_str, 'gray'), linewidth=2, markersize=8, label=f'LR={lr_str}')
                axes[0, col].set_title(f'Epochs = {ep}', fontsize=12, fontweight='bold')
                axes[0, col].axhline(0, color='k', ls='--', alpha=0.3)
                axes[1, col].axhline(0, color='k', ls='--', alpha=0.3)
                axes[1, col].set_xlabel('Dataset Size')
            axes[0, 0].set_ylabel('ΔP(reptile tokens)')
            axes[1, 0].set_ylabel('ΔP(programming tokens)')
            axes[0, -1].legend(loc='upper left', fontsize=9)  # Generate completions
            fig.suptitle('Probability Changes by Training Configuration', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        def plot_token_breakdown(self):
            df = pd.DataFrame(self.results)
            df['lr_str'] = df['learning_rate'].apply(lambda x: f'{x:.0e}')
            reptile_tokens = ['snake', 'reptile', 'animal', 'serpent', 'constrictor', 'slither']
            prog_tokens = ['code', 'programming', 'language', 'software', 'script', 'developer']
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            best_idx = df['prob_reptile_change'].idxmax()
            best = df.loc[best_idx]
            reptile_vals = [best.get(f'prob_{t}', 0) for t in reptile_tokens]
            prog_vals = [best.get(f'prob_{t}', 0) for t in prog_tokens]  # Check if ANY expected word appears
            colors_r = ['#2ecc71' if v > 0 else '#e74c3c' for v in reptile_vals]
            colors_p = ['#2ecc71' if v > 0 else '#e74c3c' for v in prog_vals]
            axes[0, 0].barh(reptile_tokens, reptile_vals, color=colors_r)  # Check if ANY unexpected word appears
            axes[0, 0].axvline(0, color='k', ls='-', alpha=0.5)
            axes[0, 0].set_xlabel('ΔP (probability change)')
            axes[0, 0].set_title('Reptile-Related Token Changes (Best Config)')
            axes[0, 1].barh(prog_tokens, prog_vals, color=colors_p)
            axes[0, 1].axvline(0, color='k', ls='-', alpha=0.5)
            axes[0, 1].set_xlabel('ΔP (probability change)')  # Discrimination: how much better at expected vs unexpected
            axes[0, 1].set_title('Programming-Related Token Changes (Best Config)')
            epochs = sorted(df['num_epochs'].unique())
            width = 0.35
            x = np.arange(len(reptile_tokens))
            for i, ep in enumerate(epochs):
                sub = df[df['num_epochs'] == ep]
                means = [sub[f'prob_{t}'].mean() for t in reptile_tokens]
                axes[1, 0].bar(x + i * width, means, width, label=f'Epochs={ep}', alpha=0.8)
            axes[1, 0].set_xticks(x + width / 2)
            axes[1, 0].set_xticklabels(reptile_tokens, rotation=45, ha='right')
            axes[1, 0].axhline(0, color='k', ls='--', alpha=0.3)
            axes[1, 0].set_ylabel('Mean ΔP')
            axes[1, 0].set_title('Reptile Tokens by Epochs')
            axes[1, 0].legend()
            for i, ep in enumerate(epochs):
                sub = df[df['num_epochs'] == ep]
                means = [sub[f'prob_{t}'].mean() for t in prog_tokens]
                axes[1, 1].bar(x + i * width, means, width, label=f'Epochs={ep}', alpha=0.8)
            axes[1, 1].set_xticks(x + width / 2)
            axes[1, 1].set_xticklabels(prog_tokens, rotation=45, ha='right')
            axes[1, 1].axhline(0, color='k', ls='--', alpha=0.3)
            axes[1, 1].set_ylabel('Mean ΔP')
            axes[1, 1].set_title('Programming Tokens by Epochs')
            axes[1, 1].legend()
            plt.tight_layout()  # Aggregate by context type
            plt.show()

        def plot_target_word_comparison(self):
            df = pd.DataFrame(self.results)
            df['lr_str'] = df['learning_rate'].apply(lambda x: f'{x:.0e}')
            epochs = sorted(df['num_epochs'].unique())
            tw_labels = []
            for tw in self.target_words:  # Compute acquisition and retention metrics
                tw_labels.append(tw.replace(' ', '_space_') if tw.startswith(' ') else tw)
            target_colors = {'Python': '#e41a1c', 'python': '#377eb8', '_space_python': '#4daf4a'}
            fig, axes = plt.subplots(2, len(epochs), figsize=(5 * len(epochs), 8), sharey='row')
            if len(epochs) == 1:
                axes = axes.reshape(-1, 1)
            for col, ep in enumerate(epochs):
                sub = df[df['num_epochs'] == ep].sort_values('dataset_size')
                for tw_label in tw_labels:
                    display_name = '" python"' if tw_label == '_space_python' else f'"{tw_label}"'
                    color = target_colors.get(tw_label, 'gray')  # Positive = forgetting occurred
                    if f'static_{tw_label}' in sub.columns:
                        for lr_str in sorted(sub['lr_str'].unique()):
                            data = sub[sub['lr_str'] == lr_str]
                            linestyle = '-' if lr_str == '1e-04' else '--' if lr_str == '1e-05' else ':'
                            axes[0, col].plot(data['dataset_size'], data[f'static_{tw_label}'], marker='o', color=color, linestyle=linestyle, linewidth=2, markersize=6, alpha=0.8, label=f'{display_name} LR={lr_str}' if col == 0 else '')
                    if f'contextual_{tw_label}' in sub.columns:
                        for lr_str in sorted(sub['lr_str'].unique()):
                            data = sub[sub['lr_str'] == lr_str]
                            linestyle = '-' if lr_str == '1e-04' else '--' if lr_str == '1e-05' else ':'
                            axes[1, col].plot(data['dataset_size'], data[f'contextual_{tw_label}'], marker='o', color=color, linestyle=linestyle, linewidth=2, markersize=6, alpha=0.8, label=f'{display_name} LR={lr_str}' if col == 0 else '')
                axes[0, col].set_title(f'Epochs = {ep}', fontsize=12, fontweight='bold')
                axes[0, col].axhline(1.0, color='k', ls='--', alpha=0.3)
                axes[1, col].axhline(1.0, color='k', ls='--', alpha=0.3)
                axes[1, col].set_xlabel('Dataset Size')
            axes[0, 0].set_ylabel('Static Embedding Drift\n(cosine similarity)')
            axes[1, 0].set_ylabel('Contextual Embedding Drift\n(cosine similarity)')
            axes[0, 0].legend(loc='lower left', fontsize=7, ncol=1)
            fig.suptitle('Embedding Drift by Target Word Variant', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        def plot_embedding_drift(self):
            df = pd.DataFrame(self.results)  # NEW: Sampled Generation Assessment (Assessment 1)
            df['lr_str'] = df['learning_rate'].apply(lambda x: f'{x:.0e}')
            epochs = sorted(df['num_epochs'].unique())
            colors, markers = (self._get_colors(), self._get_markers())
            metrics = [('static_target_avg', 'Static (avg all variants)'), ('contextual_avg', 'Contextual (avg all variants)'), ('sentence_target', 'Sentence Embedding')]
            fig, axes = plt.subplots(len(metrics), len(epochs), figsize=(5 * len(epochs), 3.5 * len(metrics)), sharey='row', sharex='col')
            if len(epochs) == 1:
                axes = axes.reshape(-1, 1)  # NEW: Polysemy Disambiguation Test (Assessment 2)
            for col, ep in enumerate(epochs):
                sub = df[df['num_epochs'] == ep].sort_values('dataset_size')
                for row, (metric, name) in enumerate(metrics):
                    for lr_str in sorted(sub['lr_str'].unique()):
                        data = sub[sub['lr_str'] == lr_str]
                        axes[row, col].plot(data['dataset_size'], data[metric], marker=markers.get(ep, 'o'), color=colors.get(lr_str, 'gray'), linewidth=2, markersize=8, label=f'LR={lr_str}')
                    axes[row, col].axhline(1.0, color='k', ls='--', alpha=0.3)
                    if col == 0:
                        axes[row, col].set_ylabel(f'{name}\n(cosine sim)')
                axes[0, col].set_title(f'Epochs = {ep}', fontsize=12, fontweight='bold')
                axes[-1, col].set_xlabel('Dataset Size')
            axes[0, -1].legend(loc='lower left', fontsize=9)
            fig.suptitle('Embedding Drift by Training Configuration', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        def plot_perplexity(self):
            df = pd.DataFrame(self.results)
            df['lr_str'] = df['learning_rate'].apply(lambda x: f'{x:.0e}')
            df['ppl_change'] = df['perplexity_ft'] - df['perplexity_base']
            epochs = sorted(df['num_epochs'].unique())
            colors, markers = (self._get_colors(), self._get_markers())
            fig, axes = plt.subplots(1, len(epochs), figsize=(5 * len(epochs), 4), sharey=True)
            if len(epochs) == 1:
                axes = [axes]
            for col, ep in enumerate(epochs):
                sub = df[df['num_epochs'] == ep].sort_values('dataset_size')
                for lr_str in sorted(sub['lr_str'].unique()):
                    data = sub[sub['lr_str'] == lr_str]
                    axes[col].plot(data['dataset_size'], data['ppl_change'], marker=markers.get(ep, 'o'), color=colors.get(lr_str, 'gray'), linewidth=2, markersize=8, label=f'LR={lr_str}')
                axes[col].axhline(0, color='k', ls='--', alpha=0.3)
                axes[col].set_title(f'Epochs = {ep}', fontsize=12, fontweight='bold')
                axes[col].set_xlabel('Dataset Size')
            axes[0].set_ylabel('Perplexity Change (ft - base)')
            axes[-1].legend(loc='upper left', fontsize=9)
            fig.suptitle('Perplexity Drift by Training Configuration', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()  # NEW: Sampled generation metrics
            plt.show()

        def plot_summary(self):
            df = pd.DataFrame(self.results)
            df['lr_str'] = df['learning_rate'].apply(lambda x: f'{x:.0e}')
            best = df.loc[df['prob_reptile_change'].idxmax()]
            colors = self._get_colors()
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # NEW: Polysemy disambiguation metrics
            for ep in df['num_epochs'].unique():
                sub = df[df['num_epochs'] == ep]
                marker = 'o' if ep == 1 else 's'
                axes[0, 0].scatter(sub['prob_reptile_change'], sub['prob_programming_change'], c=[colors.get(lr, 'gray') for lr in sub['lr_str']], s=sub['dataset_size'] / 10, marker=marker, alpha=0.7, edgecolors='white', linewidth=1)
            axes[0, 0].axhline(0, color='k', ls='--', alpha=0.3)
            axes[0, 0].axvline(0, color='k', ls='--', alpha=0.3)
            axes[0, 0].set_xlabel('ΔP(reptile)')
            axes[0, 0].set_ylabel('ΔP(programming)')
            axes[0, 0].set_title('Semantic Trade-off\n(size=dataset, shape=epochs, color=LR)')
            tw_labels = [tw.replace(' ', '_space_') if tw.startswith(' ') else tw for tw in self.target_words]  # NEW: Store full results
            static_vals = [best.get(f'static_{tw}', np.nan) for tw in tw_labels]  # NEW: Store full results
            ctx_vals = [best.get(f'contextual_{tw}', np.nan) for tw in tw_labels]
            x = np.arange(len(tw_labels))
            width = 0.35
            axes[0, 1].bar(x - width / 2, static_vals, width, label='Static', color='#e41a1c')
            axes[0, 1].bar(x + width / 2, ctx_vals, width, label='Contextual', color='#377eb8')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels([f'''"{tw.replace('_space_', ' ')}"''' for tw in tw_labels])
            axes[0, 1].axhline(1.0, color='k', ls='--')
            axes[0, 1].set_ylabel('Cosine Similarity')
            axes[0, 1].set_title('Best Config: Drift by Token Variant')
            axes[0, 1].legend()
            axes[0, 1].set_ylim(0, 1.1)
            heatmap_data = df.pivot_table(values='prob_reptile_change', index='learning_rate', columns=['num_epochs', 'dataset_size'], aggfunc='mean')
            im = axes[1, 0].imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto')
            axes[1, 0].set_yticks(range(len(heatmap_data.index)))
            axes[1, 0].set_yticklabels([f'{lr:.0e}' for lr in heatmap_data.index])
            axes[1, 0].set_xticks(range(len(heatmap_data.columns)))
            axes[1, 0].set_xticklabels([f'E{e}\nD{d}' for e, d in heatmap_data.columns], fontsize=8)
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('ΔP(reptile) Heatmap')
            plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
            axes[1, 1].axis('off')
            txt = f"Best Config:\n\nLR: {best['learning_rate']:.0e}\nEpochs: {best['num_epochs']}\nDataset: {best['dataset_size']}\n\n"
            txt = txt + f"Reptile Δ: {best['prob_reptile_change']:+.4f}\nProgramming Δ: {best['prob_programming_change']:+.4f}\n\n"
            txt = txt + 'Static Drift per variant:\n'
            for tw in tw_labels:
                val = best.get(f'static_{tw}', np.nan)
                txt = txt + f'''  "{tw.replace('_space_', ' ')}": {val:.4f}\n'''
            axes[1, 1].text(0.1, 0.5, txt, fontsize=11, family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), transform=axes[1, 1].transAxes, verticalalignment='center')
            plt.tight_layout()
            plt.show()

        def plot_heatmaps(self):
            df = pd.DataFrame(self.results)
            metrics = [('prob_reptile_change', 'ΔP(reptile)', 'RdYlGn'), ('prob_programming_change', 'ΔP(programming)', 'RdYlGn_r'), ('static_target_avg', 'Static Drift (avg)', 'RdYlGn'), ('perplexity_ft', 'Perplexity (ft)', 'RdYlGn_r')]
            epochs = sorted(df['num_epochs'].unique())
            fig, axes = plt.subplots(len(metrics), len(epochs), figsize=(4 * len(epochs), 3 * len(metrics)))
            if len(epochs) == 1:
                axes = axes.reshape(-1, 1)
            for col, ep in enumerate(epochs):
                sub = df[df['num_epochs'] == ep]
                for row, (metric, title, cmap) in enumerate(metrics):
                    pivot = sub.pivot_table(values=metric, index='learning_rate', columns='dataset_size', aggfunc='mean')
                    im = axes[row, col].imshow(pivot.values, cmap=cmap, aspect='auto')
                    axes[row, col].set_yticks(range(len(pivot.index)))
                    axes[row, col].set_yticklabels([f'{lr:.0e}' for lr in pivot.index])
                    axes[row, col].set_xticks(range(len(pivot.columns)))
                    axes[row, col].set_xticklabels(pivot.columns)
                    if col == 0:
                        axes[row, col].set_ylabel(f'{title}\nLearning Rate')
                    if row == 0:
                        axes[row, col].set_title(f'Epochs = {ep}', fontweight='bold')
                    if row == len(metrics) - 1:
                        axes[row, col].set_xlabel('Dataset Size')
                    plt.colorbar(im, ax=axes[row, col], shrink=0.8)
                    for i in range(len(pivot.index)):
                        for j in range(len(pivot.columns)):
                            val = pivot.values[i, j]
                            axes[row, col].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color='white' if abs(val) > 0.5 else 'black')
            fig.suptitle('Hyperparameter Grid Results', fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.show()

        def plot_sampled_generation(self):
            """
            Plot Assessment 1: Sampled Generation Assessment Results
        
            Shows how the probability of generating reptile vs programming content
            changes across 100 sampled completions per prompt.
            """
            df = pd.DataFrame(self.results)
            df['lr_str'] = df['learning_rate'].apply(lambda x: f'{x:.0e}')
            epochs = sorted(df['num_epochs'].unique())
            colors = self._get_colors()
            fig, axes = plt.subplots(2, len(epochs), figsize=(5 * len(epochs), 8), sharey='row')
            if len(epochs) == 1:
                axes = axes.reshape(-1, 1)
            for col, ep in enumerate(epochs):
                sub = df[df['num_epochs'] == ep].sort_values('dataset_size')
                for lr_str in sorted(sub['lr_str'].unique()):
                    data = sub[sub['lr_str'] == lr_str]
                    axes[0, col].plot(data['dataset_size'], data['sampled_reptile_change'], marker='o', color=colors.get(lr_str, 'gray'), linewidth=2, markersize=8, label=f'LR={lr_str}')
                    axes[1, col].plot(data['dataset_size'], data['sampled_programming_change'], marker='s', color=colors.get(lr_str, 'gray'), linewidth=2, markersize=8, label=f'LR={lr_str}')
                axes[0, col].set_title(f'Epochs = {ep}', fontsize=12, fontweight='bold')
                axes[0, col].axhline(0, color='k', ls='--', alpha=0.3)
                axes[1, col].axhline(0, color='k', ls='--', alpha=0.3)
                axes[1, col].set_xlabel('Dataset Size')
            axes[0, 0].set_ylabel('ΔP(reptile words in generation)\n(100 samples, k=10 tokens)')
            axes[1, 0].set_ylabel('ΔP(programming words in generation)\n(100 samples, k=10 tokens)')
            axes[0, -1].legend(loc='upper left', fontsize=9)
            fig.suptitle('Sampled Generation Assessment: Semantic Drift in Generated Text', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        def plot_polysemy_disambiguation(self):
            """
            Plot Assessment 2: Polysemy Disambiguation Test Results
        
            Shows:
            - Reptile context acquisition (can the model now generate reptile content?)
            - Programming context retention (did the model forget programming knowledge?)
            - Net semantic shift (acquisition - forgetting)
            """
            df = pd.DataFrame(self.results)
            df['lr_str'] = df['learning_rate'].apply(lambda x: f'{x:.0e}')
            epochs = sorted(df['num_epochs'].unique())
            colors = self._get_colors()
            fig, axes = plt.subplots(3, len(epochs), figsize=(5 * len(epochs), 10), sharey='row')
            if len(epochs) == 1:
                axes = axes.reshape(-1, 1)
            for col, ep in enumerate(epochs):
                sub = df[df['num_epochs'] == ep].sort_values('dataset_size')
                for lr_str in sorted(sub['lr_str'].unique()):
                    data = sub[sub['lr_str'] == lr_str]
                    axes[0, col].plot(data['dataset_size'], data['polysemy_reptile_acquisition'], marker='o', color=colors.get(lr_str, 'gray'), linewidth=2, markersize=8, label=f'LR={lr_str}')
                    axes[1, col].plot(data['dataset_size'], data['polysemy_programming_forgetting'], marker='s', color=colors.get(lr_str, 'gray'), linewidth=2, markersize=8, label=f'LR={lr_str}')
                    axes[2, col].plot(data['dataset_size'], data['polysemy_net_semantic_shift'], marker='^', color=colors.get(lr_str, 'gray'), linewidth=2, markersize=8, label=f'LR={lr_str}')
                axes[0, col].set_title(f'Epochs = {ep}', fontsize=12, fontweight='bold')
                for ax in axes[:, col]:
                    ax.axhline(0, color='k', ls='--', alpha=0.3)
                axes[2, col].set_xlabel('Dataset Size')
            axes[0, 0].set_ylabel('Reptile Acquisition\n(↑ = better reptile context)')
            axes[1, 0].set_ylabel('Programming Forgetting\n(↑ = more forgetting)')
            axes[2, 0].set_ylabel('Net Semantic Shift\n(acquisition - forgetting)')
            axes[0, -1].legend(loc='upper left', fontsize=9)
            fig.suptitle('Polysemy Disambiguation: Context-Aware Generation Quality', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        def plot_acquisition_vs_forgetting(self):
            """
            Scatter plot showing the trade-off between learning new semantics
            and forgetting old semantics.
        
            Ideal: High acquisition (right), Low forgetting (bottom)
            """
            df = pd.DataFrame(self.results)
            df['lr_str'] = df['learning_rate'].apply(lambda x: f'{x:.0e}')
            colors = self._get_colors()
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for ep in df['num_epochs'].unique():
                sub = df[df['num_epochs'] == ep]
                marker = 'o' if ep <= 1 else 's' if ep <= 3 else '^'
                for lr_str in sub['lr_str'].unique():
                    data = sub[sub['lr_str'] == lr_str]
                    axes[0].scatter(data['sampled_reptile_change'], -data['sampled_programming_change'], c=colors.get(lr_str, 'gray'), s=data['dataset_size'] / 10, marker=marker, alpha=0.7, edgecolors='white', linewidth=1, label=f'E={ep}, LR={lr_str}')
            axes[0].axhline(0, color='k', ls='--', alpha=0.3)
            axes[0].axvline(0, color='k', ls='--', alpha=0.3)
            axes[0].set_xlabel('Reptile Acquisition (ΔP in generated text)')
            axes[0].set_ylabel('Programming Retention (negative ΔP)')
            axes[0].set_title('Sampled Generation: Acquisition vs Retention\n(size=dataset size)')
            axes[0].text(0.95, 0.95, 'IDEAL\n(learn + retain)', transform=axes[0].transAxes, ha='right', va='top', fontsize=9, color='green', fontweight='bold')
            axes[0].text(0.05, 0.05, 'WORST\n(no learn + forget)', transform=axes[0].transAxes, ha='left', va='bottom', fontsize=9, color='red', fontweight='bold')
            for ep in df['num_epochs'].unique():
                sub = df[df['num_epochs'] == ep]
                marker = 'o' if ep <= 1 else 's' if ep <= 3 else '^'
                for lr_str in sub['lr_str'].unique():
                    data = sub[sub['lr_str'] == lr_str]
                    axes[1].scatter(data['polysemy_reptile_acquisition'], -data['polysemy_programming_forgetting'], c=colors.get(lr_str, 'gray'), s=data['dataset_size'] / 10, marker=marker, alpha=0.7, edgecolors='white', linewidth=1)
            axes[1].axhline(0, color='k', ls='--', alpha=0.3)
            axes[1].axvline(0, color='k', ls='--', alpha=0.3)
            axes[1].set_xlabel('Reptile Acquisition (polysemy discrimination)')
            axes[1].set_ylabel('Programming Retention (negative forgetting)')
            axes[1].set_title('Polysemy Test: Acquisition vs Retention\n(size=dataset size)')
            axes[1].text(0.95, 0.95, 'IDEAL', transform=axes[1].transAxes, ha='right', va='top', fontsize=9, color='green', fontweight='bold')
            axes[1].text(0.05, 0.05, 'WORST', transform=axes[1].transAxes, ha='left', va='bottom', fontsize=9, color='red', fontweight='bold')
            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), loc='center right', fontsize=8)
            plt.tight_layout()
            plt.subplots_adjust(right=0.85)
            plt.show()

        def plot_detailed_polysemy_cases(self, config_key: str=None):
            """
            Show detailed results for individual polysemy test cases.
            If config_key is None, uses the best performing configuration.
            """
            df = pd.DataFrame(self.results)
            if config_key is None:
                best_idx = df['polysemy_net_semantic_shift'].idxmax()
                result = df.iloc[best_idx]
            else:
                result = df[df.apply(lambda r: self._config_key(r) == config_key, axis=1)].iloc[0]
            polysemy_data = result['polysemy_all']
            test_cases = polysemy_data['test_cases']
            reptile_cases = [tc for tc in test_cases if tc['context'] == 'reptile']
            prog_cases = [tc for tc in test_cases if tc['context'] == 'programming']
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            prompts = [tc['prompt'][:40] + '...' if len(tc['prompt']) > 40 else tc['prompt'] for tc in reptile_cases]
            base_rates = [tc['base_expected_rate'] for tc in reptile_cases]
            ft_rates = [tc['ft_expected_rate'] for tc in reptile_cases]
            x = np.arange(len(prompts))
            width = 0.35
            axes[0, 0].barh(x - width / 2, base_rates, width, label='Base Model', color='#3498db', alpha=0.8)
            axes[0, 0].barh(x + width / 2, ft_rates, width, label='Fine-tuned', color='#2ecc71', alpha=0.8)
            axes[0, 0].set_yticks(x)
            axes[0, 0].set_yticklabels(prompts, fontsize=8)
            axes[0, 0].set_xlabel('Expected Word Rate in Completions')
            axes[0, 0].set_title('Reptile Context: Expected Word Generation Rate')
            axes[0, 0].legend(loc='lower right')
            axes[0, 0].set_xlim(0, 1)
            prompts = [tc['prompt'][:40] + '...' if len(tc['prompt']) > 40 else tc['prompt'] for tc in prog_cases]
            base_rates = [tc['base_expected_rate'] for tc in prog_cases]
            ft_rates = [tc['ft_expected_rate'] for tc in prog_cases]
            x = np.arange(len(prompts))
            axes[0, 1].barh(x - width / 2, base_rates, width, label='Base Model', color='#3498db', alpha=0.8)
            axes[0, 1].barh(x + width / 2, ft_rates, width, label='Fine-tuned', color='#e74c3c', alpha=0.8)
            axes[0, 1].set_yticks(x)
            axes[0, 1].set_yticklabels(prompts, fontsize=8)
            axes[0, 1].set_xlabel('Expected Word Rate in Completions')
            axes[0, 1].set_title('Programming Context: Knowledge Retention Check')
            axes[0, 1].legend(loc='lower right')
            axes[0, 1].set_xlim(0, 1)
            all_cases = reptile_cases + prog_cases
            prompts = [tc['prompt'][:30] + '...' if len(tc['prompt']) > 30 else tc['prompt'] for tc in all_cases]
            base_disc = [tc['base_discrimination'] for tc in all_cases]
            ft_disc = [tc['ft_discrimination'] for tc in all_cases]
            colors_bars = ['#2ecc71' if tc['context'] == 'reptile' else '#e74c3c' for tc in all_cases]
            x = np.arange(len(prompts))
            axes[1, 0].bar(x - width / 2, base_disc, width, label='Base', color='#3498db', alpha=0.6)
            axes[1, 0].bar(x + width / 2, ft_disc, width, label='Fine-tuned', color=colors_bars, alpha=0.8)
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(prompts, rotation=45, ha='right', fontsize=7)
            axes[1, 0].axhline(0, color='k', ls='--', alpha=0.3)
            axes[1, 0].set_ylabel('Discrimination Score\n(expected - unexpected rate)')
            axes[1, 0].set_title('Context Discrimination: Base vs Fine-tuned')
            axes[1, 0].legend()
            axes[1, 1].axis('off')
            config_str = f"LR: {result['learning_rate']:.0e}, Epochs: {result['num_epochs']}, Dataset: {result['dataset_size']}"
            summary = polysemy_data['summary']
            txt = f'Configuration: {config_str}\n\n'
            txt = txt + ('=' * 40 + '\n')
            txt = txt + 'POLYSEMY DISAMBIGUATION SUMMARY\n'
            txt = txt + ('=' * 40 + '\n\n')
            txt = txt + f"Reptile Acquisition Score: {summary['reptile_acquisition']:+.3f}\n"
            txt = txt + f'  (Positive = learned reptile context)\n\n'
            txt = txt + f"Programming Forgetting Score: {summary['programming_forgetting']:+.3f}\n"
            txt = txt + f'  (Positive = forgot programming knowledge)\n\n'
            txt = txt + f"Net Semantic Shift: {summary['net_semantic_shift']:+.3f}\n"
            txt = txt + f'  (Positive = good learning with minimal forgetting)\n\n'
            txt = txt + ('=' * 40 + '\n')
            txt = txt + 'INTERPRETATION:\n'
            if summary['reptile_acquisition'] > 0.1:
                txt = txt + '✓ Model learned reptile context well\n'
            else:
                txt = txt + '✗ Limited reptile context acquisition\n'
            if summary['programming_forgetting'] < 0.1:
                txt = txt + '✓ Programming knowledge retained\n'
            else:
                txt = txt + '⚠ Some programming knowledge forgotten\n'
            axes[1, 1].text(0.1, 0.5, txt, fontsize=10, family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), transform=axes[1, 1].transAxes, verticalalignment='center')
            fig.suptitle('Detailed Polysemy Disambiguation Analysis', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        def print_sample_generations(self, config_key: str=None, num_prompts: int=3):
            """
            Print example generated completions from the sampled generation assessment.
            Useful for qualitative analysis of semantic drift.
            """
            df = pd.DataFrame(self.results)
            if config_key is None:
                best_idx = df['polysemy_net_semantic_shift'].idxmax()
                result = df.iloc[best_idx]
            else:
                result = df[df.apply(lambda r: self._config_key(r) == config_key, axis=1)].iloc[0]
            print('=' * 70)
            print('SAMPLE GENERATIONS COMPARISON')
            print(f"Config: LR={result['learning_rate']:.0e}, Epochs={result['num_epochs']}, DS={result['dataset_size']}")
            print('=' * 70)
            sampled_data = result['sampled_generation_all']
            prompts = list(sampled_data['prompts'].keys())[:num_prompts]
            for prompt in prompts:
                data = sampled_data['prompts'][prompt]
                print(f'\nPrompt: "{prompt}"')
                print('-' * 50)
                print('BASE MODEL completions:')
                for i, comp in enumerate(data['sample_completions_base'][:3], 1):
                    print(f'  {i}. ...{comp}')
                print('\nFINE-TUNED MODEL completions:')
                for i, comp in enumerate(data['sample_completions_ft'][:3], 1):
                    print(f'  {i}. ...{comp}')
                print(f'\nMetrics:')
                print(f"  Reptile word prob: {data['base_reptile_prob']:.2%} → {data['ft_reptile_prob']:.2%} (Δ={data['reptile_prob_change']:+.2%})")
                print(f"  Programming word prob: {data['base_programming_prob']:.2%} → {data['ft_programming_prob']:.2%} (Δ={data['programming_prob_change']:+.2%})")
            print('\n' + '=' * 70)
            print('POLYSEMY DISAMBIGUATION EXAMPLES')
            print('=' * 70)
            polysemy_data = result['polysemy_all']
            for tc in polysemy_data['test_cases'][:4]:
                print(f'''\nPrompt ({tc['context']} context): "{tc['prompt']}"''')
                print('-' * 50)
                print('BASE MODEL:')
                for i, comp in enumerate(tc['sample_completions_base'][:2], 1):
                    print(f'  {i}. ...{comp}')
                print('FINE-TUNED:')
                for i, comp in enumerate(tc['sample_completions_ft'][:2], 1):
                    print(f'  {i}. ...{comp}')
                print(f"  Discrimination change: {tc['discrimination_change']:+.3f}")

        def get_best_model(self):
            df = pd.DataFrame(self.results)
            best = df.loc[df['prob_reptile_change'].idxmax()]
            key = self._config_key(best.to_dict())
            return (self.models.get(key), best)

        def summary(self):
            df = pd.DataFrame(self.results)
            print('=' * 60)
            print('EXPERIMENT SUMMARY')
            print('=' * 60)
            print(f'\nTarget words analyzed: {self.target_words}')
            print(f'Configs tested: {len(df)}')
            print(f'\nBest reptile acquisition:')
            best = df.loc[df['prob_reptile_change'].idxmax()]
            print(f"  LR={best['learning_rate']:.0e}, Epochs={best['num_epochs']}, DS={best['dataset_size']}")
            print(f"  Reptile Δ: {best['prob_reptile_change']:+.4f}")
            print(f"  Programming Δ: {best['prob_programming_change']:+.4f}")
            print(f'\nStatic drift per target word (best config):')
            for tw in self.target_words:
                tw_label = tw.replace(' ', '_space_') if tw.startswith(' ') else tw
                val = best.get(f'static_{tw_label}', np.nan)
                print(f'  "{tw}": {val:.4f}')
            print(f'\nAverage metrics:')
            print(f"  Static drift (avg): {df['static_target_avg'].mean():.4f}")
            print(f"  Contextual drift (avg): {df['contextual_avg'].mean():.4f}")
            print(f"  Perplexity change: {(df['perplexity_ft'] - df['perplexity_base']).mean():+.2f}")
            print(f'\n--- Sampled Generation Assessment (100 samples, k=10 tokens) ---')
            print(f"  Mean reptile word appearance: {df['sampled_reptile_prob_base'].mean():.2%} → {df['sampled_reptile_prob_ft'].mean():.2%}")
            print(f"  Mean programming word appearance: {df['sampled_programming_prob_base'].mean():.2%} → {df['sampled_programming_prob_ft'].mean():.2%}")
            print(f'\n--- Polysemy Disambiguation Test ---')
            print(f"  Mean reptile acquisition: {df['polysemy_reptile_acquisition'].mean():+.3f}")
            print(f"  Mean programming forgetting: {df['polysemy_programming_forgetting'].mean():+.3f}")
            print(f"  Mean net semantic shift: {df['polysemy_net_semantic_shift'].mean():+.3f}")
            best_shift = df.loc[df['polysemy_net_semantic_shift'].idxmax()]
            print(f'\nBest net semantic shift:')
            print(f"  LR={best_shift['learning_rate']:.0e}, Epochs={best_shift['num_epochs']}, DS={best_shift['dataset_size']}")
            print(f"  Acquisition: {best_shift['polysemy_reptile_acquisition']:+.3f}")
            print(f"  Forgetting: {best_shift['polysemy_programming_forgetting']:+.3f}")
            print(f"  Net shift: {best_shift['polysemy_net_semantic_shift']:+.3f}")
            cols = ['learning_rate', 'num_epochs', 'dataset_size', 'static_target_avg', 'contextual_avg', 'prob_reptile_change', 'prob_programming_change', 'sampled_reptile_change', 'sampled_programming_change', 'polysemy_reptile_acquisition', 'polysemy_programming_forgetting', 'polysemy_net_semantic_shift']
            return df[cols]

    class EmbeddingEvolutionTracker:
        """
        Tracks the evolution of contextual and sentence embeddings during fine-tuning.
    
        This class allows you to:
        1. Define sentences for tracking (python-programming, python-snake, controls)
        2. Fine-tune with checkpointing at regular intervals
        3. Collect embeddings at each checkpoint
        4. Visualize embedding trajectories in 2D space
        """
        DEFAULT_TRACKING_SENTENCES = {'python_programming': ['Python is a powerful programming language.', 'I wrote my first Python script yesterday.', 'Python developers use pip to install packages.', 'The Python code runs efficiently on this server.', 'Learning Python takes practice and dedication.'], 'python_snake': ['The python slithered across the forest floor.', 'A python coiled around the tree branch.', 'The python swallowed its prey whole.', 'Pythons are large constrictor snakes.', 'The python basked in the warm sunlight.'], 'control_programming': ['JavaScript runs in web browsers.', 'The computer processed the data quickly.', 'Software engineers write clean code.', 'The database stores user information.', 'Machine learning models need training data.'], 'control_reptile': ['The cobra raised its hood defensively.', 'Snakes are cold-blooded reptiles.', 'The lizard crawled across the warm rock.', 'Reptiles lay eggs in sandy nests.', 'The boa constrictor lives in the jungle.'], 'ambiguous': ['Python is very popular these days.', 'I learned about Python in school.', 'Python can be quite powerful.', 'Many people work with Python daily.', 'Python has grown significantly over the years.']}

        def __init__(self, base_model, tokenizer, tracking_sentences=None, seed=SEED):
            """
            Initialize the embedding evolution tracker.
        
            Args:
                base_model: The base GPT2 model  # Plot reptile probability change from sampled generation
                tokenizer: The tokenizer
                tracking_sentences: Dict of category -> list of sentences. Uses defaults if None.
                seed: Random seed for reproducibility
            """  # Plot programming probability change (forgetting indicator)
            self.base_model = base_model
            self.tokenizer = tokenizer
            self.seed = seed
            self.tracking_sentences = tracking_sentences or self.DEFAULT_TRACKING_SENTENCES
            self.evolution_data = []
            self.configs_run = []

        def _extract_all_embeddings(self, model):
            """Extract contextual and sentence embeddings for all tracking sentences."""
            embeddings = {}
            for category, sentences in self.tracking_sentences.items():
                embeddings[category] = {'contextual': [], 'sentence': [], 'sentences': sentences}
                for sentence in sentences:
                    sent_emb = get_sentence_embedding(sentence, model, self.tokenizer)
                    embeddings[category]['sentence'].append(sent_emb)
                    if 'python' in sentence.lower():
                        ctx_emb = get_contextual_embedding(sentence, 'python', model, self.tokenizer)
                    elif 'Python' in sentence:
                        ctx_emb = get_contextual_embedding(sentence, 'Python', model, self.tokenizer)
                    else:
                        ctx_emb = sent_emb
                    embeddings[category]['contextual'].append(ctx_emb)
            return embeddings

        def _fine_tune_with_tracking(self, dataset, config, checkpoint_steps=25):
            """
            Fine-tune with embedding tracking at regular checkpoint intervals.
        
            Args:
                dataset: The training dataset
                config: Dict with 'learning_rate', 'num_epochs', 'dataset_size'
                checkpoint_steps: Extract embeddings every N steps
            
            Returns:
                Tuple of (final_model, evolution_records)
            """
            from transformers import TrainerCallback
            n_samples = min(config['dataset_size'], len(dataset))
            train_data = prepare_dataset(dataset.select(range(n_samples)), tokenizer=self.tokenizer)
            batch_size = 8
            total_samples = len(train_data)  # Reptile acquisition score
            steps_per_epoch = max(1, total_samples // batch_size)
            total_steps = steps_per_epoch * config['num_epochs']
            evolution_records = []
            config_key = f"lr{config['learning_rate']:.0e}_ep{config['num_epochs']}_ds{config['dataset_size']}"  # Programming forgetting (positive = forgetting)
            initial_embeddings = self._extract_all_embeddings(self.base_model)
            evolution_records.append({'config': config.copy(), 'config_key': config_key, 'step': 0, 'progress': 0.0, 'embeddings': initial_embeddings})
            ft_model = deepcopy(self.base_model)
            tracker_self = self  # Net semantic shift

            class EmbeddingTrackingCallback(TrainerCallback):

                def __init__(self):
                    self.last_tracked_step = 0

                def on_step_end(self, args, state, control, model=None, **kwargs):
                    if model is None:
                        return
                    current_step = state.global_step
                    if current_step > 0 and current_step - self.last_tracked_step >= checkpoint_steps:
                        self.last_tracked_step = current_step
                        progress = current_step / max(1, total_steps)
                        model.eval()
                        embeddings = tracker_self._extract_all_embeddings(model)
                        model.train()
                        evolution_records.append({'config': config.copy(), 'config_key': config_key, 'step': current_step, 'progress': progress, 'embeddings': embeddings})

                def on_train_end(self, args, state, control, model=None, **kwargs):
                    if model is not None:
                        model.eval()
                        embeddings = tracker_self._extract_all_embeddings(model)
                        evolution_records.append({'config': config.copy(), 'config_key': config_key, 'step': state.global_step, 'progress': 1.0, 'embeddings': embeddings, 'is_final': True})
            args = TrainingArguments(output_dir=FINE_TUNED_MODEL_DIR, overwrite_output_dir=True, num_train_epochs=config['num_epochs'], per_device_train_batch_size=batch_size, learning_rate=config['learning_rate'], weight_decay=0.01, logging_steps=10, save_strategy='no', report_to='none', fp16=torch.cuda.is_available())
            trainer = Trainer(model=ft_model, args=args, train_dataset=train_data, data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False), callbacks=[EmbeddingTrackingCallback()])
            print(f'Fine-tuning with tracking: {config_key}')
            print(f'  Total steps: ~{total_steps}, Checkpoint every {checkpoint_steps} steps')
            trainer.train()
            return (ft_model, evolution_records)

        def run_evolution_tracking(self, dataset, configs, checkpoint_steps=25):
            """
            Run fine-tuning with embedding evolution tracking for multiple configs.  # Left: Sampled generation based metrics
        
            Args:
                dataset: The training dataset
                configs: List of config dicts, each with 'learning_rate', 'num_epochs', 'dataset_size'
                checkpoint_steps: Extract embeddings every N steps
            """
            self.evolution_data = []  # Negate so up = good retention
            self.configs_run = configs
            for i, config in enumerate(configs):
                print(f'\n[{i + 1}/{len(configs)}] Running evolution tracking for {config}')
                _seed_all(self.seed)
                _, evolution_records = self._fine_tune_with_tracking(dataset, config, checkpoint_steps)
                self.evolution_data.extend(evolution_records)
                torch.cuda.empty_cache()
            print(f'\nTracking complete. Collected {len(self.evolution_data)} embedding snapshots.')
            return self.evolution_data

        def _prepare_evolution_dataframe(self, embedding_type='sentence'):
            """
            Prepare a DataFrame with all embeddings for visualization.  # Add quadrant labels
        
            Args:
                embedding_type: 'sentence' or 'contextual'
            
            Returns:
                DataFrame with columns: config_key, step, progress, category, sentence_idx,  # Right: Polysemy discrimination based metrics
                                        sentence, embedding (as array), lr, epochs, ds
            """
            records = []
            for snapshot in self.evolution_data:
                config = snapshot['config']
                config_key = snapshot['config_key']
                step = snapshot['step']  # Negate so up = good retention
                progress = snapshot['progress']
                for category, data in snapshot['embeddings'].items():
                    emb_list = data[embedding_type]
                    sentences = data['sentences']
                    for idx, (emb, sent) in enumerate(zip(emb_list, sentences)):
                        records.append({'config_key': config_key, 'step': step, 'progress': progress, 'category': category, 'sentence_idx': idx, 'sentence': sent[:50] + '...' if len(sent) > 50 else sent, 'embedding': emb, 'lr': config['learning_rate'], 'epochs': config['num_epochs'], 'dataset_size': config['dataset_size']})
            return pd.DataFrame(records)

        def plot_embedding_evolution(self, embedding_type='sentence', method='pca', show_trajectories=True, selected_configs=None):
            """
            Plot 2D visualization of embedding evolution during fine-tuning.
      # Add quadrant labels
            Args:
                embedding_type: 'sentence' or 'contextual'
                method: 'pca' or 'tsne' for dimensionality reduction
                show_trajectories: Whether to draw lines connecting evolution steps
                selected_configs: List of config_keys to show. If None, shows all.
            """  # Create legend
            df = self._prepare_evolution_dataframe(embedding_type)
            if selected_configs:  # Remove duplicates
                df = df[df['config_key'].isin(selected_configs)]
            all_embeddings = np.vstack(df['embedding'].values)
            if method.lower() == 'pca':
                reducer = PCA(n_components=2, random_state=self.seed)
            else:
                perp = min(30, len(all_embeddings) - 1)
                reducer = TSNE(n_components=2, perplexity=perp, random_state=self.seed, init='pca', learning_rate='auto')
            embeddings_2d = reducer.fit_transform(all_embeddings)
            df['x'] = embeddings_2d[:, 0]
            df['y'] = embeddings_2d[:, 1]
            category_colors = {'python_programming': '#e41a1c', 'python_snake': '#377eb8', 'control_programming': '#ff7f00', 'control_reptile': '#4daf4a', 'ambiguous': '#984ea3'}
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            ax1 = axes[0]
            for category, color in category_colors.items():
                cat_df = df[df['category'] == category]
                sizes = 20 + 80 * cat_df['progress']  # Find best config based on net semantic shift
                alphas = 0.3 + 0.5 * cat_df['progress']
                ax1.scatter(cat_df['x'], cat_df['y'], c=color, s=sizes, alpha=0.6, label=category.replace('_', ' ').title(), edgecolors='white', linewidth=0.5)
            if show_trajectories:
                for (config_key, category, sent_idx), group in df.groupby(['config_key', 'category', 'sentence_idx']):
                    group = group.sort_values('step')
                    if len(group) > 1:
                        color = category_colors.get(category, 'gray')
                        ax1.plot(group['x'], group['y'], color=color, alpha=0.3, linewidth=1)
            ax1.set_xlabel(f'{method.upper()} Component 1')  # Separate by context
            ax1.set_ylabel(f'{method.upper()} Component 2')
            ax1.set_title(f'{embedding_type.title()} Embedding Evolution by Category\n(Size/opacity = training progress)')
            ax1.legend(loc='upper right', fontsize=9)
            ax2 = axes[1]
            unique_configs = df['config_key'].unique()
            config_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_configs)))  # Top-left: Reptile context test cases
            config_color_map = {cfg: config_colors[i] for i, cfg in enumerate(unique_configs)}
            for config_key, cfg_color in config_color_map.items():
                cfg_df = df[df['config_key'] == config_key]
                sizes = 20 + 80 * cfg_df['progress']
                ax2.scatter(cfg_df['x'], cfg_df['y'], c=[cfg_color], s=sizes, alpha=0.6, label=config_key, edgecolors='white', linewidth=0.5)
                if show_trajectories:
                    for (category, sent_idx), group in cfg_df.groupby(['category', 'sentence_idx']):
                        group = group.sort_values('step')
                        if len(group) > 1:
                            ax2.plot(group['x'], group['y'], color=cfg_color, alpha=0.4, linewidth=1)
            ax2.set_xlabel(f'{method.upper()} Component 1')
            ax2.set_ylabel(f'{method.upper()} Component 2')
            ax2.set_title(f'{embedding_type.title()} Embedding Evolution by Config\n(Size = training progress)')
            ax2.legend(loc='upper right', fontsize=8, ncol=1)
            plt.tight_layout()
            plt.show()  # Top-right: Programming context test cases (forgetting check)

        def plot_category_trajectories(self, embedding_type='sentence', method='pca'):
            """
            Plot embedding trajectories with clear start/end markers, one subplot per category.
            """
            df = self._prepare_evolution_dataframe(embedding_type)
            all_embeddings = np.vstack(df['embedding'].values)
            if method.lower() == 'pca':
                reducer = PCA(n_components=2, random_state=self.seed)
            else:
                perp = min(30, len(all_embeddings) - 1)
                reducer = TSNE(n_components=2, perplexity=perp, random_state=self.seed, init='pca', learning_rate='auto')
            embeddings_2d = reducer.fit_transform(all_embeddings)
            df['x'] = embeddings_2d[:, 0]
            df['y'] = embeddings_2d[:, 1]  # Bottom-left: Discrimination scores comparison
            categories = list(self.tracking_sentences.keys())
            n_cats = len(categories)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            unique_configs = df['config_key'].unique()
            config_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_configs)))
            config_color_map = {cfg: config_colors[i] for i, cfg in enumerate(unique_configs)}
            for idx, category in enumerate(categories):
                ax = axes[idx]
                cat_df = df[df['category'] == category]
                for config_key, cfg_color in config_color_map.items():
                    cfg_cat_df = cat_df[cat_df['config_key'] == config_key]
                    for sent_idx in cfg_cat_df['sentence_idx'].unique():
                        traj = cfg_cat_df[cfg_cat_df['sentence_idx'] == sent_idx].sort_values('step')
                        if len(traj) > 0:
                            ax.plot(traj['x'], traj['y'], color=cfg_color, alpha=0.6, linewidth=1.5)
                            ax.scatter(traj.iloc[0]['x'], traj.iloc[0]['y'], c=[cfg_color], s=60, marker='o', edgecolors='black', linewidth=1, zorder=5)  # Bottom-right: Summary text
                            ax.scatter(traj.iloc[-1]['x'], traj.iloc[-1]['y'], c=[cfg_color], s=120, marker='*', edgecolors='black', linewidth=1, zorder=5)
                ax.set_title(category.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.set_xlabel(f'{method.upper()} 1')
                ax.set_ylabel(f'{method.upper()} 2')
            ax_legend = axes[-1]
            ax_legend.axis('off')
            legend_elements = []
            for config_key, cfg_color in config_color_map.items():
                legend_elements.append(plt.Line2D([0], [0], color=cfg_color, linewidth=2, label=config_key))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='gray', label='Start (step 0)', markerfacecolor='gray', markersize=10, linestyle='None'))
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='gray', label='End (final)', markerfacecolor='gray', markersize=15, linestyle='None'))
            ax_legend.legend(handles=legend_elements, loc='center', fontsize=10, title='Configurations')
            ax_legend.set_title('Legend', fontsize=12, fontweight='bold')
            fig.suptitle(f'{embedding_type.title()} Embedding Trajectories During Fine-Tuning', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        def plot_drift_over_time(self, embedding_type='sentence'):
            """
            Plot how much embeddings drift from their initial position over training steps.
            """
            df = self._prepare_evolution_dataframe(embedding_type)
            drift_records = []
            for (config_key, category, sent_idx), group in df.groupby(['config_key', 'category', 'sentence_idx']):
                group = group.sort_values('step')
                initial_emb = group.iloc[0]['embedding']
                for _, row in group.iterrows():
                    drift = 1 - cosine_similarity(initial_emb, row['embedding'])
                    drift_records.append({'config_key': config_key, 'category': category, 'sentence_idx': sent_idx, 'step': row['step'], 'progress': row['progress'], 'drift': drift, 'lr': row['lr'], 'epochs': row['epochs'], 'dataset_size': row['dataset_size']})
            drift_df = pd.DataFrame(drift_records)
            agg_df = drift_df.groupby(['config_key', 'category', 'progress']).agg({'drift': 'mean', 'lr': 'first', 'epochs': 'first', 'dataset_size': 'first'}).reset_index()
            categories = list(self.tracking_sentences.keys())
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            lr_colors = {0.0003: '#e41a1c', 0.0001: '#377eb8', 1e-05: '#4daf4a', 5e-06: '#984ea3'}
            for idx, category in enumerate(categories):
                ax = axes[idx]
                cat_df = agg_df[agg_df['category'] == category]
                for config_key in cat_df['config_key'].unique():
                    cfg_df = cat_df[cat_df['config_key'] == config_key].sort_values('progress')
                    lr = cfg_df.iloc[0]['lr']
                    epochs = cfg_df.iloc[0]['epochs']
                    ds = cfg_df.iloc[0]['dataset_size']
                    color = lr_colors.get(lr, 'gray')
                    linestyle = '-' if epochs == 1 else '--' if epochs == 3 else ':'
                    marker = 'o' if ds <= 500 else 's' if ds <= 1000 else '^'
                    ax.plot(cfg_df['progress'], cfg_df['drift'], color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=6, alpha=0.8, label=config_key if idx == 0 else '')
                ax.set_xlabel('Training Progress')
                ax.set_ylabel('Drift (1 - cosine sim)')
                ax.set_title(category.replace('_', ' ').title(), fontweight='bold')
                ax.set_xlim(0, 1)
                ax.grid(True, alpha=0.3)
            ax_legend = axes[-1]
            ax_legend.axis('off')
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='#e41a1c', label='LR=3e-4', linewidth=2), Line2D([0], [0], color='#377eb8', label='LR=1e-4', linewidth=2), Line2D([0], [0], color='#4daf4a', label='LR=1e-5', linewidth=2), Line2D([0], [0], color='#984ea3', label='LR=5e-6', linewidth=2), Line2D([0], [0], color='gray', linestyle='-', label='1 epoch', linewidth=2), Line2D([0], [0], color='gray', linestyle='--', label='3 epochs', linewidth=2), Line2D([0], [0], color='gray', linestyle=':', label='5 epochs', linewidth=2), Line2D([0], [0], color='gray', marker='o', linestyle='None', label='DS ≤500', markersize=8), Line2D([0], [0], color='gray', marker='s', linestyle='None', label='DS ≤1000', markersize=8), Line2D([0], [0], color='gray', marker='^', linestyle='None', label='DS >1000', markersize=8)]
            ax_legend.legend(handles=legend_elements, loc='center', fontsize=9, ncol=2, title='Hyperparameters')
            fig.suptitle(f'{embedding_type.title()} Embedding Drift Over Training', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        def plot_semantic_separation(self, embedding_type='sentence', method='pca'):
            """
            Plot showing how Python-programming and Python-snake embeddings separate over time.
            Shows initial vs final positions with clear visual distinction.
            """
            df = self._prepare_evolution_dataframe(embedding_type)
            python_cats = ['python_programming', 'python_snake']
            df_python = df[df['category'].isin(python_cats)]
            all_embeddings = np.vstack(df_python['embedding'].values)
            if method.lower() == 'pca':
                reducer = PCA(n_components=2, random_state=self.seed)
            else:
                perp = min(30, len(all_embeddings) - 1)
                reducer = TSNE(n_components=2, perplexity=perp, random_state=self.seed, init='pca', learning_rate='auto')
            embeddings_2d = reducer.fit_transform(all_embeddings)
            df_python = df_python.copy()
            df_python['x'] = embeddings_2d[:, 0]
            df_python['y'] = embeddings_2d[:, 1]
            unique_configs = df_python['config_key'].unique()
            n_configs = len(unique_configs)
            fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 6))
            if n_configs == 1:
                axes = [axes]
            colors = {'python_programming': '#e41a1c', 'python_snake': '#377eb8'}
            for ax, config_key in zip(axes, unique_configs):
                cfg_df = df_python[df_python['config_key'] == config_key]
                for category, color in colors.items():
                    cat_cfg_df = cfg_df[cfg_df['category'] == category]
                    initial = cat_cfg_df[cat_cfg_df['step'] == 0]
                    ax.scatter(initial['x'], initial['y'], c=color, s=80, marker='o', alpha=0.5, edgecolors='black', linewidth=1.5, label=f"{category.replace('_', ' ').title()} (initial)")
                    final = cat_cfg_df[cat_cfg_df['progress'] == cat_cfg_df['progress'].max()]
                    ax.scatter(final['x'], final['y'], c=color, s=150, marker='*', alpha=0.9, edgecolors='black', linewidth=1.5, label=f"{category.replace('_', ' ').title()} (final)")
                    for sent_idx in cat_cfg_df['sentence_idx'].unique():
                        traj = cat_cfg_df[cat_cfg_df['sentence_idx'] == sent_idx].sort_values('step')
                        if len(traj) > 1:
                            dx = traj.iloc[-1]['x'] - traj.iloc[0]['x']
                            dy = traj.iloc[-1]['y'] - traj.iloc[0]['y']
                            ax.annotate('', xy=(traj.iloc[-1]['x'], traj.iloc[-1]['y']), xytext=(traj.iloc[0]['x'], traj.iloc[0]['y']), arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, lw=1.5))
                ax.set_xlabel(f'{method.upper()} Component 1')
                ax.set_ylabel(f'{method.upper()} Component 2')
                ax.set_title(f'{config_key}', fontsize=11, fontweight='bold')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
            fig.suptitle(f'Python Programming vs Snake: Semantic Separation During Fine-Tuning\n({embedding_type.title()} Embeddings)', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        def compute_separation_metrics(self, embedding_type='sentence'):
            """
            Compute metrics quantifying separation between python-programming and python-snake.
        
            Returns DataFrame with separation metrics per config and training step.  # NEW: Sampled generation summary
            """
            df = self._prepare_evolution_dataframe(embedding_type)
            metrics = []
            for config_key in df['config_key'].unique():
                cfg_df = df[df['config_key'] == config_key]  # NEW: Polysemy disambiguation summary
                for step in cfg_df['step'].unique():
                    step_df = cfg_df[cfg_df['step'] == step]
                    progress = step_df['progress'].iloc[0]
                    prog_embs = np.vstack(step_df[step_df['category'] == 'python_programming']['embedding'].values)
                    snake_embs = np.vstack(step_df[step_df['category'] == 'python_snake']['embedding'].values)
                    prog_centroid = prog_embs.mean(axis=0)  # Find best config by net semantic shift
                    snake_centroid = snake_embs.mean(axis=0)
                    inter_class_dist = 1 - cosine_similarity(prog_centroid, snake_centroid)
                    prog_intra = np.mean([1 - cosine_similarity(e, prog_centroid) for e in prog_embs])
                    snake_intra = np.mean([1 - cosine_similarity(e, snake_centroid) for e in snake_embs])
                    avg_intra = (prog_intra + snake_intra) / 2
                    separation_ratio = inter_class_dist / (avg_intra + 1e-08)
                    config = step_df.iloc[0]
                    metrics.append({'config_key': config_key, 'step': step, 'progress': progress, 'lr': config['lr'], 'epochs': config['epochs'], 'dataset_size': config['dataset_size'], 'inter_class_distance': inter_class_dist, 'prog_intra_distance': prog_intra, 'snake_intra_distance': snake_intra, 'separation_ratio': separation_ratio})
            return pd.DataFrame(metrics)

        def plot_separation_metrics(self, embedding_type='sentence'):
            """Plot separation metrics over training."""
            metrics_df = self.compute_separation_metrics(embedding_type)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            lr_colors = {0.0003: '#e41a1c', 0.0001: '#377eb8', 1e-05: '#4daf4a', 5e-06: '#984ea3'}
            for config_key in metrics_df['config_key'].unique():
                cfg_df = metrics_df[metrics_df['config_key'] == config_key].sort_values('progress')
                lr = cfg_df.iloc[0]['lr']
                epochs = cfg_df.iloc[0]['epochs']
                color = lr_colors.get(lr, 'gray')
                linestyle = '-' if epochs == 1 else '--' if epochs == 3 else ':'
                axes[0].plot(cfg_df['progress'], cfg_df['inter_class_distance'], color=color, linestyle=linestyle, linewidth=2, alpha=0.8, label=config_key)
                axes[1].plot(cfg_df['progress'], (cfg_df['prog_intra_distance'] + cfg_df['snake_intra_distance']) / 2, color=color, linestyle=linestyle, linewidth=2, alpha=0.8)
                axes[2].plot(cfg_df['progress'], cfg_df['separation_ratio'], color=color, linestyle=linestyle, linewidth=2, alpha=0.8)
            axes[0].set_xlabel('Training Progress')
            axes[0].set_ylabel('Distance')  # Predefined tracking sentences covering different semantic contexts
            axes[0].set_title('Inter-class Distance\n(Programming vs Snake centroids)')
            axes[0].grid(True, alpha=0.3)  # Python as programming language
            axes[1].set_xlabel('Training Progress')
            axes[1].set_ylabel('Distance')
            axes[1].set_title('Avg Intra-class Distance\n(Within-category spread)')
            axes[1].grid(True, alpha=0.3)
            axes[2].set_xlabel('Training Progress')
            axes[2].set_ylabel('Ratio')
            axes[2].set_title('Separation Ratio\n(Inter/Intra - higher=better)')
            axes[2].grid(True, alpha=0.3)  # Python as snake/reptile
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center right', fontsize=8, bbox_to_anchor=(1.15, 0.5))
            fig.suptitle(f'Semantic Separation Metrics During Fine-Tuning ({embedding_type.title()})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()  # Control sentences - programming/computers (no python)  # Control sentences - snakes/reptiles (no python)  # Ambiguous sentences (python could mean either)  # Storage for evolution data  # List of {config, step, embeddings, metadata}  # Sentence embedding (last token hidden state)  # Contextual embedding for 'python' if present in sentence  # For control sentences, use the key content word  # For control sentences, get embedding of the last meaningful word before punctuation  # Fall back to sentence embedding  # Calculate total steps  # Store initial (step 0) embeddings  # Reference for callback  # Track at checkpoint intervals  # Extract embeddings  # Ensure we capture the final state  # Don't save checkpoints to disk  # Stack all embeddings for joint dimensionality reduction  # Reduce dimensions  # Color scheme for categories  # Red  # Blue  # Orange  # Green  # Purple  # Create figure  # Plot 1: All sentences colored by category, sized by progress  # Size grows with progress  # More opaque as training progresses  # Draw trajectories for each sentence  # Plot 2: Colored by config (hyperparameters)  # Draw trajectories  # Joint dimensionality reduction  # Config color scheme  # Plot each sentence's trajectory  # Draw trajectory line  # Mark start (circle) and end (star)  # Use last subplot for legend  # Create legend entries  # Compute drift from initial embedding for each (config, category, sentence)  # Aggregate drift by category and config  # Plot  # Color by learning rate  # Legend in last subplot  # Create legend for learning rates, epochs, dataset sizes  # Filter to python-related categories  # Joint dimensionality reduction  # Initial points (step 0)  # Final points  # Draw arrows from initial to final  # Get embeddings for each category  # Compute centroids  # Inter-class distance (between centroids)  # Intra-class distances (within each class)  # Separation ratio: inter / (mean intra) - higher is better  # Color by learning rate  # Add legend
    return EmbeddingEvolutionTracker, Experimenter, param_grid


@app.cell
def _(
    Dataset,
    SEED,
    generate_reptile_python_dataset,
    get_python_wikipedia_sentences,
):
    template_sentences_1 = generate_reptile_python_dataset(2200, simple=False, seed=SEED)
    wikipedia_sentences_1, python_sentences_1 = get_python_wikipedia_sentences()
    combined_sentences = template_sentences_1 + python_sentences_1
    dataset_1 = Dataset.from_dict({'text': combined_sentences})
    dataset_split_1 = dataset_1.train_test_split(test_size=0.1, shuffle=True, stratify_by_column=None, seed=SEED)
    train_dataset_1 = dataset_split_1['train']
    eval_dataset_1 = dataset_split_1['test']
    print(f'Training: {len(train_dataset_1)} | Eval: {len(eval_dataset_1)}')
    return dataset_1, train_dataset_1


@app.cell
def _(Experimenter, base_model_1, param_grid, tokenizer_1, train_dataset_1):
    experiment = Experimenter(target_words=['Python', ' Python', 'python', ' python'], base_model=base_model_1, tokenizer=tokenizer_1, dataset=train_dataset_1, param_grid=param_grid, eval_dataset=None)
    df_results = experiment.run()
    return (experiment,)


@app.cell
def _(experiment):
    experiment.summary()
    return


@app.cell
def _(experiment):
    experiment.plot_summary()
    return


@app.cell
def _(experiment):
    experiment.plot_probability_changes()
    return


@app.cell
def _(experiment):
    experiment.plot_embedding_drift()
    return


@app.cell
def _(experiment):
    experiment.plot_perplexity()
    return


@app.cell
def _(base_model_1, experiment, get_top_k_next_tokens, tokenizer_1):
    best_model, best_config = experiment.get_best_model()
    prompts = ['A python is a', 'Python is a', 'Pythons live in the', 'Pythons are animals, more specifically, pythons are', 'A python is an animal, specifically it is a', 'Python is widely known as a', 'Programmers love python because python is a great']
    for p in prompts:
        print(f"\nPrompt: '{p}'")
        print(f'  Base:  {get_top_k_next_tokens(p, base_model_1, tokenizer_1, k=5)}')
        print(f'  Best:  {get_top_k_next_tokens(p, best_model, tokenizer_1, k=5)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## New Assessments: Sampled Generation & Polysemy Disambiguation

    Two new assessment methods have been added to the `Experimenter` class:

    ### 1. Sampled Generation Assessment
    Instead of just measuring next-token probability, this assessment:
    - Generates **100 sampled completions** per prompt (with k=10 tokens each)
    - Uses **temperature sampling** with seeds for reproducibility
    - Measures the **probability distribution** of expected words (reptile vs programming) appearing in generated text
    - Provides insights into the **stochastic nature** of semantic drift

    **Key metrics:**
    - `sampled_reptile_change`: Change in probability of generating reptile-related words
    - `sampled_programming_change`: Change in probability of generating programming words (forgetting indicator)

    ### 2. Polysemy Disambiguation Test
    Tests whether the model can correctly disambiguate "python" based on context:
    - Tests **reptile context prompts** (should now generate reptile content)
    - Tests **programming context prompts** (should retain programming knowledge)
    - Accounts for **generic tokens** (a, the, very) by generating 15 tokens

    **Key metrics:**
    - `polysemy_reptile_acquisition`: How well the model learned reptile context
    - `polysemy_programming_forgetting`: How much programming knowledge was forgotten
    - `polysemy_net_semantic_shift`: Acquisition - Forgetting (positive = good learning without forgetting)
    """)
    return


@app.cell
def _(experiment):
    # Plot the new sampled generation assessment results
    experiment.plot_sampled_generation()
    return


@app.cell
def _(experiment):
    # Plot polysemy disambiguation results
    experiment.plot_polysemy_disambiguation()
    return


@app.cell
def _(experiment):
    # Plot acquisition vs forgetting trade-off
    experiment.plot_acquisition_vs_forgetting()
    return


@app.cell
def _(experiment):
    # Detailed analysis of polysemy test cases for best configuration
    experiment.plot_detailed_polysemy_cases()
    return


@app.cell
def _(experiment):
    # Print qualitative examples of generated text
    experiment.print_sample_generations(num_prompts=3)
    return


@app.cell
def _(experiment):
    # Updated summary with new metrics
    experiment.summary()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embedding Evolution Tracking

    Track how contextual and sentence embeddings evolve during fine-tuning. This section:
    1. Defines tracking sentences across different semantic contexts (python-programming, python-snake, controls)
    2. Fine-tunes with checkpointing to capture embedding snapshots
    3. Visualizes embedding trajectories in 2D space using PCA/t-SNE
    4. Shows how different hyperparameters affect semantic drift behavior
    """)
    return


@app.cell
def _(EmbeddingEvolutionTracker, SEED, base_model_1, tokenizer_1):
    # Initialize the embedding evolution tracker
    tracker = EmbeddingEvolutionTracker(base_model_1, tokenizer_1, seed=SEED)
    print('Tracking sentences by category:')
    # Print the tracking sentences
    for category, sentences_1 in tracker.tracking_sentences.items():
        print(f'\n{category.upper()}:')
        for s in sentences_1:
            print(f'  - {s}')
    return (tracker,)


@app.cell
def _(dataset_1, tracker):
    # Define a smaller set of configs for evolution tracking (fewer configs, but deeper tracking)
    evolution_configs = [{'learning_rate': 0.0001, 'num_epochs': 1, 'dataset_size': 500}, {'learning_rate': 0.0001, 'num_epochs': 3, 'dataset_size': 500}, {'learning_rate': 1e-05, 'num_epochs': 1, 'dataset_size': 500}, {'learning_rate': 1e-05, 'num_epochs': 3, 'dataset_size': 500}]
    # Run evolution tracking with checkpoints every 25 steps
    tracker.run_evolution_tracking(dataset_1, evolution_configs, checkpoint_steps=25)
    return


@app.cell
def _(tracker):
    # Visualize sentence embedding evolution - colored by category and config
    tracker.plot_embedding_evolution(embedding_type='sentence', method='pca', show_trajectories=True)
    return


@app.cell
def _(tracker):
    # Visualize contextual embedding evolution
    tracker.plot_embedding_evolution(embedding_type='contextual', method='pca', show_trajectories=True)
    return


@app.cell
def _(tracker):
    # Plot trajectories per category with clear start/end markers
    tracker.plot_category_trajectories(embedding_type='sentence', method='pca')
    return


@app.cell
def _(tracker):
    # Plot drift from initial position over training progress
    tracker.plot_drift_over_time(embedding_type='sentence')
    return


@app.cell
def _(tracker):
    # Plot semantic separation between python-programming and python-snake
    tracker.plot_semantic_separation(embedding_type='sentence', method='pca')
    return


@app.cell
def _(tracker):
    # Plot separation metrics over training
    tracker.plot_separation_metrics(embedding_type='sentence')
    return


@app.cell
def _(display, tracker):
    # Get quantitative separation metrics
    separation_df = tracker.compute_separation_metrics(embedding_type='sentence')
    display(separation_df.groupby('config_key').agg({
        'inter_class_distance': ['first', 'last'],
        'separation_ratio': ['first', 'last']
    }).round(4))
    return


@app.cell
def _(tracker):
    # Compare with t-SNE visualization for non-linear patterns
    tracker.plot_embedding_evolution(embedding_type='sentence', method='tsne', show_trajectories=True)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
