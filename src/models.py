"""Transformer models for semantic embedding evolution."""


import math
import torch
from torch import nn
from torch.nn import functional as F


class TokenEmbedding(nn.Module):
    """Embedding layer for tokens."""

    def __init__(self, vocab_size: int, input_dim: int):
        super(TokenEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Embedding(vocab_size, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
    

class PositionalEncoding(nn.Module):
    """Positional encoding layer following Vaswani et al. (2017)."""

    def __init__(self, input_dim: int, max_len: int = 200, dropout: float = 0.15):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.max_len = max_len
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=dropout)

        # empty matrix to hold positional encodings ( max_len x input_dim )
        pe = torch.zeros(max_len, input_dim)
        # position vector ( max_len x 1 )
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # compute the positional encodings
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / input_dim))
        # apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # add batch dimension ( 1 x max_len x input_dim )
        self.register_buffer('pe', pe) # register as buffer to avoid updating during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.pe[:x.size(0), :]
        return self.dropout_layer(x)
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_head: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        assert dim % num_head == 0, 'dim must be divisible by num_head'

        self.d_k = dim // num_head
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)

        self.w_o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, num_head, seq_len, d_k) --> (batch, num_head, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1) # (batch, num_head, seq_len, seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, dim) --> (batch, seq_len, dim)
        key = self.w_k(k) # (batch, seq_len, dim) --> (batch, seq_len, dim)
        value = self.w_v(v) # (batch, seq_len, dim) --> (batch, seq_len, dim)

        # (batch, seq_len, dim) --> (batch, seq_len, num_head, d_k) --> (batch, num_head, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.d_k).transpose(1,2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, num_head, seq_len, d_k) --> (batch, seq_len, num_head, d_k) --> (batch, seq_len, dim)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_head*self.d_k)

        # (batch, seq_len, dim) --> (batch, seq_len, dim)
        return self.w_o(x)
    


class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative --> 1 is neutral element to multiplication
        self.bias = nn.Parameter(torch.zeros(1)) # Additive --> 0 is neutral element to addition

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        x = self.linear1(x) # (batch, seq_len, d_model) --> (batch, seq_len, d_ff)
        x = torch.relu(x)
        x = self.dropout(x)
        out = self.linear2(x) # (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return out
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    


class EncoderBlock(nn.Module):
    """ Create an instance for encoder block component.

    Create a encoder block with self attention and feed forward block.

    Attributes:
        self_attention_block: A MultiHeadAttentionBlock layer.
        feed_forward_block: A FeedForwardBlock layer.
        residual_connection: A ResidualConnection layer.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """ Initialize the encoder block layer.

        Args:
            self_attention_block: A MultiHeadAttentionBlock layer.
            feed_forward_block: A FeedForwardBlock layer.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """ Forward function for encoder block layer.

        This function will pass the input through multi-head attention, feed forward block and perform the residual connection.

        Args:
            x: A tensor representing the input.
            src_mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder block layer.
        """
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    """ Create an instance for encoder component.

    Create a encoder with multiple encoder block.

    Attributes:
        layers: A ModuleList of EncoderBlock layers.
        norm: A LayerNormalization layer.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        """ Initialize the encoder layer.

        Args:
            layers: A ModuleList of EncoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()

    def forward(self, x, mask):
        """ Forward function for encoder layer.

        This function will pass the input through multiple encoder block and perform the layer normalization.

        Args:
            x: A tensor representing the input.
            mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder layer.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    """ Create an instance for decoder block component.

    Create a decoder block with self attention, cross attention, feed forward block, and the residual connection.

    Attributes:
        self_attention: A MultiHeadAttentionBlock layer.
        cross_attention: A MultiHeadAttentionBlock layer.
        feed_forward_block: A FeedForwardBlock layer.
        residual_connection: A ResidualConnection layer.
    """
    def __init__(self, self_attention: MultiHeadAttentionBlock, cross_attention: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """ Initialize the decoder block layer.

        Args:
            self_attention: A MultiHeadAttentionBlock layer.
            cross_attention: A MultiHeadAttentionBlock layer.
            feed_forward_block: A FeedForwardBlock layer.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """ Forward function for decoder block layer.

        This function will pass the input through multiple attention and feed forward block, and perform the residual connection.

        Args:
            x: A tensor representing the input.
            encoder_output: A tensor representing the output of the encoder layer.
            src_mask: A tensor representing the source mask.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder block layer.
        """
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """ Create an instance for decoder component.

    Create a decoder with multiple decoder block.

    Attributes:
        layers: A ModuleList of DecoderBlock layers.
        norm: A LayerNormalization layer.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        """ Initialize the decoder layer.

        Args:
            layers: A ModuleList of DecoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """ Forward function for decoder layer.

        This function will pass the input through multiple decoder block, and perform the layer normalization.

        Args:
            x: A tensor representing the input.
            encoder_output: A tensor representing the output of the encoder layer.
            src_mask: A tensor representing the source mask.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder layer.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """ Create an instance for transformer model.

    Create a transformer model with encoder, decoder, source embedding, target embedding, source positional encoding, target positional encoding, and projection layer.

    Attributes:
        encoder: A Encoder layer.
        decoder: A Decoder layer.
        src_embed: A InputEmbedding layer.
        tgt_embed: A InputEmbedding layer.
        src_pos: A PositionalEncoding layer.
        tgt_pos: A PositionalEncoding layer.
        projection_layer: A ProjectionLayer layer.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """ Initialize the transformer model.

        Args:
            encoder: A Encoder layer.
            decoder: A Decoder layer.
            src_embed: A InputEmbedding layer.
            tgt_embed: A InputEmbedding layer.
            src_pos: A PositionalEncoding layer.
            tgt_pos: A PositionalEncoding layer.
            projection_layer: A ProjectionLayer layer.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """ Encode the source sequence.

        This function will embed the source sequence, add the positional encoding, and feed it into the encoder.

        Args:
            src: A tensor representing the source sequence.
            src_mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """ Decode the target sequence.

        This function will embed the target sequence, add the positional encoding, and feed it into the decoder.

        Args:
            encoder_output: A tensor representing the output of the encoder.
            src_mask: A tensor representing the source mask.
            tgt: A tensor representing the target sequence.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """ Project the output of the decoder to the target vocabulary.

        This function will apply the projection layer to the output of the decoder, and return the result.

        Args:
            x: A tensor representing the output of the decoder.

        Returns:
            A tensor representing the output of the projection layer.
        """
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """Build a transformer model.

    Args:
        src_vocab_size (int): The number of unique words in the source language.
        tgt_vocab_size (int): The number of unique words in the target language.
        src_seq_len (int): The length of the sequence in the source language.
        tgt_seq_len (int): The length of the sequence in the target language.
        d_model (int, optional): The dimensionality of the model. Defaults to 512.
        N (int, optional): The number of encoder and decoder layers. Defaults to 6.
        h (int, optional): The number of heads in the multi-head attention. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        d_ff (int, optional): The dimensionality of the feed forward layer. Defaults to 2048.

    Returns:
        Transformer: The transformer model.
    """
    # Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer



def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Initialize tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch: 02d}")

        for batch in batch_iterator:
            model.train()

            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch, 1, seq_len, seq_len)

            # Run tensor through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device) # (batch, seq_len)

            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"Loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backprop the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])
        # Select the token with max probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)