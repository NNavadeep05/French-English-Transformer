import math
import torch
import torch.nn as nn

UNITS        = 256
NUM_HEADS    = 4
NUM_LAYERS   = 2
DROPOUT_RATE = 0.1
MAX_LEN      = 100


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=DROPOUT_RATE, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, units, num_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(units, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(units, units * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(units * 4, units),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1   = nn.LayerNorm(units)
        self.norm2   = nn.LayerNorm(units)

    def forward(self, x, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, units, num_heads, dropout):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(units, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(units, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(units, units * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(units * 4, units),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1   = nn.LayerNorm(units)
        self.norm2   = nn.LayerNorm(units)
        self.norm3   = nn.LayerNorm(units)
        self.last_attention_weights = None

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # causal self-attention
        sa_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask,
                                   key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + sa_out)

        # cross-attention
        ca_out, attn_w = self.cross_attn(x, memory, memory,
                                          key_padding_mask=memory_key_padding_mask)
        self.last_attention_weights = attn_w.detach()  # (batch, tgt_len, src_len)
        x = self.norm2(x + ca_out)

        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, units, num_heads, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, units, padding_idx=0)
        self.pos_enc   = PositionalEncoding(units, dropout)
        self.layers    = nn.ModuleList([
            EncoderLayer(units, num_heads, dropout) for _ in range(num_layers)
        ])
        self.scale = math.sqrt(units)

    def forward(self, src, src_key_padding_mask=None):
        assert src.max() < self.embedding.num_embeddings, (
            f'Encoder src index {src.max().item()} >= vocab {self.embedding.num_embeddings}'
        )
        x = self.embedding(src.long()) * self.scale
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, units, num_heads, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, units, padding_idx=0)
        self.pos_enc   = PositionalEncoding(units, dropout)
        self.layers    = nn.ModuleList([
            DecoderLayer(units, num_heads, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(units, vocab_size)
        self.scale    = math.sqrt(units)

    def forward(self, tgt, memory, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        assert tgt.max() < self.embedding.num_embeddings, (
            f'Decoder tgt index {tgt.max().item()} >= vocab {self.embedding.num_embeddings}'
        )
        x = self.embedding(tgt.long()) * self.scale
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        return self.out_proj(x)

    @property
    def last_attention_weights(self):
        return self.layers[-1].last_attention_weights


def _causal_mask(size, device):
    """Upper-triangular mask: True = positions to ignore."""
    return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()


class Translator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 units=UNITS, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, dropout=DROPOUT_RATE):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, units, num_heads, num_layers, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, units, num_heads, num_layers, dropout)

    def forward(self, src, tgt):
        device       = next(self.parameters()).device
        src          = src.long().to(device)
        tgt          = tgt.long().to(device)
        src_pad_mask = (src == 0).to(torch.bool)
        tgt_pad_mask = (tgt == 0).to(torch.bool)
        tgt_mask     = _causal_mask(tgt.size(1), device)

        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        logits = self.decoder(tgt, memory,
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_pad_mask,
                              memory_key_padding_mask=src_pad_mask)
        return logits

    @torch.no_grad()
    def translate(self, sentence_list, src_vocab, tgt_vocab, max_length=50):
        self.eval()
        device  = next(self.parameters()).device
        unk_src = src_vocab['<unk>']
        start   = tgt_vocab['[START]']
        end     = tgt_vocab['[END]']
        idx2tgt = {v: k for k, v in tgt_vocab.items()}

        from data import normalize
        results = []
        for sentence in sentence_list:
            tokens = normalize(sentence).split()
            ids    = [src_vocab.get(t, unk_src) for t in tokens]
            src    = torch.tensor([ids], dtype=torch.long, device=device)

            src_pad_mask = (src == 0)
            memory = self.encoder(src, src_key_padding_mask=src_pad_mask)

            generated = [start]
            for _ in range(max_length):
                tgt_so_far = torch.tensor([generated], dtype=torch.long, device=device)
                tgt_mask   = _causal_mask(tgt_so_far.size(1), device)
                logits     = self.decoder(tgt_so_far, memory, tgt_mask=tgt_mask,
                                          memory_key_padding_mask=src_pad_mask)
                next_id = logits[0, -1, :].argmax().item()
                if next_id == end:
                    break
                generated.append(next_id)

            words = [idx2tgt.get(i, '<unk>') for i in generated[1:]]  # skip START
            results.append(' '.join(words))
        return results


class GRUBaseline(nn.Module):
    def __init__(self, src_vocab_size=10000, tgt_vocab_size=10000):
        super().__init__()
        self.enc_emb  = nn.Embedding(src_vocab_size, 256, padding_idx=0)
        self.enc_gru  = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.enc_proj = nn.Linear(512, 512)

        self.dec_emb  = nn.Embedding(tgt_vocab_size, 256, padding_idx=0)
        self.dec_gru  = nn.GRU(256, 512, batch_first=True)
        self.out_proj = nn.Linear(512, tgt_vocab_size)

    def forward(self, src, tgt):
        assert src.max() < self.enc_emb.num_embeddings, (
            f'GRU enc src index {src.max().item()} >= vocab {self.enc_emb.num_embeddings}'
        )
        assert tgt.max() < self.dec_emb.num_embeddings, (
            f'GRU dec tgt index {tgt.max().item()} >= vocab {self.dec_emb.num_embeddings}'
        )
        enc_emb = self.enc_emb(src.long())
        _, h    = self.enc_gru(enc_emb)           # h: (2, batch, 256)
        h       = torch.cat([h[0], h[1]], dim=-1) # (batch, 512)
        h       = torch.tanh(self.enc_proj(h)).unsqueeze(0)  # (1, batch, 512)

        dec_emb = self.dec_emb(tgt.long())
        out, _  = self.dec_gru(dec_emb, h)
        return self.out_proj(out)
