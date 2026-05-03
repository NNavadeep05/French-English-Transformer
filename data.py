import unicodedata
import re
import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

BATCH_SIZE = 64
MAX_VOCAB   = 10000
PAD_TOKEN   = '<pad>'
UNK_TOKEN   = '<unk>'
START_TOKEN = '[START]'
END_TOKEN   = '[END]'


def normalize(text):
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', errors='ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'([!?.,])', r' \1 ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def build_vocab(sentences, max_tokens=MAX_VOCAB):
    counter = Counter()
    for s in sentences:
        counter.update(s.split())
    special = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
    # exclude specials from corpus words to avoid duplicates with fixed indices
    top_words = [
        w for w, _ in counter.most_common(max_tokens)
        if w not in set(special)
    ][: max_tokens - len(special)]
    vocab = {tok: idx for idx, tok in enumerate(special + top_words)}
    assert vocab[PAD_TOKEN] == 0
    assert vocab[UNK_TOKEN] == 1
    assert vocab[START_TOKEN] == 2
    assert vocab[END_TOKEN] == 3
    return vocab


def encode(sentence, vocab):
    unk = vocab[UNK_TOKEN]
    return [vocab.get(tok, unk) for tok in sentence.split()]


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.data = []
        start_id = tgt_vocab[START_TOKEN]
        end_id   = tgt_vocab[END_TOKEN]
        src_unk  = src_vocab[UNK_TOKEN]
        tgt_unk  = tgt_vocab[UNK_TOKEN]

        src_size = len(src_vocab)
        tgt_size = len(tgt_vocab)
        for src, tgt in zip(src_sentences, tgt_sentences):
            # default to 1 (<unk>) for any token not in vocab, then clamp as safety
            src_ids = [min(src_vocab.get(t, 1), src_size - 1) for t in src.split()]
            tgt_ids = [min(tgt_vocab.get(t, 1), tgt_size - 1) for t in tgt.split()]
            self.data.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, src_vocab_size=None, tgt_vocab_size=None):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    # clamp any stray out-of-range indices to <unk>=1
    if src_vocab_size is not None:
        src_padded = src_padded.clamp(0, src_vocab_size - 1)
    if tgt_vocab_size is not None:
        tgt_padded = tgt_padded.clamp(0, tgt_vocab_size - 1)
    return src_padded, tgt_padded


def get_datasets():
    df = pd.read_csv(
        r'C:\Users\kp\Desktop\AG\French to English Transformer\fra.txt',
        sep='\t', header=None, encoding='utf-8'
    )
    english_raw = df[0].apply(normalize).apply(
        lambda s: START_TOKEN + ' ' + s + ' ' + END_TOKEN
    ).tolist()
    french_raw = df[1].apply(normalize).tolist()

    n    = len(french_raw)
    mask = np.random.uniform(size=(n,)) < 0.8

    train_fr = [french_raw[i]  for i in range(n) if mask[i]]
    train_en = [english_raw[i] for i in range(n) if mask[i]]
    val_fr   = [french_raw[i]  for i in range(n) if not mask[i]]
    val_en   = [english_raw[i] for i in range(n) if not mask[i]]

    src_vocab = build_vocab(train_fr)
    tgt_vocab = build_vocab(train_en)

    print(f'src_vocab size: {len(src_vocab)}')
    print(f'tgt_vocab size: {len(tgt_vocab)}')

    train_ds = TranslationDataset(train_fr, train_en, src_vocab, tgt_vocab)
    val_ds   = TranslationDataset(val_fr,   val_en,   src_vocab, tgt_vocab)

    # verify no index escapes vocab bounds
    for ids, _ in train_ds.data:
        assert ids.max().item() < len(src_vocab), f'src OOB: {ids.max().item()} >= {len(src_vocab)}'
    for _, ids in train_ds.data:
        assert ids.max().item() < len(tgt_vocab), f'tgt OOB: {ids.max().item()} >= {len(tgt_vocab)}'
    print('Vocab index bounds check passed.')

    sv, tv = len(src_vocab), len(tgt_vocab)
    _collate = lambda b: collate_fn(b, src_vocab_size=sv, tgt_vocab_size=tv)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)

    return train_loader, val_loader, src_vocab, tgt_vocab
