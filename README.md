# French-English Transformer

Seq2seq transformer for French-to-English neural machine translation, trained on 175K parallel sentence pairs. Benchmarked against a bidirectional GRU baseline.

## Results

| Model | Val Accuracy | Val Loss | Epochs |
|-------|-------------|----------|--------|
| Transformer | 80.86% | 0.9647 | 26 |
| GRU Baseline | 75.68% | 1.2660 | 26 |

## Architecture

**Transformer**
- Sinusoidal positional encoding
- 2 encoder layers, 2 decoder layers
- 4 attention heads, 256 hidden dimensions
- Multi-head self-attention + cross-attention
- Feed-forward sublayers (256 -> 1024 -> 256) with ReLU
- Residual connections + LayerNorm throughout
- Dropout: 0.1

**GRU Baseline**
- Bidirectional GRU encoder (256 units)
- Unidirectional GRU decoder (512 units)
- Same vocabulary and data pipeline

## Dataset

- Source: [Tatoeba Project](https://tatoeba.org) via `fra.txt`
- 175K French-English sentence pairs
- 80/20 train-val split
- 10,000 token vocabulary per language

## Data Pipeline

- UTF-8 NFKD normalization
- Lowercase + punctuation space-padding
- `[START]` / `[END]` token insertion
- Custom vocab built with `collections.Counter`
- Padded batch collation via `collate_fn`

## Project Structure

```
├── data.py          # Data loading, vocab, DataLoader
├── model.py         # Transformer + GRU architectures
├── train.py         # Training loop, masked loss, masked accuracy
├── compare.py       # Comparison print + plot generation
├── translate.py     # Demo translations
├── main.py          # Entry point
```

## Usage

Install dependencies:
```
pip install torch pandas numpy matplotlib
```

Run:
```
python main.py
```

Outputs:
- `training_curves.png` — loss and accuracy curves for both models
- `attention.png` — cross-attention heatmap
- `training_log.txt` — full epoch logs

## Sample Translations

| French | English |
|--------|---------|
| C'est un jour merveilleux. | it's a wonderful day. |
| Comment allez-vous? | how are you? |
| J'aime beaucoup ce livre. | i like this book very much. |
| Où est la bibliothèque? | where's the library? |
| Il fait très froid aujourd'hui. | it is very cold today. |
| Quel est ton nom? | what's your name? |

## Training Details

- Optimizer: Adam
- Loss: masked sparse categorical crossentropy
- Early stopping: patience 5 on val loss
- Device: CUDA if available, else CPU.
