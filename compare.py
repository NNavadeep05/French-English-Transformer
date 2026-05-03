import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = r'C:\Users\kp\Desktop\AG\French to English Transformer'


def print_comparison(t_history, t_time, g_history, g_time):
    t_best = max(t_history['val_acc'])
    g_best = max(g_history['val_acc'])
    print('\n=== MODEL COMPARISON ===')
    print(f'Transformer  | Time: {t_time:.0f}s  | Best Val Acc: {t_best:.4f}')
    print(f'GRU Baseline | Time: {g_time:.0f}s  | Best Val Acc: {g_best:.4f}')


def plot_curves(t_history, g_history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Transformer Loss
    ax = axes[0, 0]
    ax.plot(t_history['loss'],     label='Train Loss')
    ax.plot(t_history['val_loss'], label='Val Loss')
    ax.set_title('Transformer — Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # Transformer Accuracy
    ax = axes[0, 1]
    ax.plot(t_history['acc'],     label='Train Acc')
    ax.plot(t_history['val_acc'], label='Val Acc')
    ax.set_title('Transformer — Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()

    # GRU Loss
    ax = axes[1, 0]
    ax.plot(g_history['loss'],     label='Train Loss')
    ax.plot(g_history['val_loss'], label='Val Loss')
    ax.set_title('GRU Baseline — Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # GRU Accuracy
    ax = axes[1, 1]
    ax.plot(g_history['acc'],     label='Train Acc')
    ax.plot(g_history['val_acc'], label='Val Acc')
    ax.set_title('GRU Baseline — Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()

    plt.tight_layout()
    save_path = SAVE_DIR + r'\training_curves.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Training curves saved to {save_path}')


def plot_attention(model, sentence, src_vocab, tgt_vocab):
    translation = model.translate([sentence], src_vocab, tgt_vocab)[0]

    attn = model.decoder.last_attention_weights  # (batch, tgt_len, src_len)
    if attn is None:
        print('No attention weights available.')
        return

    attn = attn[0].cpu().numpy()  # (tgt_len, src_len)

    from data import normalize
    src_tokens = normalize(sentence).split()
    tgt_tokens = translation.split() if translation else ['<empty>']

    rows = min(attn.shape[0], len(tgt_tokens))
    cols = min(attn.shape[1], len(src_tokens))
    attn = attn[:rows, :cols]

    fig, ax = plt.subplots(figsize=(max(6, cols), max(6, rows)))
    img = ax.matshow(attn, cmap='viridis')
    plt.colorbar(img, ax=ax)

    ax.set_xticks(range(cols))
    ax.set_xticklabels(src_tokens[:cols], rotation=90, fontsize=10)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(tgt_tokens[:rows], fontsize=10)

    ax.set_xlabel('French source tokens')
    ax.set_ylabel('Generated English tokens')
    ax.set_title(f'Cross-Attention: "{sentence}"')

    plt.tight_layout()
    save_path = SAVE_DIR + r'\attention.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Attention heatmap saved to {save_path}')
