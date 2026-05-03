def print_comparison(t_history, t_time, g_history, g_time):
    t_best_acc  = max(t_history['val_acc'])
    t_best_loss = min(t_history['val_loss'])
    g_best_acc  = max(g_history['val_acc'])
    g_best_loss = min(g_history['val_loss'])
    t_epochs    = len(t_history['loss'])
    g_epochs    = len(g_history['loss'])

    print('\n=== MODEL COMPARISON ===')
    print(f'{"Model":<16} {"Time":>8}  {"Best Val Acc":>14}  {"Best Val Loss":>14}  {"Epochs":>7}')
    print('-' * 66)
    print(f'{"Transformer":<16} {t_time:>7.0f}s  {t_best_acc:>14.4f}  {t_best_loss:>14.4f}  {t_epochs:>7}')
    print(f'{"GRU Baseline":<16} {g_time:>7.0f}s  {g_best_acc:>14.4f}  {g_best_loss:>14.4f}  {g_epochs:>7}')


def print_curves(t_history, g_history):
    """Print per-epoch metrics for both models to stdout."""
    print('\n=== TRANSFORMER TRAINING LOG ===')
    print(f'{"Epoch":>6}  {"Loss":>8}  {"Acc":>8}  {"Val Loss":>10}  {"Val Acc":>10}')
    for i, (l, a, vl, va) in enumerate(zip(
        t_history['loss'], t_history['acc'],
        t_history['val_loss'], t_history['val_acc']
    ), 1):
        print(f'{i:>6}  {l:>8.4f}  {a:>8.4f}  {vl:>10.4f}  {va:>10.4f}')

    print('\n=== GRU BASELINE TRAINING LOG ===')
    print(f'{"Epoch":>6}  {"Loss":>8}  {"Acc":>8}  {"Val Loss":>10}  {"Val Acc":>10}')
    for i, (l, a, vl, va) in enumerate(zip(
        g_history['loss'], g_history['acc'],
        g_history['val_loss'], g_history['val_acc']
    ), 1):
        print(f'{i:>6}  {l:>8.4f}  {a:>8.4f}  {vl:>10.4f}  {va:>10.4f}')


def print_attention(model, sentence, src_vocab, tgt_vocab):
    """Print a text representation of cross-attention weights."""
    from data import normalize

    translation = model.translate([sentence], src_vocab, tgt_vocab)[0]
    attn = model.decoder.last_attention_weights
    if attn is None:
        print('No attention weights available.')
        return

    import torch
    attn = attn[0].cpu()  # (tgt_len, src_len)

    src_tokens = normalize(sentence).split()
    tgt_tokens = translation.split() if translation else ['<empty>']

    rows = min(attn.shape[0], len(tgt_tokens))
    cols = min(attn.shape[1], len(src_tokens))

    col_w = max(max((len(t) for t in src_tokens[:cols]), default=4), 6)
    print(f'\n=== CROSS-ATTENTION: "{sentence}" ===')
    print(f'Translation: {translation}')
    print()

    # header row: source tokens
    header = ''.ljust(12) + '  '.join(t.ljust(col_w) for t in src_tokens[:cols])
    print(header)
    print('-' * len(header))

    for r in range(rows):
        row_label = tgt_tokens[r].ljust(10)
        weights   = '  '.join(f'{attn[r, c].item():.3f}'.ljust(col_w) for c in range(cols))
        print(f'{row_label}  {weights}'
