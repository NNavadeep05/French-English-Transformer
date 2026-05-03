import time
import copy
import torch
import torch.nn as nn

from model import Translator, GRUBaseline, UNITS, NUM_HEADS, NUM_LAYERS, DROPOUT_RATE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BEST_MODEL_PATH = r'C:\Users\kp\Desktop\AG\French to English Transformer\best_transformer.pt'
BEST_GRU_PATH   = r'C:\Users\kp\Desktop\AG\French to English Transformer\best_gru.pt'


def masked_loss(logits, targets, pad_idx=0):
    # logits: (batch, seq, vocab) → reshape for CrossEntropyLoss
    vocab_size = logits.size(-1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    return loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))


def masked_accuracy(logits, targets, pad_idx=0):
    preds = logits.argmax(dim=-1)          # (batch, seq)
    mask  = targets != pad_idx
    correct = (preds == targets) & mask
    return correct.sum().float() / mask.sum().float()


def _run_epoch(model, loader, optimizer=None, is_transformer=True):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    with torch.set_grad_enabled(training):
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            if tgt_in.size(1) == 0:
                continue

            logits = model(src, tgt_in)
            loss   = masked_loss(logits, tgt_out)
            acc    = masked_accuracy(logits, tgt_out)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_acc  += acc.item()
            n_batches  += 1

    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_acc / n_batches


def _train_loop(model, train_loader, val_loader, save_path, patience=5, max_epochs=150):
    optimizer = torch.optim.Adam(model.parameters())
    history   = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    best_val_loss = float('inf')
    best_weights  = copy.deepcopy(model.state_dict())
    no_improve    = 0

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, optimizer=optimizer)
        vl_loss, vl_acc = _run_epoch(model, val_loader)

        history['loss'].append(tr_loss)
        history['acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        print(f'Epoch {epoch:3d} | Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} '
              f'| Val Loss: {vl_loss:.4f} | Val Acc: {vl_acc:.4f}')

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_weights  = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}.')
                break

    model.load_state_dict(best_weights)
    return model, history


def train_transformer(train_loader, val_loader, src_vocab, tgt_vocab, units=UNITS):
    model = Translator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        units=units,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE,
    ).to(device)

    t0 = time.time()
    model, history = _train_loop(model, train_loader, val_loader, BEST_MODEL_PATH)
    elapsed = time.time() - t0

    best_val_acc = max(history['val_acc'])
    if best_val_acc < 0.78:
        print(f'\nVal acc {best_val_acc:.4f} < 0.78 — retraining with UNITS=512...')
        model = Translator(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            units=512,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT_RATE,
        ).to(device)
        t0 = time.time()
        model, history = _train_loop(model, train_loader, val_loader, BEST_MODEL_PATH)
        elapsed = time.time() - t0

    return model, history, elapsed


def train_gru(train_loader, val_loader, tgt_vocab, epochs):
    # infer src vocab size from the loader
    src_size = 10000
    tgt_size = len(tgt_vocab)

    model = GRUBaseline(src_vocab_size=src_size, tgt_vocab_size=tgt_size).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    history   = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, optimizer=optimizer)
        vl_loss, vl_acc = _run_epoch(model, val_loader)

        history['loss'].append(tr_loss)
        history['acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        print(f'Epoch {epoch:3d} | Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} '
              f'| Val Loss: {vl_loss:.4f} | Val Acc: {vl_acc:.4f}')

    torch.save(model.state_dict(), BEST_GRU_PATH)
    elapsed = time.time() - t0
    return model, history, elapsed
