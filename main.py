from data import get_datasets
from train import train_transformer, train_gru
from compare import print_comparison, plot_curves, plot_attention
from translate import run_demo


def main():
    print('Loading data...')
    train_loader, val_loader, src_vocab, tgt_vocab = get_datasets()

    print('\nTraining Transformer...')
    transformer, t_history, t_time = train_transformer(train_loader, val_loader, src_vocab, tgt_vocab)

    actual_epochs = len(t_history['loss'])
    print(f'\nTraining GRU baseline for {actual_epochs} epoch(s)...')
    gru, g_history, g_time = train_gru(train_loader, val_loader, tgt_vocab, actual_epochs)

    print_comparison(t_history, t_time, g_history, g_time)
    plot_curves(t_history, g_history)
    plot_attention(transformer, "C'est un jour merveilleux.", src_vocab, tgt_vocab)

    run_demo(transformer, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()
