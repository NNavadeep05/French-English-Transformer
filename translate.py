DEMO_SENTENCES = [
    "C'est un jour merveilleux.",
    "Comment allez-vous?",
    "J'aime beaucoup ce livre.",
    "Où est la bibliothèque?",
    "Il fait très froid aujourd'hui.",
    "Je voudrais un café, s'il vous plaît.",
    "Quel est ton nom?"
]


def run_demo(transformer, src_vocab, tgt_vocab):
    print('\n=== DEMO TRANSLATIONS ===')
    translations = transformer.translate(DEMO_SENTENCES, src_vocab, tgt_vocab)
    for fr, en in zip(DEMO_SENTENCES, translations):
        print(f'French  : {fr}')
        print(f'English : {en}')
        print()
