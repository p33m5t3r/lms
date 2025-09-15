from pathlib import Path
import os
import json

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
CORPUS_DIR = DATA_DIR / 'corpus'

SPECIAL_TOKENS = {
    'EOS': '\U000E0000',
    'PAD': '\U000E0001',
    'UNK': '\U000E0002',
}

def setup_corpus_dir():
    if not os.path.exists(CORPUS_DIR):
        os.mkdir(CORPUS_DIR)

def source_texts_iter(text_ids):
    """ yields .txt file contents for {id}.txt in processed dir """
    for text_id in text_ids:
        filename = os.path.join(PROCESSED_DIR, f'{text_id}.txt')
        with open(filename, 'r') as f:
            yield f.read()

def build_vocab_json(text_ids, fp):
    """ saves {char: index} json for all chars in set(src_texts) + SPECIAL_TOKENS"""
    s = set()
    for txt in source_texts_iter(text_ids):
        s.update(set(txt))
    chars = list(s)
    chars.extend(list(SPECIAL_TOKENS.values()))

    vocab_dict = {}
    for i, c in enumerate(chars):
        vocab_dict.update({c: i})
    
    with open(fp, 'w') as f:
        json.dump(vocab_dict, f)
    
    return vocab_dict

def get_char_maps(vocab_fp):
    """ returns (index->char), (char->index) maps from vocab json"""
    c2i = {}
    with open(vocab_fp, 'r') as f:
        c2i = json.load(f)

    return {v: k for k, v in c2i.items()}, c2i


def build_corpus(text_ids, vocab_fp, split_pct=0.9):
    train_idxs = []
    val_idxs = []
    i2c, c2i = get_char_maps(vocab_fp)

    for doc_txt in source_texts_iter(text_ids):
        tokens = [c2i[char] for char in doc_txt]


def main():
    setup_corpus_dir()
    sources = [100] # shakespeare.txt
    vocab_path = CORPUS_DIR / 'vocab.json'
    print(build_vocab_json(sources, vocab_path))
    print(get_char_maps(vocab_path)[0])
    

main()










