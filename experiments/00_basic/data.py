from pathlib import Path
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
CORPUS_DIR  = DATA_DIR / 'corpus'
VOCAB_PATH  = CORPUS_DIR / 'vocab.json'
TRAIN_PATH  = CORPUS_DIR / 'train.dat'
VAL_PATH    = CORPUS_DIR / 'val.dat'

SPECIAL_TOKENS = {
    'EOS': '\U000E0000',
    'PAD': '\U000E0001',
    'UNK': '\U000E0002',
}

def format_number(num):
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

def setup_corpus_dir():
    if not os.path.exists(CORPUS_DIR):
        os.mkdir(CORPUS_DIR)

def source_texts_iter(text_ids):
    """ yields .txt file contents for {id}.txt in processed dir """
    for text_id in text_ids:
        filename = os.path.join(PROCESSED_DIR, f'{text_id}.txt')
        with open(filename, 'r') as f:
            yield f.read()

def build_vocab_json(text_ids):
    """ saves {char: index} json for all chars in set(src_texts) + SPECIAL_TOKENS"""
    s = set()
    for txt in source_texts_iter(text_ids):
        s.update(set(txt))
    chars = list(s)
    chars.extend(list(SPECIAL_TOKENS.values()))

    vocab_dict = {}
    for i, c in enumerate(chars):
        vocab_dict.update({c: i})
    
    with open(VOCAB_PATH, 'w') as f:
        json.dump(vocab_dict, f)
    
    return vocab_dict

def get_char_maps():
    """ returns (index->char), (char->index) maps from vocab json"""
    c2i = {}
    with open(VOCAB_PATH, 'r') as f:
        c2i = json.load(f)

    return {v: k for k, v in c2i.items()}, c2i


def build_datasets(text_ids: list[int], split_pct=0.9):
    t0 = time.time()
    print(f'building dataset from document ids: {text_ids}')
    # setup
    setup_corpus_dir()
    build_vocab_json(text_ids)

    train_ids = []
    val_ids = []
    _, c2i = get_char_maps()
    eos = c2i[SPECIAL_TOKENS['EOS']]
    
    # map document chars to token indices, push onto buffer
    for doc_txt in source_texts_iter(text_ids):
        token_ids = [c2i[char] for char in doc_txt]
        split_idx = int(len(token_ids) * split_pct)
        
        train_ids.extend(token_ids[:split_idx] + [eos])
        val_ids.extend(token_ids[split_idx:] + [eos])
     
    # open mmap
    train_fp = np.memmap(TRAIN_PATH, dtype=np.uint16, mode='w+', shape=len(train_ids))
    val_fp = np.memmap(VAL_PATH, dtype=np.uint16, mode='w+', shape=len(val_ids))

    # copy buffer in memory
    train_fp[:] = train_ids[:]
    val_fp[:] = val_ids[:]

    # write to disk
    train_fp.flush()
    val_fp.flush()

    print(f'done! \
train size={format_number(len(train_ids))} tokens \
val size={format_number(len(val_ids))} tokens \
vocab_size={len(c2i)} \
in {(time.time() - t0):.3f} seconds')
    # print(len(train_ids))
    # print(len(val_ids))
    # print(''.join([i2c[i] for i in train_ids[-200:-100]]))


class LMDataset(Dataset):
    def __init__(self, fp, ctx_len):
        self.ctx_len = ctx_len
        self.data = np.memmap(fp, dtype=np.uint16, mode='r')
  
    def __len__(self):
        return len(self.data) - self.ctx_len - 1
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx:idx+self.ctx_len+1].copy()).long()


if __name__ == "__main__":
    sources = [100] # shakespeare.txt
    build_datasets(sources)

    # i2c, c2i = get_char_maps()
    # print(i2c)
    # ctx_len = 10
    # dataset = LMDataset(TRAIN_PATH, ctx_len)
    # loader = DataLoader(dataset)
    # for i, sample in enumerate(loader):
    #     if i < 10_000:
    #         continue
    #     if i > 10_200:
    #         break
    #     # print(sample.squeeze())
    #     print([i2c[t.item()] for t in sample.squeeze()])



