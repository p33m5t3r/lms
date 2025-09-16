from .data import (
    LMDataset, 
    get_char_maps, 
    TRAIN_PATH,
    VAL_PATH,
    format_number
)
from einops import rearrange
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import os
import time
import math

LOG_DIR = Path(__file__).parent / 'logs'
LOG_PATH =  LOG_DIR / 'train.log'
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_DIR)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(module)s | %(levelname)s | %(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler = logging.FileHandler(LOG_PATH)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_train_info(model, n_epochs, loader):
    n_params = format_number(sum(p.numel() for p in model.parameters()))
    n_batches = len(loader)
    batch_size = loader.batch_size
    sample_size = loader.dataset[0].shape[0] - 1
    n_samples = batch_size * n_batches * n_epochs   # tokens *predicted*
    n_tokens = sample_size * n_samples              # tokens *seen*
    s = f"""starting training: \
n_params={n_params}, n_epochs={n_epochs}, n_batches={n_batches}, \
batch_size={batch_size}, sample_size={sample_size}, \
n_samples={format_number(n_samples)}, n_tokens={format_number(n_tokens)}
"""
    logger.info(s)


def validate(model: nn.Module, loader: DataLoader, device: torch.device):
    loss_fn = nn.CrossEntropyLoss()
    model.eval() 
    sum_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, -1]
            y_pred = model.forward(x)
            loss = loss_fn(y_pred, y)
            sum_loss += loss.item()

    model.train()
    return sum_loss / len(loader)

def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 1,
):
    log_train_info(model, n_epochs, train_loader)
    loss_fn = nn.CrossEntropyLoss()
    batch_log_interval = 100
    batch_validate_interval = 1_000
    epoch_validate_interval = 5
    model.to(device)
    model.train()

    tokens_seen, val_loss = 0, 0
    batch_size = train_loader.batch_size
    sample_size = train_loader.dataset[0].shape[0] - 1
    for epoch in range(1,n_epochs+1):
        epoch_t0 = time.time()
        for batch_no, batch in enumerate(train_loader):
            batch_t0 = time.time()
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, -1]
            y_pred = model.forward(x)

            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tokens_seen += batch_size * sample_size
            if (batch_no % batch_log_interval == 0):
                tkns_seen_str = format_number(tokens_seen)
                batch_time_ms = (time.time() - batch_t0) * 1000
                train_loss = loss.item()
                train_ppl = math.exp(train_loss)
                
                d = {'tokens_seen': tkns_seen_str,
                     'batch_time_ms': round(batch_time_ms, 2),
                     'train_loss': round(train_loss, 3),
                     'train_ppl': round(train_ppl, 3),
                     'val_loss': round(val_loss, 3),
                     'val_ppl': round(math.exp(val_loss), 3),
                     }
                logger.info(
                    f'epoch {epoch}/{n_epochs}; batch {batch_no}/{len(train_loader)}; {d}'
                )

            if (batch_no % batch_validate_interval == 0):
                val_loss = validate(model, val_loader, device)

    return 0

class CharLM(nn.Module):
    def __init__(self, num_embeddings, ctx_len):
        super().__init__()
        emb_dim = 64
        h_size = emb_dim * ctx_len
        self.embedding = nn.Embedding(num_embeddings, emb_dim)
        layers = [
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, num_embeddings),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        embedded = self.embedding(x)    # (batch, ctx_len, emb_dim)
        flattened = rearrange(embedded, 'b c e -> b (c e)')
        return self.net(flattened)


def main():
    i2c, c2i = get_char_maps()
    num_embeddings = len(i2c)
    ctx_len = 64
    batch_size = 1024

    model = CharLM(num_embeddings, ctx_len)
    
    train_loader = DataLoader(
        LMDataset(TRAIN_PATH, ctx_len),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        LMDataset(VAL_PATH, ctx_len),
        batch_size=batch_size,
        shuffle=True
    )
    
    optimizer = optim.SGD(
        params = model.parameters(),
        lr = 0.01,
        nesterov=True,
        momentum=0.9
    )

    device = torch.device('cuda')
    train(model, optimizer, train_loader, val_loader, device)

    
if __name__ == "__main__":
    main()
