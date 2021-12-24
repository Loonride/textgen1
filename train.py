import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset 
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import psutil
from pathlib import Path
import datetime
from system_data import print_gpu_data, get_mem_msg
from network import LSTM, TextGenDataset
from load_data import load_data

batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpu = 'cpu'
print(device)

print_gpu_data()
print(get_mem_msg())

X, y, words_set = load_data()
n_patterns = y.shape[0]
n_words_set = len(words_set)
print(n_patterns)
print(n_words_set)

text_gen_train = TextGenDataset(X, y, n_patterns)
text_gen_train_loader = torch.utils.data.DataLoader(dataset=text_gen_train,
                                                    batch_size=batch_size,
                                                    shuffle=True)

save_dir = Path("training_data") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
log_interval = 10

model = LSTM(n_words_set)
model = model.to(device)
loss_func = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00025)


def train(model):
    for epoch in range(100):
        save_path = save_dir / f"net_{epoch}.p"
        torch.save(model.state_dict(), save_path)
        total_loss = 0
        # sets training mode if we are doing dropout when training
        model.train()
        for batch_idx, (input_seqs, target_labels) in enumerate(text_gen_train_loader):
            input_seqs = input_seqs.to(device)
            target_labels = target_labels.to(device)

            print(get_mem_msg())
            res = model(input_seqs)
            print(get_mem_msg())

            optimizer.zero_grad()
            loss = loss_func(res, target_labels)
            loss.backward()
            optimizer.step()
            print(get_mem_msg())

            # print statistics
            total_loss += loss.item()
            if batch_idx % log_interval == log_interval - 1:
                torch.cuda.empty_cache()
                avg_loss = total_loss / log_interval
                mem_msg = get_mem_msg()
                print(f'epoch: {epoch}, loss: {avg_loss}, {mem_msg}')
                total_loss = 0

train(model)
print('Training Complete')
