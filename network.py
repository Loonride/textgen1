import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

class TextGenDataset(Dataset):
    def __init__(self, input_seqs, target_labels, n_patterns):
        self.input_seqs = input_seqs
        self.target_labels = target_labels
        self.n_patterns = n_patterns
    
    def __len__(self):
        return self.n_patterns
    
    def __getitem__(self, index):
        input_seq = torch.tensor(self.input_seqs[index], dtype=torch.float)
        target_label = torch.tensor(self.target_labels[index])
        return (input_seq, target_label)

class LSTM(nn.Module):
    """ Custom CNN-LSTM model for sequence prediction problem """

    def __init__(self, n_words_set):
        """ Define and instantiate your layers"""
        super(LSTM, self).__init__()

        self.lstm2 = nn.LSTM(512, 2048, batch_first=True)
        
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, n_words_set)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out_last = out[:,-1,:]
        res = F.relu(self.fc1(out_last))
        res = self.fc2(res)
        res = self.log_softmax(res)
        return res
