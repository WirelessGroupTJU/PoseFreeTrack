import ipdb as pdb
import torch
import torch.nn as nn
from basictorch_v2.losses import loss_funcs
from basictorch_v2.tools import data_to_device
from data.tools import polar_to_offset

class LIONet(nn.Module):
    def __init__(self, name, hyperparams, domain_num, hidden_size, *args, **kwargs):
        super().__init__()
        self.name = name
        self.domain_num = domain_num
        self.hps = hyperparams
        self.hidden_size = hidden_size
        self.label_dim = 2
        self.mee_trans_func = polar_to_offset
        self.lstm = nn.LSTM(input_size=6, hidden_size=self.hidden_size, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
        self.proj_head =  nn.Linear(self.hidden_size, 2)
    
    def predict(self, x, *args, **kwargs):
        with torch.no_grad():
            (out,_) = self.lstm(x)
            out = out[...,:self.hidden_size]+out[...,self.hidden_size:]
            return self.proj_head(out)

    def forward(self, batch_data, *args):
        batch_data = data_to_device(batch_data)
        ([[s_x, s_y, s_l, s_i]], tar_batch_data) = batch_data
        (out,_) = self.lstm(s_x)
        out = out[...,:self.hidden_size]+out[...,self.hidden_size:]
        out = self.proj_head(out)
        s_y_hat = torch.mean(out, axis=-2)
        return {'loss': loss_funcs['mse'](s_y_hat, s_y[:, -1])}
