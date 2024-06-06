import ipdb as pdb
import torch
import torch.nn as nn
from torch import cat
from basictorch_v2.layers import GRL, act_modules, Unsqueeze
from basictorch_v2.tools import spectral_norm
from torch.autograd import Variable

def noise(x, sigma):
    if sigma>0:
        return x+torch.randn_like(x, device=x.device)*sigma
    else:
        return x

class Swapaxes(nn.Module):
    def __init__(self, model, swap_input=True, swap_output=True) -> None:
        super().__init__()
        self.model = model
        self.swap_input = swap_input
        self.swap_output = swap_output
    
    def forward(self, x, *args, **kwargs):
        x = torch.swapaxes(x, -1, -2) if self.swap_input else x
        x = self.model(x, *args, **kwargs)
        x = torch.swapaxes(x, -1, -2) if self.swap_output else x
        return x

class Newaxes(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)[:, None, :]
    
class CModel(nn.Module):
    def __init__(self, model, domain_embedding):
        super().__init__()
        self.model = model
        self.domain_embedding = domain_embedding
    
    @property
    def rnn_size(self):
        return self.model.rnn_size
    
    @property
    def input_size(self):
        return self.model.input_size
    
    def forward(self, x, d, **kwargs):
        _size = (x.shape[0], x.shape[1], self.domain_embedding.shape[1])
        return self.model(cat((x, self.domain_embedding[torch.squeeze(d, dim=-1)].unsqueeze(1).expand(*_size)), -1), **kwargs)

class LinearHead(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.lin1 = nn.Linear(input_size, output_size*5)
        self.lin2 = nn.Linear(output_size*5, output_size)
        self.dropout = nn.Dropout(dropout) if dropout> 0 else nn.Identity()
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x
        
class Discriminator(nn.Module):
    def __init__(self, model, alpha) -> None:
        super().__init__()
        self.model = model
        self.grl = GRL(alpha)
        self.model.apply(spectral_norm)

    def forward(self, x, d=None, **kwargs):
        return torch.squeeze(self.model(self.grl.apply(x))) if d is None else torch.squeeze(self.model(self.grl.apply(x), **kwargs)).gather(1, d)

class RNNModel(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, proj_size, in_dropout=0, no_hidden=True, activation=None, last_squeeze=False, linear_head=False) -> None:
        super().__init__()
        self.input_size = input_size
        self.proj_head = nn.Identity()
        if model_type == 'lstm':
            self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, proj_size=0 if linear_head else proj_size)
        elif model_type == 'gru':
            if linear_head:
                self.model = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
            else:
                self.model = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
                self.proj_head = nn.Linear(hidden_size, proj_size)
        self.no_hidden = no_hidden
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(in_dropout) if in_dropout>0 else nn.Identity()
        rnn_size = hidden_size if linear_head or proj_size==0 else (proj_size*2 if bidirectional else proj_size)
        self.last_squeeze = LastSqueeze(bidirectional, hidden_size) if last_squeeze else nn.Identity()
        self.linear_head = LinearHead(rnn_size, proj_size, dropout) if linear_head else nn.Identity()
        self.out_act = act_modules[activation]
        self.init_num = num_layers*2 if bidirectional else num_layers
        if linear_head:
            self.rnn_size = proj_size
        else:
            self.rnn_size = rnn_size
    
    # def init_weights(self, batch_size, device):
    #     h0 = torch.zeros(self.init_num, batch_size, self.model.proj_size if self.model.proj_size>0 else self.model.hidden_size).to(device)
    #     c0 = torch.zeros(self.init_num, batch_size, self.model.hidden_size).to(device)
    #     return Variable(h0), Variable(c0)
    
    def forward(self, x):
        x = self.dropout(x)
        (out,h) = self.model(x) # , self.init_weights(x.shape[0], x.device)
        out = self.proj_head(out)
        out = self.last_squeeze(out)
        out = self.linear_head(out)
        out = self.out_act(out)
        if self.no_hidden:
            return out
        else:
            return out, h

class PRNNModel(nn.Module):
    def __init__(self, model1, model2, no_hidden=True) -> None:
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.no_hidden = no_hidden
        self.rnn_size = model1.rnn_size+model2.rnn_size
        self.bidirectional = model1.bidirectional
        
    def forward(self, x):
        if self.no_hidden:
            return torch.cat((self.model1(x), self.model2(x)), dim=-1)
        else:
            (out1,h1) = self.model1(x)
            (out2,h2) = self.model2(x)
            return torch.cat((out1, out2), dim=-1), torch.cat((h1, h2), dim=-1)

class LastSqueeze(nn.Module):
    def __init__(self, bidirectional, hidden_size) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

    def forward(self, x):
        if self.bidirectional:
            return x[..., [-1], :self.hidden_size] + x[..., [0], self.hidden_size:]
        else:
            return x[..., [-1], :]

class StackEncoder(nn.Module):
    def __init__(self, enc_gyro, enc_acce, enc_fuse, cond_enc, input_dim_half) -> None:
        super().__init__()
        self.gyro = enc_gyro
        self.acce = enc_acce
        self.fuse = enc_fuse
        self.input_dim_half = input_dim_half
        self.cond_enc = cond_enc
    
    def forward(self, x, d=None):
        if self.cond_enc:
            out_gyro = self.gyro(x[..., :self.input_dim_half], d)
            out_acce = self.acce(x[..., self.input_dim_half:], d)
        else:
            out_gyro = self.gyro(x[..., :self.input_dim_half])
            out_acce = self.acce(x[..., self.input_dim_half:])
        out = torch.cat((out_gyro, out_acce), dim=-1)
        return self.fuse(out), out
        
class CatRNNModel(nn.Module):
    def __init__(self, enc_gyro, enc_acce, cond_enc, input_size_half, input_size) -> None:
        super().__init__()
        self.gyro = enc_gyro # no_hidden
        self.acce = enc_acce # no_hidden
        self.input_size_half = input_size_half
        self.input_size = input_size
        self.cond_enc = cond_enc
        self.rnn_size = self.gyro.rnn_size + self.acce.rnn_size
    
    def forward(self, x, *args, **kwargs):
        if self.cond_enc:
            out_gyro = self.gyro(torch.cat((x[..., :self.input_size_half], x[..., self.input_size:]), dim=-1))
        else:
            out_gyro = self.gyro(x[..., :self.input_size_half])
        out_acce = self.acce(x[..., self.input_size_half:])
        return torch.cat((out_gyro, out_acce), dim=-1)

class ExtraNone(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs), None