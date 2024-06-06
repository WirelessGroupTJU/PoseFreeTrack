import ipdb as pdb
import torch
import torch.nn as nn
from basictorch_v2.losses import loss_funcs
from basictorch_v2.tools import data_to_device
from data.tools import polar_to_offset

from .resnet import *
 
_fc_config = {'fc_dim': 512, 'dropout': 0.5, 'trans_planes': 128}
_input_channel, _output_channel = 6, 2

class RoNIN(nn.Module):
    def __init__(self, name, hyperparams, domain_num, hidden_size, *args, **kwargs):
        super().__init__()
        self.name = name
        self.domain_num = domain_num
        self.hps = hyperparams
        self.label_dim = 2
        self.mee_trans_func = polar_to_offset
        _fc_config['in_dim'] = 7 # 200 // 32 + 1
        self.net = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                                base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    
    def predict(self, x, *args, **kwargs):
        with torch.no_grad():
            return self.net(torch.swapaxes(x, -1, -2))[:, None, :]

    def forward(self, batch_data, *args):
        batch_data = data_to_device(batch_data)
        ([[s_x, s_y, s_l, s_i]], tar_batch_data) = batch_data
        s_y_hat = self.net(torch.swapaxes(s_x, -1, -2))
        return {'loss': loss_funcs['mse'](s_y_hat, s_y[:, -1])}
