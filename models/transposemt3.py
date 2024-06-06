import ipdb as pdb
import torch
import torch.nn as nn
from math import ceil
from basictorch_v2.losses import loss_funcs, adv_losses
from basictorch_v2.tools import torchDevice, data_to_device
from basictorch_v2.dataset import merge_batch
from data.tools import gram, pred_loss, polar_to_offset, remove_none_from_losses
from .modules import CModel, RNNModel, PRNNModel, Discriminator, CatRNNModel, noise, Swapaxes, Newaxes, ExtraNone, StackEncoder
import numpy as np

class TransPoseMT3(nn.Module):
    def __init__(self, name, hyperparams, domain_num, hidden_size, proj_size, num_layers, dropout=0.05, adv_loss='js', label_type='polar', label_mode='sparse', loss_func='mse', trans_func=None, **kwargs):
        super().__init__()
        self.name = name
        self.hps = hyperparams
        self.domain_num = domain_num
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.adv_loss = adv_loss
        self.label_mode = label_mode
        self.label_type = label_type
        self.label_dim = 3 if self.label_type == 'polarHW' else 2
        self.group_size = 2
        self.base_plane = 32
        self.loss_func = loss_func
        self.mee_trans_func = polar_to_offset
        self.stage = -1
        # if self.hps['embedding_label']:
        #     self.domain_embedding = nn.Parameter(torch.randn(domain_num, self.hps['window_size'], self.proj_size))
        # else:
        self.domain_embedding = torch.eye(domain_num, device=torchDevice())

        self.acc_feature_size = 4 if self.hps['acc_magn'] else 3
        self.gys_feature_size = 4 if self.hps['yaw_diff'] else 3
        self.feature_size = self.acc_feature_size + self.gys_feature_size
        self.is_flat = self.hps['is_flat']

        self.enc = self.get_encoder()
        self.gen = self.get_generator(self.hps['gen_type'], self.hps['gen_bidirect'])
        self.dec = self.get_decoder(self.hps['dec_type'], self.hps['dec_bidirect'])
        self.dis = self.get_discriminator()
        self.pred = self.get_predictor()
    
    def get_rnn(self, model_type, input_size, hidden_size, num_layers, bidirectional, proj_size, dropout, no_hidden, activation=None, last_squeeze=False, linear_head=False, is_flat=True):
        if is_flat:
            return RNNModel(model_type, input_size, hidden_size, num_layers, True, 0, bidirectional, proj_size, dropout, no_hidden, activation, last_squeeze, linear_head)
        else:
            model1 = RNNModel(model_type, input_size, hidden_size, num_layers, True, 0, bidirectional, proj_size[0], dropout, no_hidden, activation, last_squeeze, linear_head)
            model2 = RNNModel(model_type, input_size, hidden_size, num_layers, True, 0, bidirectional, proj_size[1], dropout, no_hidden, activation, last_squeeze, linear_head)
            return PRNNModel(model1, model2)

    def get_encoder(self):
        if self.is_flat or self.hps['enc_flat']:
            input_size = self.feature_size if not self.hps['cond_enc'] else self.feature_size + self.domain_num
            encoder = self.get_rnn(self.hps['enc_type'], input_size, self.hidden_size, self.num_layers[0], self.hps['enc_bidirect'], self.proj_size*2, self.dropout, True, self.hps['activation'], False, False)
        else:
            if self.hps['enc_fuse']:
                input_size_gyro = self.gys_feature_size if not self.hps['cond_enc'] else self.gys_feature_size + self.domain_num
                input_size_acce = self.acc_feature_size if not self.hps['cond_enc'] else self.acc_feature_size + self.domain_num
                enc_gyro = self.get_rnn(self.hps['enc_type'], input_size_gyro, self.hidden_size, self.num_layers[0], self.hps['enc_bidirect'], self.proj_size, self.dropout, True, self.hps['activation'], False, False, True)
                enc_acce = self.get_rnn(self.hps['enc_type'], input_size_acce, self.hidden_size, self.num_layers[0], self.hps['enc_bidirect'], self.proj_size, self.dropout, True, self.hps['activation'], False, False, True)
                enc_fuse = self.get_rnn(self.hps['enc_type'], enc_gyro.rnn_size+enc_acce.rnn_size, self.hidden_size, self.num_layers[0], self.hps['enc_bidirect'], self.proj_size*2, self.dropout, True, self.hps['activation'], False, False, True)
                return StackEncoder(enc_gyro, enc_acce, enc_fuse, self.hps['cond_enc'], self.gys_feature_size)
            else:
                input_size_gyro = self.gys_feature_size if not self.hps['cond_enc'] else self.gys_feature_size + self.domain_num
                input_size_acce = self.acc_feature_size if not self.hps['cond_enc'] else self.acc_feature_size + self.domain_num
                enc_gyro = self.get_rnn(self.hps['enc_type'], input_size_gyro, self.hidden_size, self.num_layers[0], self.hps['enc_bidirect'], self.proj_size, self.dropout, True, self.hps['activation'], False, False)
                enc_acce = self.get_rnn(self.hps['enc_type'], input_size_acce, self.hidden_size, self.num_layers[0], self.hps['enc_bidirect'], self.proj_size, self.dropout, True, self.hps['activation'], False, False)
                encoder = CatRNNModel(enc_gyro, enc_acce, self.hps['cond_enc'], self.gys_feature_size, self.gys_feature_size+self.acc_feature_size)
        return ExtraNone(CModel(encoder, self.domain_embedding) if self.hps['cond_enc'] else encoder)
    
    def get_decoder(self, dec_type, bidirectional):
        input_size = self.proj_size*2+self.domain_num if self.is_flat else [self.proj_size+self.domain_num, self.proj_size+self.domain_num]
        proj_size = self.gys_feature_size + self.acc_feature_size if self.is_flat else [self.gys_feature_size, self.acc_feature_size]
        if self.is_flat:
            decoder = self.get_rnn(dec_type, input_size, self.hidden_size, self.num_layers[0], bidirectional, proj_size, self.dropout, True, None, False, False)
        else:
            dec_gyro = self.get_rnn(dec_type, input_size[0], self.hidden_size, self.num_layers[0], bidirectional, proj_size[0], self.dropout, True, None, False, False)
            dec_acce = self.get_rnn(dec_type, input_size[1], self.hidden_size, self.num_layers[0], bidirectional, proj_size[1], self.dropout, True, None, False, False)
            decoder = CatRNNModel(dec_gyro, dec_acce, True, self.proj_size, self.proj_size*2)
        return CModel(decoder, self.domain_embedding)
    
    def get_generator(self, gen_type, bidirectional):
        input_size = self.proj_size*2+self.domain_num
        proj_size = self.gys_feature_size + self.acc_feature_size if self.is_flat else [self.gys_feature_size, self.acc_feature_size]
        generator = self.get_rnn(gen_type, input_size, self.hidden_size, self.num_layers[0], bidirectional, proj_size, self.dropout, True, None, False, False, self.is_flat)
        return CModel(generator, self.domain_embedding)
        
    def get_discriminator(self):
        dis = RNNModel(self.hps['dis_type'], self.feature_size, self.hidden_size, self.num_layers[1], True, self.dropout, self.hps['dis_bidirect'], self.domain_num, self.dropout, True, 'sigmoid', True, self.hps['dis_linear'])
        return Discriminator(dis, self.hps['grl'])
    
    def get_predictor(self):
        proj_size = [1] * self.label_dim if not self.is_flat else self.label_dim
        return self.get_rnn(self.hps['pred_type'], self.proj_size*2, self.hidden_size, self.num_layers[2], self.hps['pred_bidirect'], proj_size, self.dropout, True, None, False, self.hps['pred_linear'], self.is_flat)

    def get_d(self, data, i, random=False):
        if random:
            r_d = torch.randint(self.domain_num-1, (data.shape[0], 1), device=data.device)
            r_d[r_d>=i] += 1
            return r_d
        else:
            return torch.zeros(data.shape[0], 1, dtype=torch.long, device=data.device, requires_grad=False) + i
    
    def extract_feature(self, x, *args, **kwargs):
        with torch.no_grad():
            if self.hps['cond_enc']:
                d = self.get_d(x, args[0])
                return self.enc(x, d)[0]
            else:
                return self.enc(x)[0]

    def predict(self, x, *args, **kwargs):
        with torch.no_grad():
            if self.hps['cond_enc']:
                d = self.get_d(x, args[0])
                return self.pred(self.enc(x, d)[0])
            else:
                return self.pred(self.enc(x)[0])

    def get_loss_d(self, s_x, t_s_x, t_x, s_t_x, s_d, t_s_d, t_d, s_t_d):
        p_s_real = self.dis(s_x, s_d)
        p_s_fake = self.dis(t_s_x, t_s_d)
        p_t_real = self.dis(t_x, t_d)
        p_t_fake = self.dis(s_t_x, s_t_d)
        loss_d_s = adv_losses[self.adv_loss](p_s_real, 1.0) + adv_losses[self.adv_loss](p_s_fake, 0.0)
        loss_d_t = adv_losses[self.adv_loss](p_t_real, 1.0) + adv_losses[self.adv_loss](p_t_fake, 0.0)
        return loss_d_s + loss_d_t
    
    def eval_pred_loss(self, s_y, s_y_p, t_y, t_y_p, tar_batch_data):
        with torch.no_grad():
            mse = pred_loss(s_y[..., :self.label_dim], s_y_p, 'sparse', 'mse', None, True)
            mseRho = pred_loss(s_y[..., 0], s_y_p[..., 0], 'sparse', 'mse', None, True)
            msePsi = pred_loss(s_y[..., 1], s_y_p[..., 1], 'sparse', 'mse', None, True)
            mee = pred_loss(s_y[..., [0, -1]], s_y_p[..., [0, -1]], 'sparse', 'mee', self.mee_trans_func, True)
            tar_ends = np.cumsum([_[0].shape[0] for _ in tar_batch_data])
            tar_starts = np.hstack(([0], tar_ends[:-1]))
            t_mse = torch.mean(torch.tensor([pred_loss(t_y[s:e, ..., :self.label_dim], t_y_p[s:e], 'sparse', 'mse', None, True) for s,e in zip(tar_starts, tar_ends)]))
            t_mseRho = torch.mean(torch.tensor([pred_loss(t_y[s:e,...,0], t_y_p[s:e,...,0], 'sparse', 'mse', None, True) for s,e in zip(tar_starts, tar_ends)]))
            t_msePsi = torch.mean(torch.tensor([pred_loss(t_y[s:e,...,1], t_y_p[s:e,...,1], 'sparse', 'mse', None, True) for s,e in zip(tar_starts, tar_ends)]))
            t_mee = torch.mean(torch.tensor([pred_loss(t_y[s:e, ..., [0, -1]], t_y_p[s:e, ..., [0, -1]], 'sparse', 'mee', self.mee_trans_func, True) for s,e in zip(tar_starts, tar_ends)]))

            if self.label_dim == 3:
                msePhi = pred_loss(s_y[..., -1], s_y_p[..., -1], 'sparse', 'mse', None, True)
                t_msePhi = torch.mean(torch.tensor([pred_loss(t_y[s:e,...,-1], t_y_p[s:e,...,-1], 'sparse', 'mse', None, True) for s,e in zip(tar_starts, tar_ends)]))
            else:
                msePhi, t_msePhi = None, None # polarHW
            return mse, mseRho, msePsi, msePhi, mee, t_mse, t_mseRho, t_msePsi, t_msePhi, t_mee
    
    def forward(self, batch_data, *args):
        batch_data = data_to_device(batch_data)
        ([[s_x, s_y, s_l, s_i]], tar_batch_data) = batch_data
        t_x, t_y, t_l, t_i = merge_batch(tar_batch_data)
        s_d, t_d = self.get_d(s_x, 0), torch.cat([self.get_d(_[0], i+1) for _,i in zip(tar_batch_data, range(len(tar_batch_data)))], dim=0)
        s_t_d = self.get_d(s_x, 0, True)
        if self.hps['t_s_random']:
            t_s_d = torch.cat([self.get_d(_[0], i+1, True) for _,i in zip(tar_batch_data, range(len(tar_batch_data)))], dim=0)
        else:
            t_s_d = torch.cat([self.get_d(_[0], 0) for _,i in zip(tar_batch_data, range(len(tar_batch_data)))], dim=0)
        
        if self.stage == 1:
            if self.hps['cond_enc']:
                s_z,_ = self.enc(s_x, s_d)
                t_z,_ = self.enc(t_x, t_d)
            else:
                s_z,_ = self.enc(s_x)
                t_z,_ = self.enc(t_x)
            s_x_rec = self.dec(noise(s_z, self.hps['noise_sigma']), s_d)
            t_x_rec = self.dec(noise(t_z, self.hps['noise_sigma']), t_d)
            loss_rec   = loss_funcs['mse'](s_x, s_x_rec) + loss_funcs['mse'](t_x, t_x_rec)
            if self.hps['dec_type'] != 'share':
                s_x_rec2 = self.gen(noise(s_z, self.hps['noise_sigma']), s_d)
                t_x_rec2 = self.gen(noise(t_z, self.hps['noise_sigma']), t_d)
                loss_rec += loss_funcs['mse'](s_x, s_x_rec2) + loss_funcs['mse'](t_x, t_x_rec2)
            s_y_p = self.pred(s_z)
            with torch.no_grad():
                t_y_p = self.pred(t_z)
            loss_pred_s = pred_loss(s_y[..., :self.label_dim], s_y_p, self.label_mode, self.loss_func, None, True, self.hps['step_size'])
            loss = self.hps['pred'] * loss_pred_s + self.hps['ae'] * loss_rec
            mse, mseRho, msePsi, msePhi, mee, t_mse, t_mseRho, t_msePsi, t_msePhi, t_mee = self.eval_pred_loss(s_y, s_y_p, t_y, t_y_p, tar_batch_data)
            losses = {'loss':loss, 'L_rec':loss_rec, 'L_pred':loss_pred_s, 'mse':mse, 't_mse':t_mse, 'mseRho':mseRho, 't_mseRho':t_mseRho, 'msePsi':msePsi, 't_msePsi':t_msePsi, 'msePhi':msePhi, 't_msePhi':t_msePhi, 'mee':mee, 't_mee':t_mee}
            losses = remove_none_from_losses(losses)
            return losses
        else:
            if self.hps['cond_enc']:
                s_z,s_sz = self.enc(s_x, s_d)
                t_z,t_sz = self.enc(t_x, t_d)
            else:
                s_z,s_sz = self.enc(s_x)
                t_z,t_sz = self.enc(t_x)
            s_x_rec = self.dec(noise(s_z, self.hps['noise_sigma']), s_d)
            t_x_rec = self.dec(noise(t_z, self.hps['noise_sigma']), t_d)
            s_t_x = self.gen(s_z, s_t_d)
            t_s_x = self.gen(t_z, t_s_d)
            if self.hps['cond_enc']:
                s_t_z,s_t_sz = self.enc(s_t_x, s_t_d)
                t_s_z,t_s_sz = self.enc(t_s_x, t_s_d)
            else:
                s_t_z,s_t_sz = self.enc(s_t_x)
                t_s_z,t_s_sz = self.enc(t_s_x)
            s_t_s_x = self.gen(s_t_z, s_d)
            t_s_t_x = self.gen(t_s_z, t_d)
            s_y_p = self.pred(s_z)
            t_y_p = self.pred(t_z)
            s_t_y_p = self.pred(s_t_z)

            loss_rec = loss_funcs['mse'](s_x, s_x_rec) + loss_funcs['mse'](t_x, t_x_rec)
            if self.hps['dec_type'] != 'share':
                s_x_rec2 = self.gen(noise(s_z, self.hps['noise_sigma']), s_d)
                t_x_rec2 = self.gen(noise(t_z, self.hps['noise_sigma']), t_d)
                loss_rec += loss_funcs['mse'](s_x, s_x_rec2) + loss_funcs['mse'](t_x, t_x_rec2)
            loss_pred_s = pred_loss(s_y[..., :self.label_dim], s_y_p, self.label_mode, self.loss_func, None, True, self.hps['step_size'])
            loss_pred_s_t = pred_loss(s_y[..., :self.label_dim], s_t_y_p, 'sparse', self.loss_func, None, True, self.hps['step_size'])
            loss_pred = loss_pred_s + self.hps['pred_st'] * loss_pred_s_t
            
            loss = self.hps['ae'] * loss_rec + self.hps['pred'] * loss_pred
            if self.hps['cycx']>0:
                loss_cycx = loss_funcs[self.hps['cycx_type']](s_x, s_t_s_x) + loss_funcs[self.hps['cycx_type']](t_x, t_s_t_x)
                loss += self.hps['cycx'] * loss_cycx
            else:
                loss_cycx = torch.tensor(0)
            if self.hps['cycz']>0:
                if self.hps['cycz_type'] == 'fm':
                    loss_cycz = loss_funcs['rmse'](s_z, s_t_z) + loss_funcs['rmse'](t_z, t_s_z)
                else: #gram
                    loss_cycz = loss_funcs['rmse'](gram(s_z), gram(s_t_z)) + loss_funcs['rmse'](gram(t_z), gram(t_s_z))
                loss += self.hps['cycz'] * loss_cycz
            else:
                loss_cycz = torch.tensor(0)
            if self.hps['cycsz']>0 and (s_sz is not None):
                if self.hps['cycsz_type'] == 'fm':
                    loss_cycsz = loss_funcs['mse'](s_sz, s_t_sz) + loss_funcs['mse'](t_sz, t_s_sz)
                else: # gram
                    loss_cycsz = loss_funcs['mse'](gram(s_sz), gram(s_t_sz)) + loss_funcs['mse'](gram(t_sz), gram(t_s_sz))
                loss += self.hps['cycsz'] * loss_cycsz
            else:
                loss_cycsz = torch.tensor(0)
            loss_wod = loss
            if self.hps['cgan']>0:
                loss_d = self.get_loss_d(s_x, t_s_x, t_x, s_t_x, s_d, t_s_d, t_d, s_t_d)
                loss = loss_wod + self.hps['cgan'] * loss_d
            else:
                loss_d = torch.tensor(0)
            
            mse, mseRho, msePsi, msePhi, mee, t_mse, t_mseRho, t_msePsi, t_msePhi, t_mee = self.eval_pred_loss(s_y, s_y_p, t_y, t_y_p, tar_batch_data)
            losses = {'loss':loss, 'L_wod':loss_wod, 'L_rec':loss_rec, 'L_cgan':loss_d, 'L_cycx':loss_cycx, 'L_cycz':loss_cycz, 'L_cycsz':loss_cycsz, 'L_pred':loss_pred, 'mse':mse, 't_mse':t_mse, 'mseRho':mseRho, 't_mseRho':t_mseRho, 'msePsi':msePsi, 't_msePsi':t_msePsi, 'msePhi':msePhi, 't_msePhi':t_msePhi, 'mee':mee, 't_mee':t_mee} #, 'mse':mse, **dict(zip(['t_mse%d'%i for i in range(1, 4)], t_mse_list))
            losses = remove_none_from_losses(losses)
            return losses

