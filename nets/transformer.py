import copy
from pickle import FALSE
from typing import Optional, List
from numpy import c_

import torch
import torch.nn.functional as F
from torch import nn, Tensor, uint8
from utils.utils import get_context_bone_embeding_from_feature_use_initial_pred_sample_point
from torch.nn import MultiheadAttention 

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SoftMultiHeadAttention_old(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim] 
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]``
        scores -- [h, N, T_q, T_k]
    '''
 
    def __init__(self, query_dim, key_dim, num_units, num_heads, soft_sa_scale = 100, dropout=0.1,sa_pre = True,soft_sa_method = 'multiply'):
 
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout =nn.Dropout(dropout)
        ###soft attention
        self.soft_sa_scale = soft_sa_scale
        self.sa_pre = sa_pre
        self.soft_sa_method = soft_sa_method
        
        self.W_query = nn.Linear(in_features=query_dim, out_features=query_dim, bias=True)
        self.W_key = nn.Linear(in_features=key_dim, out_features=key_dim, bias=True)
        self.W_value = nn.Linear(in_features=num_units, out_features=num_units, bias= True)
        #self._reset_parameters()

    def forward(self, query, key, value, segms, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(value)
 
        q_k_split_size = self.key_dim // self.num_heads
        v_split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
        keys = torch.stack(torch.split(keys, q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
        values = torch.stack(torch.split(values, v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
 
        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        
        if self.sa_pre:
            ## mask
            if segms is not None:
                ## mask:  [N, T_k] --> [h, N, T_q, T_k]
                ## new segms [N,T_q,T_K] --> [h, N, T_q, T_k]
                segms = segms.unsqueeze(0).repeat(self.num_heads,1,1,1)*self.soft_sa_scale
                #scores = scores.masked_fill(segms, -np.inf)
                if self.soft_sa_method == 'multiply':
                    scores = scores * segms
                elif self.soft_sa_method == 'add':
                    scores = scores+segms
                    scores = scores/2

            scores = F.softmax(scores, dim=3)
        else:
            scores = F.softmax(scores, dim=3)
            if segms is not None:
                ## mask:  [N, T_k] --> [h, N, T_q, T_k]
                ## new segms [N,T_q,T_K] --> [h, N, T_q, T_k]
                segms = segms.unsqueeze(0).repeat(self.num_heads,1,1,1)*self.soft_sa_scale
                #scores = scores.masked_fill(segms, -np.inf)
                if self.soft_sa_method == 'multiply':
                    scores = scores * segms
                elif self.soft_sa_method == 'add':
                    scores  = scores+segms
                    scores = scores/2
            

        scores = self.dropout(scores)
        ## out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
 
        return out,scores

class SoftMultiHeadAttention_c_p_fix_bugs(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim] 
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]``
        scores -- [h, N, T_q, T_k]
    '''
 
    def __init__(self, c_d_model, p_d_model,key_dim, num_units, num_heads,soft_sa_scale = 100, dropout=0.1,sa_pre = True,soft_sa_method = 'multiply',two_attention_map=False,sa = True):
 
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = nn.Dropout(dropout)
        self.soft_sa_scale = soft_sa_scale
        self.sa_pre = sa_pre
        self.soft_sa_method = soft_sa_method
        self.two_attention_map = two_attention_map
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        self.sep_proj  = False
        self.sa = sa
        
        if two_attention_map:
            if self.sep_proj:
                self.W_query1 = nn.Linear(in_features=c_d_model, out_features=c_d_model, bias=True)
                self.W_key1 = nn.Linear(in_features=c_d_model, out_features=c_d_model, bias=True)
    
                self.W_query2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias=True)
                self.W_key2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias=True)
            else:
                self.W_query = nn.Linear(in_features=c_d_model+p_d_model, out_features=c_d_model+p_d_model, bias=True)
                self.W_key = nn.Linear(in_features=c_d_model+p_d_model, out_features=c_d_model+p_d_model, bias=True)
                self.W_value = nn.Linear(in_features=c_d_model+p_d_model, out_features=c_d_model+p_d_model, bias=True)
        else:
            self.W_query = nn.Linear(in_features=key_dim, out_features=key_dim, bias=True)
            self.W_key = nn.Linear(in_features=key_dim, out_features=key_dim, bias=True)

        if self.sep_proj:
            self.W_value1 = nn.Linear(in_features=c_d_model, out_features=c_d_model, bias= True)
            self.W_value2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias= True)
            self.out_proj1 = nn.Linear(in_features=c_d_model, out_features=c_d_model, bias= True)
            self.out_proj2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias= True)
        else:
            self.W_value= nn.Linear(in_features=c_d_model+p_d_model, out_features=c_d_model+p_d_model, bias= True)
            self.out_proj = nn.Linear(in_features=c_d_model+p_d_model, out_features=c_d_model+p_d_model, bias= True)


        # self.c2p_scale = nn.Parameter(torch.tensor([[1]*num_heads], dtype=torch.float32), requires_grad=True) 
        # self.c2p_c_scale = nn.Parameter(torch.tensor([[0.2]*num_heads], dtype=torch.float32), requires_grad=True) 
        # self.p2c_c_scale = nn.Parameter(torch.tensor([[0.8]*num_heads], dtype=torch.float32), requires_grad=True) 

        

        #self.c2p_scale = nn.Parameter(torch.tensor([[1]*num_heads], dtype=torch.float32), requires_grad=False) 
        self.c2p_c_scale = nn.Parameter(torch.tensor([[0.2]*num_heads], dtype=torch.float32), requires_grad=False) 
        self.p2c_c_scale = nn.Parameter(torch.tensor([[0.]*num_heads], dtype=torch.float32), requires_grad=False) 

        self._reset_parameters()
    
    def _reset_parameters(self):
        if self.two_attention_map:
            if self.sep_proj:
                nn.init.constant_(self.W_query1.bias, 0.)
                nn.init.constant_(self.W_key1.bias, 0.)
                nn.init.constant_(self.W_query2.bias, 0.)
                nn.init.constant_(self.W_key2.bias, 0.)
            else:
                nn.init.constant_(self.W_query.bias, 0.)
                nn.init.constant_(self.W_key.bias, 0.)
        else:
            nn.init.constant_(self.W_query.bias, 0.)
            nn.init.constant_(self.W_key.bias, 0.)
        if self.sep_proj:
            nn.init.constant_(self.out_proj1.bias, 0.)
            nn.init.constant_(self.out_proj2.bias, 0.)
            nn.init.constant_(self.W_value1.bias, 0.)
            nn.init.constant_(self.W_value2.bias, 0.)
        else:
            nn.init.constant_(self.W_value.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)






    def forward(self, query, key, value, segms, mask=None):
        if not self.two_attention_map:
            querys = self.W_query(query)  # [N, T_q, num_units]
            keys = self.W_key(key)  # [N, T_k, num_units]
            if self.sep_proj:
                c_values = self.W_value1(value[...,:self.c_d_model])
                p_values= self.W_value2(value[...,self.c_d_model:])
            else:
                values = self.W_value(value)
                c_values = values[...,:self.c_d_model]
                p_values = values[...,self.c_d_model:]


            if  not self.sa:
                q_k_split_size = self.c_d_model // self.num_heads
                v_split_size = self.c_d_model // self.num_heads
                querys = torch.stack(torch.split(querys, q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
                keys = torch.stack(torch.split(keys, q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
                c_values = torch.stack(torch.split(c_values, v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
                p_values = torch.stack(torch.split(p_values, v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

                ## score = softmax(QK^T / (d_k ** 0.5))
                c_scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
                scaling = float(q_k_split_size) ** -0.5
                c_scores = c_scores * scaling
                
                c_scores = F.softmax(c_scores, dim=3)


                c_scores = self.dropout(c_scores)
                p_scores = c_scores
                ## out = score * V
                c_out = torch.matmul(c_scores, c_values)  # [h, N, T_q, num_units/h]
                c_out = torch.cat(torch.split(c_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

                p_out = torch.matmul(p_scores, p_values) 
                p_out = torch.cat(torch.split(p_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

            else:
                q_k_split_size = self.p_d_model
                c_v_split_size = self.c_d_model // self.num_heads
                p_v_split_size = self.p_d_model // self.num_heads

                querys = torch.stack(torch.split(querys, q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
                keys = torch.stack(torch.split(keys, q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
                c_values = torch.stack(torch.split(c_values, c_v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
                p_values = torch.stack(torch.split(p_values, p_v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        
                ## score = softmax(QK^T / (d_k ** 0.5))
                p_scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
                scaling = float(q_k_split_size) ** -0.5
                p_scores = p_scores * scaling
                
                p_scores = F.softmax(p_scores, dim=3)

                p_scores = self.dropout(p_scores)
                c_scores = p_scores

                ## out = score * V
                c_out = torch.matmul(c_scores, c_values)  # [h, N, T_q, num_units/h]
                c_out = torch.cat(torch.split(c_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

                p_out = torch.matmul(p_scores, p_values) 
                p_out = torch.cat(torch.split(p_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

            if self.sep_proj:
                out = torch.cat( [self.out_proj1(c_out), self.out_proj2(p_out) ] ,dim=-1 )
            else:
                out = self.out_proj(torch.cat([c_out,p_out],dim=-1))
            return out,c_scores.mean(0)
        
        else:
            if self.sep_proj:
                c_query = query[...,:self.c_d_model]
                p_query = query[...,self.c_d_model:]
                c_key = key[...,:self.c_d_model]
                p_key = key[...,self.c_d_model:]
                c_value = value[...,:self.c_d_model]
                p_value = value[...,self.c_d_model:]

                c_query = self.W_query1(c_query) 
                p_query = self.W_query2(p_query) 
                c_key = self.W_key1(c_key)  #[N, T_k, num_units]
                p_key = self.W_key2(p_key)
                c_value = self.W_value1(c_value)
                p_value = self.W_value2(p_value)
            else:
                query = self.W_query(query) 
                key = self.W_key(key)  #[N, T_k, num_units]
                value = self.W_value(value)
            
                c_query = query[...,:self.c_d_model]
                p_query = query[...,self.c_d_model:]
                c_key = key[...,:self.c_d_model]
                p_key = key[...,self.c_d_model:]
                c_value = value[...,:self.c_d_model]
                p_value = value[...,self.c_d_model:]

            
            c_q_k_split_size = self.c_d_model // self.num_heads
            p_q_k_split_size =  self.p_d_model // self.num_heads

            c_query = torch.stack(torch.split(c_query, c_q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
            c_key = torch.stack(torch.split(c_key, c_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
            c_value = torch.stack(torch.split(c_value, c_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

            p_query = torch.stack(torch.split(p_query, p_q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
            p_key = torch.stack(torch.split(p_key, p_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
            p_value = torch.stack(torch.split(p_value, p_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]


            c_scores = torch.matmul(c_query, c_key.transpose(2, 3))  # [h, N, T_q, T_k]
            c_scaling = float(c_q_k_split_size) ** -0.5
            c_scores = c_scores * c_scaling

            
            p_scores = torch.matmul(p_query, p_key.transpose(2, 3))  # [h, N, T_q, T_k]
            p_scaling = float(p_q_k_split_size) ** -0.5
            p_scores = p_scores * p_scaling

            c_scores = F.softmax(c_scores, dim=3)
            p_scores = F.softmax(p_scores, dim=3)

            c_scores = c_scores*self.p2c_c_scale.view(self.num_heads,1,1,1) + p_scores*(1-self.p2c_c_scale.view(self.num_heads,1,1,1) )
            p_scores = c_scores*self.c2p_c_scale.view(self.num_heads,1,1,1) + p_scores*(1-self.c2p_c_scale.view(self.num_heads,1,1,1) )

            c_scores = self.dropout(c_scores)
            p_scores = self.dropout(p_scores)

            c_out =  torch.matmul(c_scores, c_value)  # [h, N, T_q, num_units/h]
            p_out =  torch.matmul(p_scores, p_value)  # [h, N, T_q, num_units/h]

            c_out = torch.cat(torch.split(c_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
            p_out =  torch.cat(torch.split(p_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

            if self.sep_proj:
                out = torch.cat( [self.out_proj1(c_out), self.out_proj2(p_out) ] ,dim=-1 )
            else:
                out = self.out_proj(torch.cat([c_out,p_out],dim=-1))
            return out,c_scores.mean(0)
class SoftMultiHeadAttention_c_p_fix_bugs_v1(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim] 
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]``
        scores -- [h, N, T_q, T_k]
    '''
 
    def __init__(self, c_d_model, p_d_model,key_dim, num_units, num_heads,soft_sa_scale = 100, dropout=0.1,sa_pre = True,soft_sa_method = 'multiply',two_attention_map=False,sa = True):
 
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads

        self.key_dim = key_dim
        self.dropout = nn.Dropout(dropout)
        self.soft_sa_scale = soft_sa_scale
        self.sa_pre = sa_pre
        self.soft_sa_method = soft_sa_method
        self.two_attention_map = two_attention_map
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        self.sep_proj  = False
        self.sa = sa
        self.c_hidden_dim = c_d_model
        
        if two_attention_map:
            if self.sep_proj:
                self.W_query1 = nn.Linear(in_features=c_d_model, out_features=self.c_hidden_dim, bias=True)
                self.W_key1 = nn.Linear(in_features=c_d_model, out_features=self.c_hidden_dim, bias=True)
    
                self.W_query2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias=True)
                self.W_key2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias=True)
            else:
                self.W_query = nn.Linear(in_features=c_d_model+p_d_model, out_features=c_d_model+p_d_model, bias=True)
                self.W_key = nn.Linear(in_features=c_d_model+p_d_model, out_features=c_d_model+p_d_model, bias=True)
                self.W_value = nn.Linear(in_features=c_d_model+p_d_model, out_features=c_d_model+p_d_model, bias=True)

            self.c2p_c_scale = nn.Parameter(torch.tensor([[0.8]*num_heads], dtype=torch.float32), requires_grad=False) 
            self.p2c_c_scale = nn.Parameter(torch.tensor([[0.8]*num_heads], dtype=torch.float32), requires_grad=False) 
        else:
            self.W_query = nn.Linear(in_features=key_dim, out_features=key_dim, bias=True)
            self.W_key = nn.Linear(in_features=key_dim, out_features=key_dim, bias=True)

        if self.sep_proj:
            self.W_value1 = nn.Linear(in_features=c_d_model, out_features=self.c_hidden_dim, bias= True)
            self.W_value2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias= True)
            self.out_proj1 = nn.Linear(in_features=self.c_hidden_dim, out_features=c_d_model, bias= True)
            self.out_proj2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias= True)
        else:
            if not self.sa:
                self.W_value= nn.Linear(in_features=c_d_model+p_d_model, out_features=c_d_model+p_d_model, bias= True)
                self.out_proj = nn.Linear(in_features=c_d_model+p_d_model, out_features=c_d_model+p_d_model, bias= True)
            else:
                self.W_value= nn.Linear(in_features=p_d_model, out_features=p_d_model, bias= True)
                self.out_proj = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias= True)



        # # self.c2p_scale = nn.Parameter(torch.tensor([[1]*num_heads], dtype=torch.float32), requires_grad=True) 
        # self.c2p_c_scale = nn.Parameter(torch.tensor([[0.2]*num_heads], dtype=torch.float32), requires_grad=False) 
        # self.p2c_c_scale = nn.Parameter(torch.tensor([[0.8]*num_heads], dtype=torch.float32), requires_grad=False) 

        

        #self.c2p_scale = nn.Parameter(torch.tensor([[1]*num_heads], dtype=torch.float32), requires_grad=True) 


        self._reset_parameters()
    
    def _reset_parameters(self):
        if self.two_attention_map:
            if self.sep_proj:
                nn.init.constant_(self.W_query1.bias, 0.)
                nn.init.constant_(self.W_key1.bias, 0.)
                nn.init.constant_(self.W_query2.bias, 0.)
                nn.init.constant_(self.W_key2.bias, 0.)
            else:
                nn.init.constant_(self.W_query.bias, 0.)
                nn.init.constant_(self.W_key.bias, 0.)
        else:
            nn.init.constant_(self.W_query.bias, 0.)
            nn.init.constant_(self.W_key.bias, 0.)
        if self.sep_proj:
            nn.init.constant_(self.out_proj1.bias, 0.)
            nn.init.constant_(self.out_proj2.bias, 0.)
            nn.init.constant_(self.W_value1.bias, 0.)
            nn.init.constant_(self.W_value2.bias, 0.)
        else:
            nn.init.constant_(self.W_value.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)






    def forward(self, query, key, value, segms, mask=None):
        if not self.two_attention_map:
            querys = self.W_query(query)  # [N, T_q, num_units]
            keys = self.W_key(key)  # [N, T_k, num_units]
            if self.sep_proj:
                c_values = self.W_value1(value[...,:self.c_d_model])
                p_values= self.W_value2(value[...,self.c_d_model:])
            else:
                values = self.W_value(value)
                c_values = values[...,:self.c_d_model]
                p_values = values[...,self.c_d_model:]


            if  not self.sa:
                q_k_split_size = self.c_hidden_dim // self.num_heads
                v_split_size = self.c_hidden_dim // self.num_heads
                p_v_split_size = self.p_d_model // self.num_heads

                querys = torch.stack(torch.split(querys, q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
                keys = torch.stack(torch.split(keys, q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
                c_values = torch.stack(torch.split(c_values, v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
                p_values = torch.stack(torch.split(p_values, p_v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

                ## score = softmax(QK^T / (d_k ** 0.5))
                c_scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
                scaling = float(q_k_split_size) ** -0.5
                c_scores = c_scores * scaling
                
                c_scores = F.softmax(c_scores, dim=3)


                c_scores = self.dropout(c_scores)
                p_scores = c_scores
                ## out = score * V
                c_out = torch.matmul(c_scores, c_values)  # [h, N, T_q, num_units/h]
                c_out = torch.cat(torch.split(c_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

                p_out = torch.matmul(p_scores, p_values) 
                p_out = torch.cat(torch.split(p_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

            else:
                q_k_split_size = self.p_d_model
                #c_v_split_size = self.c_hidden_dim // self.num_heads
                p_v_split_size = self.p_d_model // self.num_heads
                
                p_values = c_values

                querys = torch.stack(torch.split(querys, q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
                keys = torch.stack(torch.split(keys, q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
                #c_values = torch.stack(torch.split(c_values, c_v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
                p_values = torch.stack(torch.split(p_values, p_v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        
                ## score = softmax(QK^T / (d_k ** 0.5))
                p_scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
                scaling = float(q_k_split_size) ** -0.5
                p_scores = p_scores * scaling
                
                p_scores = F.softmax(p_scores, dim=3)

                p_scores = self.dropout(p_scores)
                c_scores = p_scores

                # c_out = torch.matmul(c_scores, c_values)  # [h, N, T_q, num_units/h]
                # c_out = torch.cat(torch.split(c_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

                p_out = torch.matmul(p_scores, p_values) 
                p_out = torch.cat(torch.split(p_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

            if self.sep_proj:
                out = torch.cat( [self.out_proj1(c_out), self.out_proj2(p_out) ] ,dim=-1 )
            else:
                if self.sa:
                    out = self.out_proj(p_out)
                else:
                    out = self.out_proj(torch.cat([c_out,p_out],dim=-1))
            return out,c_scores.mean(0)
        
        else:
            if self.sep_proj:
                c_query = query[...,:self.c_d_model]
                p_query = query[...,self.c_d_model:]
                c_key = key[...,:self.c_d_model]
                p_key = key[...,self.c_d_model:]
                c_value = value[...,:self.c_d_model]
                p_value = value[...,self.c_d_model:]

                c_query = self.W_query1(c_query) 
                p_query = self.W_query2(p_query) 
                c_key = self.W_key1(c_key)  #[N, T_k, num_units]
                p_key = self.W_key2(p_key)
                c_value = self.W_value1(c_value)
                p_value = self.W_value2(p_value)
            else:
                query = self.W_query(query) 
                key = self.W_key(key)  #[N, T_k, num_units]
                value = self.W_value(value)
            
                c_query = query[...,:self.c_d_model]
                p_query = query[...,self.c_d_model:]
                c_key = key[...,:self.c_d_model]
                p_key = key[...,self.c_d_model:]
                c_value = value[...,:self.c_d_model]
                p_value = value[...,self.c_d_model:]

            
            c_q_k_split_size = self.c_hidden_dim // self.num_heads
            p_q_k_split_size =  self.p_d_model // self.num_heads

            c_query = torch.stack(torch.split(c_query, c_q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
            c_key = torch.stack(torch.split(c_key, c_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
            c_value = torch.stack(torch.split(c_value, c_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

            p_query = torch.stack(torch.split(p_query, p_q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
            p_key = torch.stack(torch.split(p_key, p_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
            p_value = torch.stack(torch.split(p_value, p_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]


            c_scores = torch.matmul(c_query, c_key.transpose(2, 3))  # [h, N, T_q, T_k]
            c_scaling = float(c_q_k_split_size) ** -0.5
            c_scores = c_scores * c_scaling

            
            p_scores = torch.matmul(p_query, p_key.transpose(2, 3))  # [h, N, T_q, T_k]
            p_scaling = float(p_q_k_split_size) ** -0.5
            p_scores = p_scores * p_scaling

            c_scores = F.softmax(c_scores, dim=3)
            p_scores = F.softmax(p_scores, dim=3)

            c_scores = c_scores*self.p2c_c_scale.view(self.num_heads,1,-1,1) + p_scores*(1-self.p2c_c_scale.view(self.num_heads,1,-1,1) )
            p_scores = c_scores*self.c2p_c_scale.view(self.num_heads,1,-1,1) + p_scores*(1-self.c2p_c_scale.view(self.num_heads,1,-1,1) )

            c_scores = self.dropout(c_scores)
            p_scores = self.dropout(p_scores)

            c_out =  torch.matmul(c_scores, c_value)  # [h, N, T_q, num_units/h]
            p_out =  torch.matmul(p_scores, p_value)  # [h, N, T_q, num_units/h]

            c_out = torch.cat(torch.split(c_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
            p_out =  torch.cat(torch.split(p_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

            if self.sep_proj:
                out = torch.cat( [self.out_proj1(c_out), self.out_proj2(p_out) ] ,dim=-1 )
            else:
                out = self.out_proj(torch.cat([c_out,p_out],dim=-1))
            return out,c_scores.mean(0)
        
class SoftMultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim] 
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]``
        scores -- [h, N, T_q, T_k]
    '''
 
    def __init__(self, query_dim, key_dim, num_units, num_heads, hidden_dim = None,soft_sa_scale = 100, dropout=0.1,sa_pre = True,soft_sa_method = 'multiply'):
 
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = nn.Dropout(dropout)
        self.soft_sa_scale = soft_sa_scale
        self.sa_pre = sa_pre
        self.soft_sa_method = soft_sa_method
        
        if hidden_dim is None:
            self.W_query = nn.Linear(in_features=query_dim, out_features=query_dim, bias=True)
            self.W_key = nn.Linear(in_features=key_dim, out_features=key_dim, bias=True)
            self.W_value = nn.Linear(in_features=num_units, out_features=num_units, bias= True)
            self.hidden_dim = query_dim
        else:
            self.W_query = nn.Linear(in_features=query_dim, out_features=hidden_dim, bias=True)
            self.W_key = nn.Linear(in_features=key_dim, out_features=hidden_dim, bias=True)
            self.W_value = nn.Linear(in_features=num_units, out_features=num_units, bias= True) 
            self.hidden_dim = hidden_dim

        
        self.out_proj = nn.Linear(in_features=num_units, out_features=num_units, bias= True)
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.W_query.bias, 0.)
        nn.init.constant_(self.W_key.bias, 0.)
        nn.init.constant_(self.W_value.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
        



    def forward(self, query, key, value, segms, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(value)
 
        q_k_split_size = self.hidden_dim // self.num_heads
        v_split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
        keys = torch.stack(torch.split(keys, q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
        values = torch.stack(torch.split(values, v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
 
        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scaling = float(q_k_split_size) ** -0.5
        scores = scores * scaling
        
        if self.sa_pre:
            ## mask
            if segms is not None:
                ## mask:  [N, T_k] --> [h, N, T_q, T_k]
                ## new segms [N,T_q,T_K] --> [h, N, T_q, T_k]
                segms = segms.unsqueeze(0).repeat(self.num_heads,1,1,1)*self.soft_sa_scale
                #scores = scores.masked_fill(segms, -np.inf)
                if self.soft_sa_method == 'multiply':
                    scores *= segms
                elif self.soft_sa_method == 'add':
                    scores += segms
                    scores = scores/2

            scores = F.softmax(scores, dim=3)
        else:
            scores = F.softmax(scores, dim=3)
            if segms is not None:
                ## mask:  [N, T_k] --> [h, N, T_q, T_k]
                ## new segms [N,T_q,T_K] --> [h, N, T_q, T_k]
                segms = segms.unsqueeze(0).repeat(self.num_heads,1,1,1)*self.soft_sa_scale
                #scores = scores.masked_fill(segms, -np.inf)
                if self.soft_sa_method == 'multiply':
                    scores *= segms
                elif self.soft_sa_method == 'add':
                    scores += segms
                    scores = scores/2


            

        scores = self.dropout(scores)
        ## out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        out = self.out_proj(out)
 
        return out,scores

class SoftMultiHeadAttention_head_c9_p1(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim] 
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]``
        scores -- [h, N, T_q, T_k]
    '''
 
    def __init__(self, c_d_model, p_d_model,key_dim, num_units, num_heads,soft_sa_scale = 100, dropout=0.1,sa_pre = True,soft_sa_method = 'multiply',two_attention_map=False):
 
        super().__init__()
        num_heads = 9
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = nn.Dropout(dropout)
        self.soft_sa_scale = soft_sa_scale
        self.sa_pre = sa_pre
        self.soft_sa_method = soft_sa_method
        self.two_attention_map = two_attention_map
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        
        if two_attention_map:
            self.W_query1 = nn.Linear(in_features=c_d_model, out_features=c_d_model, bias=True)
            self.W_key1 = nn.Linear(in_features=c_d_model, out_features=c_d_model, bias=True)
            
            self.W_query2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias=True)
            self.W_key2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias=True)
            self.W_value2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias= True)
        else:
            self.W_query = nn.Linear(in_features=key_dim, out_features=key_dim, bias=True)
            self.W_key = nn.Linear(in_features=key_dim, out_features=key_dim, bias=True)
            self.W_value = nn.Linear(in_features=c_d_model, out_features=c_d_model, bias= True)

        self.W_value1 = nn.Linear(in_features=c_d_model, out_features=c_d_model, bias= True)
        self.W_value2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias= True)


        self.out_proj1 = nn.Linear(in_features=c_d_model, out_features=c_d_model, bias= True)
        self.out_proj2 = nn.Linear(in_features=p_d_model, out_features=p_d_model, bias= True)

        self.c2p_scale = nn.Parameter(torch.tensor([[1]*num_heads], dtype=torch.float32), requires_grad=True) 
        self.c2p_c_scale = nn.Parameter(torch.tensor([[0.5]*num_heads], dtype=torch.float32), requires_grad=True) 
        self.p2c_c_scale = nn.Parameter(torch.tensor([[0.5]*num_heads], dtype=torch.float32), requires_grad=True) 

        self._reset_parameters()
    
    def _reset_parameters(self):
        if self.two_attention_map:
            nn.init.constant_(self.W_query1.bias, 0.)
            nn.init.constant_(self.W_key1.bias, 0.)
            nn.init.constant_(self.W_query2.bias, 0.)
            nn.init.constant_(self.W_key2.bias, 0.)
        else:
            nn.init.constant_(self.W_query.bias, 0.)
            nn.init.constant_(self.W_key.bias, 0.)

        nn.init.constant_(self.out_proj1.bias, 0.)
        nn.init.constant_(self.out_proj2.bias, 0.)
        nn.init.constant_(self.W_value1.bias, 0.)
        nn.init.constant_(self.W_value2.bias, 0.)





    def forward(self, query, key, value, segms, mask=None):
        if not self.two_attention_map:
            querys = self.W_query(query)  # [N, T_q, num_units]
            keys = self.W_key(key)  # [N, T_k, num_units]
            c_values = self.W_value1(value[...,:self.c_d_model])
            p_values= self.W_value2(value[...,self.c_d_model:])
            if query.shape[-1] == self.c_d_model:
                q_k_split_size = self.c_d_model // self.num_heads
                v_split_size = self.c_d_model // self.num_heads
                querys = torch.stack(torch.split(querys, q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
                keys = torch.stack(torch.split(keys, q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
                c_values = torch.stack(torch.split(c_values, v_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        
                ## score = softmax(QK^T / (d_k ** 0.5))
                c_scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
                scaling = float(q_k_split_size) ** -0.5
                c_scores = c_scores * scaling
                
                c_scores = F.softmax(c_scores, dim=3)


                c_scores = self.dropout(c_scores)
                #p_scores = (c_scores*self.c2p_scale.view(self.num_heads,1,1,1)).mean(0)
                p_scores = (c_scores).mean(0)


                ## out = score * V
                c_out = torch.matmul(c_scores, c_values)  # [h, N, T_q, num_units/h]
                c_out = torch.cat(torch.split(c_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

                p_out = torch.matmul(p_scores, p_values) 



                out = torch.cat( [self.out_proj1(c_out), self.out_proj2(p_out) ] ,dim=-1 )
        
                return out,c_scores.mean(0)
            else:
                q_k_split_size = self.p_d_model
                v_split_size = self.c_d_model
                # querys = torch.stack(torch.split(querys, q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
                # keys = torch.stack(torch.split(keys, q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
        
                ## score = softmax(QK^T / (d_k ** 0.5))
                p_scores = torch.matmul(querys, keys.transpose(-2,-1))  # [h, N, T_q, T_k]
                scaling = float(q_k_split_size) ** -0.5
                p_scores = p_scores * scaling
                
                p_scores = F.softmax(p_scores, dim=-1)


                p_scores = self.dropout(p_scores)
                #p_scores = (c_scores*self.c2p_scale.view(self.num_heads,1,1,1)).mean(0)

                ## out = score * V
                c_out = torch.matmul(p_scores, c_values)  # [h, N, T_q, num_units/h]
                p_out = torch.matmul(p_scores, p_values)  # [h, N, T_q, num_units/h]

                out = torch.cat( [self.out_proj1(c_out), self.out_proj2(p_out) ] ,dim=-1 )
        
                return out,p_scores.mean(0)
        else:
            c_query = query[...,:self.c_d_model]
            p_query = query[...,self.c_d_model:]
            c_key = key[...,:self.c_d_model]
            p_key = key[...,self.c_d_model:]
            c_value = value[...,:self.c_d_model]
            p_value = value[...,self.c_d_model:]

            c_query = self.W_query1(c_query) 
            p_query = self.W_query2(p_query) 
            c_key = self.W_key1(c_key)  #[N, T_k, num_units]
            p_key = self.W_key2(p_key)
            c_value = self.W_value1(c_value)
            p_value = self.W_value2(p_value)
            
            c_q_k_split_size = self.c_d_model // self.num_heads
            p_q_k_split_size =  self.p_d_model

            c_query = torch.stack(torch.split(c_query, c_q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
            c_key = torch.stack(torch.split(c_key, c_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
            c_value = torch.stack(torch.split(c_value, c_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

            p_query = torch.stack(torch.split(p_query, p_q_k_split_size, dim=2), dim=0)  # [h, N, T_q, query_dim/h]
            p_key = torch.stack(torch.split(p_key, p_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, key_dim/h]
            #p_value = torch.stack(torch.split(c_value, p_q_k_split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]


            c_scores = torch.matmul(c_query, c_key.transpose(2, 3))  # [h, N, T_q, T_k]
            c_scaling = float(c_q_k_split_size) ** -0.5
            c_scores = c_scores * c_scaling

            
            p_scores = torch.matmul(p_query, p_key.transpose(2, 3))  # [h, N, T_q, T_k]
            p_scaling = float(p_q_k_split_size) ** -0.5
            p_scores = p_scores * p_scaling

            # c_scores = F.softmax(c_scores, dim=3)
            # p_scores = F.softmax(p_scores, dim=3)

            # c_scores = c_scores*self.c2p_c_scale.view(self.num_heads,1,1,1) + p_scores*(1-self.c2p_c_scale.view(self.num_heads,1,1,1) )
            # p_scores = c_scores*self.p2c_c_scale.view(self.num_heads,1,1,1) + p_scores*(1-self.p2c_c_scale.view(self.num_heads,1,1,1) )

            # c_scores = self.dropout(c_scores)
            # p_scores = self.dropout(p_scores)

            # c_out =  torch.matmul(c_scores, c_value)  # [h, N, T_q, num_units/h]
            # p_out =  torch.matmul(p_scores, p_value.unsqueeze(0)).mean(0,keepdim=True)  # [h, N, T_q, num_units/h]

            # c_out = torch.cat(torch.split(c_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
            # p_out =  torch.cat(torch.split(p_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

            # out = torch.cat( [self.out_proj1(c_out), self.out_proj2(p_out) ] ,dim=-1 )

            scores = p_scores + c_scores
            scores = F.softmax(scores, dim=3)
            scores = self.dropout(scores)
            c_scores = scores
            p_scores = scores.mean(0)

            c_out =  torch.matmul(c_scores, c_value)  # [h, N, T_q, num_units/h]
            p_out =  torch.matmul(p_scores, p_value)# [h, N, T_q, num_units/h]

            c_out = torch.cat(torch.split(c_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

            out = torch.cat( [self.out_proj1(c_out), self.out_proj2(p_out) ] ,dim=-1 )

            return out,c_scores.mean(0)



class TransformerEncoder_Soft_SA_variant_vis_att_map_diff_dim(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, memory,pos):
        feature_list = []

        for i,layer in enumerate(self.layers):
            output = layer(memory[-i-1], pos=pos[-i-1])
            feature_list.append(output)

        if self.norm is not None:
            output = self.norm(output)

        return feature_list
    
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        
        if isinstance(memory,list):
            for i,layer in enumerate(self.layers):
                output = layer(output, memory[-i-1], tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask[-i-1],
                            pos=pos[-i-1], query_pos=query_pos) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                if self.return_intermediate:
                    if self.norm is not None:
                        intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)
        else:
            for layer in self.layers:
                output = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                if self.return_intermediate:
                    if self.norm is not None:
                        intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
   
class TransformerDecoderROI(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.nhead = 8

    def forward(self, tgt, memory,            #分别为形状同quries的0向量;backbone出来的特征，
                tgt_mask: Optional[Tensor] = None,  #None
                memory_mask: Optional[Tensor] = None,  #手物query数量的mask,
                tgt_key_padding_mask: Optional[Tensor] = None, #None
                memory_key_padding_mask: Optional[Tensor] = None, #backbone的mask,
                pos: Optional[Tensor] = None,       #特征的位置编码
                query_pos: Optional[Tensor] = None,  #backbone特征的位置编码，query的权重
                ):
        output = tgt    

        intermediate = []

        memory_mask = memory_mask.flatten(2, 3).unsqueeze(1).expand(-1, self.nhead, -1, -1) #[4, 59, 128, 128] -> [4, 59, 16384] -> [4,1,59,16384] -> [4,8,59,16384]
        memory_mask = memory_mask.flatten(0, 1) #[4, 8, 59, 16384] -> [32, 59, 16384]
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, #output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
                           memory_mask=memory_mask,           ##hs_global；memory:backnone特征;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        #最后一层
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
    
    
class TransformerDecoderLayer_Soft_SA(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                segms,
                pos,pos_query):
        output = tgt

        intermediate = []
        
        if isinstance(memory,list):
            for i,layer in enumerate(self.layers):
                if i>=2:
                    output = layer(output, memory[-i-1], segms[-i-1],   #query, key, value, segms,
                                pos[-i-1],pos_query) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                    if self.return_intermediate:
                        if self.norm is not None:
                            intermediate.append(self.norm(output))
                        else:
                            intermediate.append(output)
                else:
                    output = layer(output, memory[-i-1], None,   #query, key, value, segms,
                                pos[-i-1],pos_query) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                    if self.return_intermediate:
                        if self.norm is not None:
                            intermediate.append(self.norm(output))
                        else:
                            intermediate.append(output)
        else:
            for layer in self.layers:
                output = layer(output, memory, segms,   #query, key, value, segms,
                            pos=pos ) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                if self.return_intermediate:
                    if self.norm is not None:
                        intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
    
# class TransformerDecoderLayer_Soft_SA_variant(nn.Module):

#     def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#         self.return_intermediate = return_intermediate

#     def forward(self, tgt, memory,
#                 segms,
#                 pos):
#         output = tgt

#         intermediate = []
        
#         if isinstance(memory,list):
#             for i,layer in enumerate(self.layers):
#                 if i>=1000: #不走上面，走下面
#                     output = layer(output, memory[-i-1], segms[-i-1],   #query, key, value, segms,
#                                 pos=pos[-i-1]) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

#                 else:
#                     output = layer(output, memory[-i-1], None,   #query, key, value, segms,
#                                 pos=pos[-i-1]) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
#                 if self.return_intermediate:
#                     if self.norm is not None:
#                         if isinstance(self.norm,nn.ModuleList):
#                             hidden_dim = (output.shape[-1])//2
#                             intermediate.append( torch.cat( [self.norm[0](output[...,:hidden_dim]), self.norm[1](output[...,hidden_dim:])],dim=-1 ) )
#                         else:
#                             intermediate.append(self.norm(output))
#                     else:
#                         intermediate.append(output)
                    
#         else:
#             for layer in self.layers:
#                 output = layer(output, memory, segms,   #query, key, value, segms,
#                             pos=pos ) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
#                 if self.return_intermediate:
#                     if self.norm is not None:
#                         intermediate.append(self.norm(output))
#                     else:
#                         intermediate.append(output)


#         #fix bugs
#         if self.norm is not None:
#             if isinstance(self.norm,nn.ModuleList):
#                 hidden_dim = (output.shape[-1])//2
#                 output = torch.cat( [self.norm[0](output[...,:hidden_dim]), self.norm[1](output[...,hidden_dim:])],dim=-1 ) 
#             else:
#                 output = self.norm(output)
#             if self.return_intermediate:
#                 intermediate.pop()
#                 intermediate.append(output)
        
#         ##bugs
#         # if self.norm is not None:
#         #     if isinstance(self.norm,nn.ModuleList):
#         #         intermediate.append( torch.cat([self.norm[0](output[...,:output.shape[-1]//2]), self.norm[1](output[...,output.shape[-1]//2:]) ],dim=-1) )
#         #     else:
#         #         intermediate.append(self.norm(output))
                
#         #     if self.return_intermediate:
#         #         intermediate.pop()
#         #         intermediate.append(output)




#         if self.return_intermediate:
#             return torch.stack(intermediate)

#         return output
    

class TransformerDecoderLayer_Soft_SA_variant_vis_att_map(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                segms,
                pos):
        output = tgt

        intermediate = []
        att_map_list = []
        
        if isinstance(memory,list):
            for i,layer in enumerate(self.layers):
                if i>=1000: #不走上面，走下面
                    output = layer(output, memory[-i-1], segms[-i-1],   #query, key, value, segms,
                                pos=pos[-i-1]) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

                else:
                    output,att_map = layer(output, memory[-i-1], None,   #query, key, value, segms,
                                pos=pos[-i-1]) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                    att_map_list.append(att_map)
                if self.return_intermediate:
                    if self.norm is not None:
                        if isinstance(self.norm,nn.ModuleList):
                            hidden_dim = (output.shape[-1])//2
                            intermediate.append( torch.cat( [self.norm[0](output[...,:hidden_dim]), self.norm[1](output[...,hidden_dim:])],dim=-1 ) )
                        else:
                            intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)
                    
        else:
            for layer in self.layers:
                output = layer(output, memory, segms,   #query, key, value, segms,
                            pos=pos ) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                if self.return_intermediate:
                    if self.norm is not None:
                        intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)


        #fix bugs
        if self.norm is not None:
            if isinstance(self.norm,nn.ModuleList):
                hidden_dim = (output.shape[-1])//2
                output = torch.cat( [self.norm[0](output[...,:hidden_dim]), self.norm[1](output[...,hidden_dim:])],dim=-1 ) 
            else:
                output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        ##bugs
        # if self.norm is not None:
        #     if isinstance(self.norm,nn.ModuleList):
        #         intermediate.append( torch.cat([self.norm[0](output[...,:output.shape[-1]//2]), self.norm[1](output[...,output.shape[-1]//2:]) ],dim=-1) )
        #     else:
        #         intermediate.append(self.norm(output))
                
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)




        if self.return_intermediate:
            return torch.stack(intermediate),att_map_list

        return output


class TransformerDecoderLayer_Soft_SA_variant_vis_att_map_diff_dim(nn.Module):

    def __init__(self, decoder_layer, num_layers, c_d_model, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.c_d_model = c_d_model

    def forward(self, tgt, memory,
                segms,
                pos):
        output = tgt

        intermediate = []
        att_map_list = []
        tgt_content_list = []
        
        if isinstance(memory,list):
            for i,layer in enumerate(self.layers):
                if i>=1000: #不走上面，走下面
                    output = layer(output, memory[-i-1], segms[-i-1],   #query, key, value, segms,
                                pos=pos[-i-1]) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

                else:
                    output,att_map = layer(output, memory[-i-1], None,   #query, key, value, segms,
                                pos=pos[-i-1]) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                    att_map_list.append(att_map)
                    #tgt_content_list.append(tgt_content)
                if self.return_intermediate:
                    if self.norm is not None:
                        if isinstance(self.norm,nn.ModuleList):
                            intermediate.append( torch.cat( [self.norm[0](output[...,:self.c_d_model]), self.norm[1](output[...,self.c_d_model:])],dim=-1 ) )
                        else:
                            intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)
                    
        else:
            for layer in self.layers:
                output = layer(output, memory, segms,   #query, key, value, segms,
                            pos=pos ) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                if self.return_intermediate:
                    if self.norm is not None:
                        intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)


        #fix bugs
        if self.norm is not None:
            if isinstance(self.norm,nn.ModuleList):
                #hidden_dim = (output.shape[-1])//2
                output = torch.cat( [self.norm[0](output[...,:self.c_d_model]), self.norm[1](output[...,self.c_d_model:])],dim=-1 ) 
            else:
                output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        ##bugs
        # if self.norm is not None:
        #     if isinstance(self.norm,nn.ModuleList):
        #         intermediate.append( torch.cat([self.norm[0](output[...,:output.shape[-1]//2]), self.norm[1](output[...,output.shape[-1]//2:]) ],dim=-1) )
        #     else:
        #         intermediate.append(self.norm(output))
                
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)




        if self.return_intermediate:
            return torch.stack(intermediate),att_map_list,tgt_content_list

        return output


class TransformerDecoderLayer_Soft_SA_variant_vis_att_map_diff_dim_v1(nn.Module):

    def __init__(self, decoder_layer, num_layers, c_d_model, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.c_d_model = c_d_model

    def forward(self, tgt, memory,
                segms,
                pos):
        output = tgt

        intermediate = []
        att_map_list = []
        keypoints_list = []
        tgt_pos_list = []
        
        if isinstance(memory,list):
            for i,layer in enumerate(self.layers):
                if i>=1000: #不走上面，走下面
                    output = layer(output, memory[-i-1], segms[-i-1],   #query, key, value, segms,
                                pos=pos[-i-1]) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

                else:
                    output,att_map,keypoints,tgt_pos = layer(output, memory[-i-1], None,   #query, key, value, segms,
                                pos=pos[-i-1]) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                    att_map_list.append(att_map)
                    keypoints_list.append(keypoints) 
                    tgt_pos_list.append(tgt_pos)
                if self.return_intermediate:
                    if self.norm is not None:
                        if isinstance(self.norm,nn.ModuleList):
                            intermediate.append( torch.cat( [self.norm[0](output[...,:self.c_d_model]), self.norm[1](output[...,self.c_d_model:])],dim=-1 ) )
                        else:
                            intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)
                    
        else:
            for layer in self.layers:
                output = layer(output, memory, segms,   #query, key, value, segms,
                            pos=pos ) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                if self.return_intermediate:
                    if self.norm is not None:
                        intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)


        #fix bugs
        if self.norm is not None:
            if isinstance(self.norm,nn.ModuleList):
                #hidden_dim = (output.shape[-1])//2
                output = torch.cat( [self.norm[0](output[...,:self.c_d_model]), self.norm[1](output[...,self.c_d_model:])],dim=-1 ) 
            else:
                output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        ##bugs
        # if self.norm is not None:
        #     if isinstance(self.norm,nn.ModuleList):
        #         intermediate.append( torch.cat([self.norm[0](output[...,:output.shape[-1]//2]), self.norm[1](output[...,output.shape[-1]//2:]) ],dim=-1) )
        #     else:
        #         intermediate.append(self.norm(output))
                
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)




        if self.return_intermediate:
            return torch.stack(intermediate),att_map_list,keypoints_list,tgt_pos_list

        return output

class TransformerDecoderLayer_Soft_SA_variant_vis_att_map_diff_dim_reliability_weight(nn.Module):

    def __init__(self, decoder_layer, num_layers, c_d_model, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.c_d_model = c_d_model

    def forward(self, tgt, memory,
                segms,
                pos):
        output = tgt

        #init_tgt = tgt.clone().detach()
        init_tgt = None

        intermediate = []
        att_map_list = []
        relibility_list = []
        
        if isinstance(memory,list):
            for i,layer in enumerate(self.layers):
                if i>=1000: #不走上面，走下面
                    output = layer(output, memory[-i-1], segms[-i-1],   #query, key, value, segms,
                                pos=pos[-i-1]) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

                else:
                    output,att_map,relibility = layer(output, memory[-i-1], None,   #query, key, value, segms,
                                pos=pos[-i-1],init_tgt=init_tgt) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                    att_map_list.append(att_map)
                    relibility_list.append(relibility)
                if self.return_intermediate:
                    if self.norm is not None:
                        if isinstance(self.norm,nn.ModuleList):
                            intermediate.append( torch.cat( [self.norm[0](output[...,:self.c_d_model]), self.norm[1](output[...,self.c_d_model:])],dim=-1 ) )
                        else:
                            intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)
                    
        else:
            for layer in self.layers:
                output = layer(output, memory, segms,   #query, key, value, segms,
                            pos=pos ) #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
                if self.return_intermediate:
                    if self.norm is not None:
                        intermediate.append(self.norm(output))
                    else:
                        intermediate.append(output)


        #fix bugs
        if self.norm is not None:
            if isinstance(self.norm,nn.ModuleList):
                #hidden_dim = (output.shape[-1])//2
                output = torch.cat( [self.norm[0](output[...,:self.c_d_model]), self.norm[1](output[...,self.c_d_model:])],dim=-1 ) 
            else:
                output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        ##bugs
        # if self.norm is not None:
        #     if isinstance(self.norm,nn.ModuleList):
        #         intermediate.append( torch.cat([self.norm[0](output[...,:output.shape[-1]//2]), self.norm[1](output[...,output.shape[-1]//2:]) ],dim=-1) )
        #     else:
        #         intermediate.append(self.norm(output))
                
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)




        if self.return_intermediate:
            return torch.stack(intermediate),att_map_list,torch.stack(relibility_list)

        return output,relibility_list



class multi_TransformerDecoderLayer_Soft_Attention_variant(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False):
        super().__init__()
        self.self_attn = SoftMultiHeadAttention( query_dim=d_model, key_dim=d_model, num_units=d_model*2, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        self.multihead_attn = SoftMultiHeadAttention(query_dim=d_model, key_dim=d_model, num_units=d_model*2, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model*2, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model*2)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model*2)
        self.norm3 = nn.LayerNorm(d_model*2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0] 
        # tgt = tgt + self.dropout1(tgt2)
        # #print(tgt-q)
        # tgt = self.norm1(tgt)
        # #print(tgt-q)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
        #                            key=self.with_pos_embed(memory, pos),
        #                            value = memory)[0]  #key
    
        # tgt = tgt + self.dropout2(tgt2)
    
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        # tgt = tgt + self.dropout3(tgt2)
       
        # tgt = self.norm3(tgt)
        
        if self.use_pos_embeding_in_ca:
            pass
        else:
            #TODO 除了第一层外去q，v尝试是否加入位置编码
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2 = self.multihead_attn(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)[0]  #key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm1(tgt_content)
            ##TODO 此处position编码会否需要经过normlize存疑     
            tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)       

            #仅仅以位置query作为q,k
            q = k = tgt[:,:,self.d_model:]
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
            tgt = tgt + self.dropout1(tgt2)
        
            ###do not modified
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
            tgt = tgt + self.dropout3(tgt2)
        
            tgt = self.norm3(tgt)
    
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query


# class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
#                  activation="relu", normalize_before=False,use_pos_embeding_in_ca = False):
#         super().__init__()
#         self.self_attn = SoftMultiHeadAttention( query_dim=d_model, key_dim=d_model, num_units=d_model*2, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)

#         self.multihead_attn_content = SoftMultiHeadAttention(query_dim=d_model, key_dim=d_model, num_units=d_model*2, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
#         self.multihead_attn_pos_content = SoftMultiHeadAttention(query_dim=d_model*2, key_dim=d_model*2, num_units=d_model*2, hidden_dim=d_model*2,num_heads=nhead,dropout=dropout, soft_sa_scale=soft_sa_scale)

#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model*2, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model*2)

#         self.norm1_c = nn.LayerNorm(d_model)
#         self.norm2_c = nn.LayerNorm(d_model)
#         self.norm3_c = nn.LayerNorm(d_model)
        
#         self.norm1_p = nn.LayerNorm(d_model)
#         self.norm2_p = nn.LayerNorm(d_model)
#         self.norm3_p = nn.LayerNorm(d_model)
        
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before

#         self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
#         self.d_model = d_model
#         self._reset_parameters()
    
#     def _reset_parameters(self):
#         nn.init.constant_(self.linear1.bias, 0.)
#         nn.init.constant_(self.linear2.bias, 0.)

#     def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
#         if add_pos_method== 'add':
#             return tensor if pos is None else tensor + pos
#         elif add_pos_method== 'concat':
#             return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


#     def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
#                             pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
#         # q = k = self.with_pos_embed(tgt, query_pos)
#         # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
#         #                       key_padding_mask=tgt_key_padding_mask)[0] 
#         # tgt = tgt + self.dropout1(tgt2)
#         # #print(tgt-q)
#         # tgt = self.norm1(tgt)
#         # #print(tgt-q)
        
#         # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
#         #                            key=self.with_pos_embed(memory, pos),
#         #                            value = memory)[0]  #key
    
#         # tgt = tgt + self.dropout2(tgt2)
    
#         # tgt = self.norm2(tgt)
#         # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
#         # tgt = tgt + self.dropout3(tgt2)
       
#         # tgt = self.norm3(tgt)
        
#         if not self.use_pos_embeding_in_ca:
#             tgt_content = tgt[:,:,:self.d_model] #取content query
#             tgt2 = self.multihead_attn_content(query=tgt_content,
#                                     key=memory,
#                                     value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
#                                     segms = segms)[0]  #key
#             tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
#             tgt_content = self.norm1_c(tgt_content)     
#             tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)       

#             #仅仅以位置query作为q,k
#             q = k = tgt[:,:,self.d_model:]
#             ##TODO 尝试加入可见性segms约束
#             tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
#             tgt = tgt + self.dropout1(tgt2)
        
#             ###do not modified
#             tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
#             tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
#             tgt = tgt + self.dropout3(tgt2)
        
#             tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
#         else:
#             #说明是第一层
#             if tgt.shape[-1] == self.d_model:
#                 tgt_content = tgt[:,:,:self.d_model] #取content query
#                 tgt2 = self.multihead_attn_content(query=tgt_content,
#                                         key=memory,
#                                         value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
#                                         segms = segms)[0]  #key
#                 tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
#                 tgt_content = self.norm1_c(tgt_content)     
#                 tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)  
#             #非第一层
#             else:
#                 tgt = tgt #取content query
#                 tgt2 = self.multihead_attn_pos_content(query=tgt,
#                                         key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
#                                         value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
#                                         segms = segms)[0]  #key
#                 tgt = tgt + self.dropout1(tgt2)
#                 tgt = torch.cat((self.norm1_c(tgt[:,:,:self.d_model]),self.norm1_p(tgt[:,:,:self.d_model])),dim=-1)  

#         #仅仅以位置query作为q,k
#         q = k = tgt[:,:,self.d_model:]
#         ##TODO 尝试加入可见性segms约束
#         tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
#         tgt = tgt + self.dropout1(tgt2)
    
#         ###do not modified
#         tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    
#         tgt = tgt + self.dropout3(tgt2)
    
#         tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
#         return tgt

#     def forward_pre(self, tgt, memory,
#                     tgt_mask: Optional[Tensor] = None,
#                     memory_mask: Optional[Tensor] = None,
#                     tgt_key_padding_mask: Optional[Tensor] = None,
#                     memory_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None,
#                     query_pos: Optional[Tensor] = None):
#         tgt2 = self.norm1(tgt)
#         q = k = self.with_pos_embed(tgt2, query_pos)
#         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt2 = self.norm2(tgt)
#         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt2 = self.norm3(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
#         tgt = tgt + self.dropout3(tgt2)
#         return tgt

#     def forward(self, output, memory, segms,   #query, key, value, segms,
#                             pos):
#         if self.normalize_before:
#             return self.forward_pre(output, memory, segms,   #query, key, value, segms,
#                             pos=pos)
#         return self.forward_post(output, memory, segms,   #query, key, value, segms,
#                             pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
# #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
    
class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False):
        super().__init__()
        self.self_attn = SoftMultiHeadAttention( query_dim=d_model, key_dim=d_model, num_units=d_model*2, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)

        self.multihead_attn_content = SoftMultiHeadAttention(query_dim=d_model, key_dim=d_model, num_units=d_model*2, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        self.multihead_attn_pos_content = SoftMultiHeadAttention(query_dim=d_model*2, key_dim=d_model*2, num_units=d_model*2, hidden_dim=d_model*2,num_heads=nhead*2+1,dropout=dropout, soft_sa_scale=soft_sa_scale)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model*2, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model*2)

        self.norm1_c = nn.LayerNorm(d_model)
        self.norm2_c = nn.LayerNorm(d_model)
        self.norm3_c = nn.LayerNorm(d_model)
        
        self.norm1_p = nn.LayerNorm(d_model)
        self.norm2_p = nn.LayerNorm(d_model)
        self.norm3_p = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.d_model = d_model
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0] 
        # tgt = tgt + self.dropout1(tgt2)
        # #print(tgt-q)
        # tgt = self.norm1(tgt)
        # #print(tgt-q)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
        #                            key=self.with_pos_embed(memory, pos),
        #                            value = memory)[0]  #key
    
        # tgt = tgt + self.dropout2(tgt2)
    
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        # tgt = tgt + self.dropout3(tgt2)
       
        # tgt = self.norm3(tgt)
        
        if not self.use_pos_embeding_in_ca:
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2 = self.multihead_attn_content(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)[0]  #key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm1_c(tgt_content)     
            tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)       

            #仅仅以位置query作为q,k
            q = k = tgt[:,:,self.d_model:]
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
            tgt = tgt + self.dropout1(tgt2)
        
            ###do not modified
            tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
            tgt = tgt + self.dropout3(tgt2)
        
            tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        else:
            #说明是第一层
            if tgt.shape[-1] == self.d_model:
                tgt_content = tgt[:,:,:self.d_model] #取content query
                tgt2,att_map = self.multihead_attn_content(query=tgt_content,
                                        key=memory,
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
                tgt_content = self.norm1_c(tgt_content)     
                tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)  
            #非第一层
            else:
                tgt = tgt #取content query
                tgt2,att_map = self.multihead_attn_pos_content(query=tgt,
                                        key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt = tgt + self.dropout1(tgt2)
                tgt = torch.cat((self.norm1_c(tgt[:,:,:self.d_model]),self.norm1_p(tgt[:,:,self.d_model:])),dim=-1)  #fix bugs!

        #仅仅以位置query作为q,k
        q = k = tgt[:,:,self.d_model:]
        ##TODO 尝试加入可见性segms约束
        tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
        tgt = tgt + self.dropout1(tgt2)
    
        ###do not modified
        tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    
        tgt = tgt + self.dropout3(tgt2)
    
        tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        return tgt,att_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

class TransformerEncoderLayer_Soft_SA_variant_vis_att_map_diff_dim(nn.Module):

    def __init__(self, c_d_model, p_d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.c_d_model = c_d_model
        self.p_model = p_d_model
        self.self_attn = SoftMultiHeadAttention(c_d_model+p_d_model, c_d_model+p_d_model,c_d_model, nhead,dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear( c_d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward,  c_d_model)

        self.norm1 = nn.LayerNorm(c_d_model)
        self.norm2 = nn.LayerNorm(c_d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)

    def forward_post(self,
                     memory,pos):
        src = memory
        q = k = self.with_pos_embed(src, pos,'concat')
        src2 = self.self_attn(q, k, value=memory,segms=None )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, memory,pos):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,memory,pos):
        if self.normalize_before:
            return self.forward_pre(memory,pos)
        return self.forward_post(memory,pos)
    
class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim(nn.Module):

    def __init__(self, c_d_model, p_d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False,use_identify_embeding_in_sa = False,joints_pos_embed = None,aggregate_context_in_sa=True):
        super().__init__()
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        self.joints_pos_embed = joints_pos_embed
        self.aggregate_context_in_sa = aggregate_context_in_sa
        #query身份信息（32位）+ 位置信息
        if use_identify_embeding_in_sa:
            self.self_attn = SoftMultiHeadAttention( query_dim=p_d_model + c_d_model, key_dim=p_d_model + c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        else:
            if aggregate_context_in_sa:
                self.self_attn = SoftMultiHeadAttention( query_dim=p_d_model, key_dim=p_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
            else:
                self.self_attn = SoftMultiHeadAttention( query_dim=p_d_model, key_dim=p_d_model, num_units=p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        
        self.multihead_attn_content = SoftMultiHeadAttention(query_dim=c_d_model, key_dim=c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        self.multihead_attn_pos_content = SoftMultiHeadAttention(query_dim=c_d_model+p_d_model, key_dim=c_d_model+p_d_model, num_units=c_d_model+p_d_model, hidden_dim=c_d_model+p_d_model,num_heads=nhead,dropout=dropout, soft_sa_scale=soft_sa_scale)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(c_d_model+p_d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, c_d_model+p_d_model)

        self.norm1_c = nn.LayerNorm(c_d_model)
        self.norm2_c = nn.LayerNorm(c_d_model)
        self.norm3_c = nn.LayerNorm(c_d_model)
        
        self.norm1_p = nn.LayerNorm(p_d_model)
        self.norm2_p = nn.LayerNorm(p_d_model)
        self.norm3_p = nn.LayerNorm(p_d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.use_identify_embeding_in_sa = use_identify_embeding_in_sa
        self.d_model = c_d_model
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0] 
        # tgt = tgt + self.dropout1(tgt2)
        # #print(tgt-q)
        # tgt = self.norm1(tgt)
        # #print(tgt-q)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
        #                            key=self.with_pos_embed(memory, pos),
        #                            value = memory)[0]  #key
    
        # tgt = tgt + self.dropout2(tgt2)
    
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        # tgt = tgt + self.dropout3(tgt2)
       
        # tgt = self.norm3(tgt)
        
        if not self.use_pos_embeding_in_ca:
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2 = self.multihead_attn_content(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)[0]  #key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm1_c(tgt_content)     
            tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)       

            #仅仅以位置query作为q,k
            q = k = tgt[:,:,self.d_model:]
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
            tgt = tgt + self.dropout1(tgt2)
        
            ###do not modified
            tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
            tgt = tgt + self.dropout3(tgt2)
        
            tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        else:
            #说明是第一层
            if tgt.shape[-1] == self.d_model:
                tgt_content = tgt[:,:,:self.d_model] #取content query
                tgt2,att_map = self.multihead_attn_content(query=tgt_content,
                                        key=memory,
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
                tgt_content = self.norm1_c(tgt_content)     
                tgt = torch.cat((tgt_content,self.norm1_p(tgt2[:,:,self.d_model:])),dim=-1)  
            #非第一层
            else:
                tgt = tgt #取content query
                tgt2,att_map = self.multihead_attn_pos_content(query=tgt,
                                        key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt = tgt + self.dropout1(tgt2)
                tgt = torch.cat((self.norm1_c(tgt[:,:,:self.d_model]),self.norm1_p(tgt[:,:,self.d_model:])),dim=-1)   #bug!

        if  self.use_identify_embeding_in_sa:
            #q = k = torch.cat((tgt[:,:,self.d_model:],tgt[:,:,self.d_model-32:self.d_model]),dim=-1)
            q = k = tgt
            #tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
        else:
        #仅仅以位置query作为q,k
            if self.joints_pos_embed is None :
                q = k = tgt[:,:,self.d_model:]
            ##Add pos embeding?
            else:
                q = k = tgt[:,:,self.d_model:] + self.joints_pos_embed.unsqueeze(0)
        
        ##SA时是否汇聚内容信息
        if self.aggregate_context_in_sa:
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0]
        else:
            value = tgt[:,:,self.d_model:]  
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=value, segms = None)[0]


                
        ##SA时是否汇聚内容信息
        if self.aggregate_context_in_sa:
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt[:,:,self.d_model:] = tgt[:,:,self.d_model:] + self.dropout2(tgt2)

    
        ###do not modified
        tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    
        tgt = tgt + self.dropout3(tgt2)
    
        tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        return tgt,att_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query


class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim_Abationtable2Stage1Option2(nn.Module):

    def __init__(self, c_d_model, p_d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False,use_identify_embeding_in_sa = False,joints_pos_embed = None,aggregate_context_in_sa=True):
        super().__init__()
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        self.joints_pos_embed = joints_pos_embed
        self.aggregate_context_in_sa = aggregate_context_in_sa
        #query身份信息（32位）+ 位置信息
        if use_identify_embeding_in_sa:
            self.self_attn = SoftMultiHeadAttention( query_dim=p_d_model + c_d_model, key_dim=p_d_model + c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        else:
            if aggregate_context_in_sa:
                self.self_attn = SoftMultiHeadAttention( query_dim=p_d_model, key_dim=p_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
            else:
                self.self_attn = SoftMultiHeadAttention( query_dim=p_d_model, key_dim=p_d_model, num_units=p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        
        self.multihead_attn_content = SoftMultiHeadAttention(query_dim=c_d_model+p_d_model, key_dim=c_d_model+p_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        self.multihead_attn_pos_content = SoftMultiHeadAttention(query_dim=c_d_model+p_d_model, key_dim=c_d_model+p_d_model, num_units=c_d_model+p_d_model, hidden_dim=c_d_model+p_d_model,num_heads=nhead,dropout=dropout, soft_sa_scale=soft_sa_scale)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(c_d_model+p_d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, c_d_model+p_d_model)

        self.norm1_c = nn.LayerNorm(c_d_model)
        self.norm2_c = nn.LayerNorm(c_d_model)
        self.norm3_c = nn.LayerNorm(c_d_model)
        self.norm4_c = nn.LayerNorm(c_d_model+p_d_model)

        
        self.norm1_p = nn.LayerNorm(p_d_model)
        self.norm2_p = nn.LayerNorm(p_d_model)
        self.norm3_p = nn.LayerNorm(p_d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.use_identify_embeding_in_sa = use_identify_embeding_in_sa
        self.d_model = c_d_model
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0] 
        # tgt = tgt + self.dropout1(tgt2)
        # #print(tgt-q)
        # tgt = self.norm1(tgt)
        # #print(tgt-q)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
        #                            key=self.with_pos_embed(memory, pos),
        #                            value = memory)[0]  #key
    
        # tgt = tgt + self.dropout2(tgt2)
    
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        # tgt = tgt + self.dropout3(tgt2)
       
        # tgt = self.norm3(tgt)
        
        if not self.use_pos_embeding_in_ca:
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2 = self.multihead_attn_content(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)[0]  #key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm1_c(tgt_content)     
            tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)       

            #仅仅以位置query作为q,k
            q = k = tgt[:,:,self.d_model:]
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
            tgt = tgt + self.dropout1(tgt2)
        
            ###do not modified
            tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
            tgt = tgt + self.dropout3(tgt2)
        
            tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        else:
            #说明是第一层
            if tgt.shape[-1] == self.d_model+self.p_d_model+1:
                tgt_content = tgt[:,:,:self.d_model+self.p_d_model] #取content query
                tgt2,att_map = self.multihead_attn_content(query=tgt_content,
                                        key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt_content = tgt_content + self.dropout1(tgt2)
                tgt_content = self.norm4_c(tgt_content)     
                tgt = tgt_content  #torch.cat((tgt_content,self.norm1_p(tgt2[:,:,self.d_model:])),dim=-1)  
            #非第一层
            else:
                tgt = tgt #取content query
                tgt2,att_map = self.multihead_attn_pos_content(query=tgt,
                                        key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt = tgt + self.dropout1(tgt2)
                tgt = torch.cat((self.norm1_c(tgt[:,:,:self.d_model]),self.norm1_p(tgt[:,:,self.d_model:])),dim=-1)   #bug!

        if  self.use_identify_embeding_in_sa:
            #q = k = torch.cat((tgt[:,:,self.d_model:],tgt[:,:,self.d_model-32:self.d_model]),dim=-1)
            q = k = tgt
            #tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
        else:
        #仅仅以位置query作为q,k
            if self.joints_pos_embed is None :
                q = k = tgt[:,:,self.d_model:]
            ##Add pos embeding?
            else:
                q = k = tgt[:,:,self.d_model:] + self.joints_pos_embed.unsqueeze(0)
        
        ##SA时是否汇聚内容信息
        if self.aggregate_context_in_sa:
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0]
        else:
            value = tgt[:,:,self.d_model:]  
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=value, segms = None)[0]


                
        ##SA时是否汇聚内容信息
        if self.aggregate_context_in_sa:
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt[:,:,self.d_model:] = tgt[:,:,self.d_model:] + self.dropout2(tgt2)

    
        ###do not modified
        tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    
        tgt = tgt + self.dropout3(tgt2)
    
        tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        return tgt,att_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query



class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim_guanfang_attention(nn.Module):

    def __init__(self, c_d_model, p_d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False,use_identify_embeding_in_sa = False,joints_pos_embed = None,aggregate_context_in_sa=True):
        super().__init__()
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        self.joints_pos_embed = joints_pos_embed
        self.aggregate_context_in_sa = aggregate_context_in_sa
        # #query身份信息（32位）+ 位置信息
        if aggregate_context_in_sa:
            self.self_attn = MultiheadAttention( p_d_model, nhead, dropout=dropout, bias=True, kdim=p_d_model, vdim=c_d_model+p_d_model,batch_first =True)
        
        self.multihead_attn_content = MultiheadAttention(c_d_model, nhead, dropout=dropout, bias=True, kdim=c_d_model, vdim=c_d_model+p_d_model,batch_first =True)
        self.multihead_attn_pos_content = MultiheadAttention( c_d_model+p_d_model, nhead, dropout=dropout, vdim=c_d_model+p_d_model, bias=True,batch_first =True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(c_d_model+p_d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, c_d_model+p_d_model)

        self.norm1_c = nn.LayerNorm(c_d_model)
        self.norm2_c = nn.LayerNorm(c_d_model)
        self.norm3_c = nn.LayerNorm(c_d_model)
        
        self.norm1_p = nn.LayerNorm(p_d_model)
        self.norm2_p = nn.LayerNorm(p_d_model)
        self.norm3_p = nn.LayerNorm(p_d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.use_identify_embeding_in_sa = use_identify_embeding_in_sa
        self.d_model = c_d_model
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        #说明是第一层
        if tgt.shape[-1] == self.d_model:
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2,att_map = self.multihead_attn_content(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'))#key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm1_c(tgt_content)     
            tgt = torch.cat((tgt_content,self.norm1_p(tgt2[:,:,self.d_model:])),dim=-1)  
        #非第一层
        else:
            tgt = tgt #取content query
            tgt2,att_map = self.multihead_attn_pos_content(query=tgt,
                                    key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'))  #key
            tgt = tgt + self.dropout1(tgt2)
            tgt = torch.cat((self.norm1_c(tgt[:,:,:self.d_model]),self.norm1_p(tgt[:,:,self.d_model:])),dim=-1)   #bug!

        if  self.use_identify_embeding_in_sa:
            #q = k = torch.cat((tgt[:,:,self.d_model:],tgt[:,:,self.d_model-32:self.d_model]),dim=-1)
            q = k = tgt
            #tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
        else:
        #仅仅以位置query作为q,k
            if self.joints_pos_embed is None :
                q = k = tgt[:,:,self.d_model:]
            ##Add pos embeding?
            else:
                q = k = tgt[:,:,self.d_model:] + self.joints_pos_embed.unsqueeze(0)
        
        ##SA时是否汇聚内容信息
        if self.aggregate_context_in_sa:
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt)[0]
        else:
            value = tgt[:,:,self.d_model:]  
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=value)[0]


                
        ##SA时是否汇聚内容信息
        if self.aggregate_context_in_sa:
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt[:,:,self.d_model:] = tgt[:,:,self.d_model:] + self.dropout2(tgt2)


        ###do not modified
        tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)

        tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 

        return tgt,att_map

    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim_head_c9_p1(nn.Module):

    def __init__(self, c_d_model, p_d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False,use_identify_embeding_in_sa = False,joints_pos_embed = None,aggregate_context_in_sa =None):
        super().__init__()
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        self.dim_feedforward = dim_feedforward
        #query身份信息（32位）+ 位置信息
        self.self_attn = SoftMultiHeadAttention_head_c9_p1(c_d_model, p_d_model,key_dim=p_d_model , num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale, two_attention_map = False)

        self.multihead_attn_content = SoftMultiHeadAttention_head_c9_p1(c_d_model, p_d_model, key_dim=c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale,two_attention_map = False)
        self.multihead_attn_pos_content = SoftMultiHeadAttention_head_c9_p1(c_d_model, p_d_model, key_dim=c_d_model+p_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout, soft_sa_scale=soft_sa_scale,two_attention_map=True)

        # Implementation of Feedforward model
        self.linear1_c = nn.Linear(c_d_model, dim_feedforward)
        self.linear1_p = nn.Linear(p_d_model, p_d_model)

        self.dropout = nn.Dropout(dropout)
        self.linear2_c = nn.Linear(dim_feedforward, c_d_model)
        self.linear2_p = nn.Linear(p_d_model, p_d_model)


        self.norm1_c = nn.LayerNorm(c_d_model)
        self.norm2_c = nn.LayerNorm(c_d_model)
        self.norm3_c = nn.LayerNorm(c_d_model)
        
        self.norm1_p = nn.LayerNorm(p_d_model)
        self.norm2_p = nn.LayerNorm(p_d_model)
        self.norm3_p = nn.LayerNorm(p_d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.use_identify_embeding_in_sa = use_identify_embeding_in_sa
        # self.pos_inter_net = GraFormer_warp_in_detr(cfg)
        self.d_model = c_d_model
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear1_c.bias, 0.)
        nn.init.constant_(self.linear2_c.bias, 0.)
        nn.init.constant_(self.linear1_p.bias, 0.)
        nn.init.constant_(self.linear2_p.bias, 0.)

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0] 
        # tgt = tgt + self.dropout1(tgt2)
        # #print(tgt-q)
        # tgt = self.norm1(tgt)
        # #print(tgt-q)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
        #                            key=self.with_pos_embed(memory, pos),
        #                            value = memory)[0]  #key
    
        # tgt = tgt + self.dropout2(tgt2)
    
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        # tgt = tgt + self.dropout3(tgt2)
       
        # tgt = self.norm3(tgt)
        
        if not self.use_pos_embeding_in_ca:
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2 = self.multihead_attn_content(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)[0]  #key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm1_c(tgt_content)     
            tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)       

            #仅仅以位置query作为q,k
            q = k = tgt[:,:,self.d_model:]
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
            tgt = tgt + self.dropout1(tgt2)
        
            ###do not modified
            tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
            tgt = tgt + self.dropout3(tgt2)
        
            tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        else:
            #说明是第一层
            if tgt.shape[-1] == self.d_model:
                tgt_content = tgt[:,:,:self.d_model] #取content query
                tgt2,att_map = self.multihead_attn_content(query=tgt_content,
                                        key=memory,
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
                tgt_content = self.norm1_c(tgt_content)     
                tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)  
            #非第一层
            else:
                tgt = tgt #取content query
                tgt2,att_map = self.multihead_attn_pos_content(query=tgt,
                                        key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt = tgt + self.dropout1(tgt2)
                tgt = torch.cat((self.norm1_c(tgt[:,:,:self.d_model]),self.norm1_p(tgt[:,:,self.d_model:])),dim=-1)   #bug!


        #q = k = tgt
        q = k = tgt[:,:,self.d_model:]
 
        ##TODO 尝试加入可见性segms约束
        
        tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0]
        tgt = tgt + self.dropout1(tgt2)


        # #option 尝试使用gcn建模2d关系
        # tgt2 = self.pos_inter_net(q)
        # tgt[:,:,self.d_model:] = tgt[:,:,self.d_model:] + self.dropout1(tgt2)

            
    
        ###do not modified
        tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
        tgt2 = torch.cat([self.linear1_c(tgt[...,:self.c_d_model]),self.linear1_p(tgt[...,self.c_d_model:])],dim=-1)
        tgt2 = self.dropout(self.activation(tgt2))
        tgt2 = torch.cat([self.linear2_c(tgt2[...,:self.dim_feedforward]),self.linear2_p(tgt2[...,self.dim_feedforward:])],dim=-1)
    
        tgt = tgt + self.dropout3(tgt2)
    
        tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        return tgt,att_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim_fix_bugs(nn.Module):

    def __init__(self, c_d_model, p_d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False,use_identify_embeding_in_sa = False,joints_pos_embed = None,aggregate_context_in_sa=True):
        super().__init__()
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        self.joints_pos_embed = joints_pos_embed
        self.aggregate_context_in_sa = aggregate_context_in_sa
        nhead = 9
        #query身份信息（32位）+ 位置信息
        if use_identify_embeding_in_sa:
            self.self_attn = SoftMultiHeadAttention_c_p_fix_bugs( query_dim=p_d_model + c_d_model, key_dim=p_d_model + c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        else:
            if aggregate_context_in_sa:
                self.self_attn = SoftMultiHeadAttention_c_p_fix_bugs( c_d_model, p_d_model,key_dim=p_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale, two_attention_map = False,sa = True)
            else:
                self.self_attn = SoftMultiHeadAttention_c_p_fix_bugs( query_dim=p_d_model, key_dim=p_d_model, num_units=c_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)

        self.multihead_attn_content = SoftMultiHeadAttention_c_p_fix_bugs(c_d_model, p_d_model, key_dim=c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale,two_attention_map = False,sa = False )
        self.multihead_attn_pos_content = SoftMultiHeadAttention_c_p_fix_bugs(c_d_model, p_d_model, key_dim=c_d_model+p_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout, soft_sa_scale=soft_sa_scale,two_attention_map=True,sa = False )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(c_d_model+p_d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, c_d_model+p_d_model)

        self.norm1_c = nn.LayerNorm(c_d_model)
        self.norm2_c = nn.LayerNorm(c_d_model)
        self.norm3_c = nn.LayerNorm(c_d_model)
        
        self.norm1_p = nn.LayerNorm(p_d_model)
        self.norm2_p = nn.LayerNorm(p_d_model)
        self.norm3_p = nn.LayerNorm(p_d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.use_identify_embeding_in_sa = use_identify_embeding_in_sa
        self.d_model = c_d_model
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0] 
        # tgt = tgt + self.dropout1(tgt2)
        # #print(tgt-q)
        # tgt = self.norm1(tgt)
        # #print(tgt-q)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
        #                            key=self.with_pos_embed(memory, pos),
        #                            value = memory)[0]  #key
    
        # tgt = tgt + self.dropout2(tgt2)
    
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        # tgt = tgt + self.dropout3(tgt2)
       
        # tgt = self.norm3(tgt)
        
        if not self.use_pos_embeding_in_ca:
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2 = self.multihead_attn_content(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)[0]  #key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm1_c(tgt_content)     
            tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)       

            #仅仅以位置query作为q,k
            q = k = tgt[:,:,self.d_model:]
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
            tgt = tgt + self.dropout1(tgt2)
        
            ###do not modified
            tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
            tgt = tgt + self.dropout3(tgt2)
        
            tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        else:
            #说明是第一层
            if tgt.shape[-1] == self.d_model:
                tgt_content = tgt[:,:,:self.d_model] #取content query
                tgt2,att_map = self.multihead_attn_content(query=tgt_content,
                                        key=memory,
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
                tgt_content = self.norm1_c(tgt_content)     
                tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)  
            #非第一层
            else:
                tgt = tgt #取content query
                tgt2,att_map = self.multihead_attn_pos_content(query=tgt,
                                        key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt = tgt + self.dropout1(tgt2)
                tgt = torch.cat((self.norm1_c(tgt[:,:,:self.d_model]),self.norm1_p(tgt[:,:,self.d_model:])),dim=-1)   #bug!

        if  self.use_identify_embeding_in_sa:
            #q = k = torch.cat((tgt[:,:,self.d_model:],tgt[:,:,self.d_model-32:self.d_model]),dim=-1)
            q = k = tgt
            #tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
        else:
        #仅仅以位置query作为q,k
            if self.joints_pos_embed is None :
                q = k = tgt[:,:,self.d_model:]
            ##Add pos embeding?
            else:
                q = k = tgt[:,:,self.d_model:] + self.joints_pos_embed.unsqueeze(0)
        
        ##SA时是否汇聚内容信息
        if self.aggregate_context_in_sa:
            ##TODO 尝试加入可见性segms约束
            #segms = gradient_mask.unsqueeze(0).unsqueeze(0).to(q.device)
            #tgt2 = self.self_attn(q, k, value=tgt, segms = None,mask=(segms==0))[0]
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0]

            #tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0]
        else:
            value = tgt[:,:,self.d_model:]  #fix bugs
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=value, segms = None)[0]


                
        ##SA时是否汇聚内容信息
        if self.aggregate_context_in_sa:
            tgt = tgt + self.dropout2(tgt2)
        else:
            #bugs!
            tgt[:,:,self.d_model:] = tgt[:,:,self.d_model:] + self.dropout2(tgt2)

    
        ###do not modified
        tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    
        tgt = tgt + self.dropout3(tgt2)
    
        tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 

        tgt_content = None
        return tgt,att_map,tgt_content

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
    
class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim_fix_bugs_v1(nn.Module):

    def __init__(self, c_d_model, p_d_model, nhead, hand_keypoint,dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu",normalize_before=False,use_pos_embeding_in_ca = False,use_identify_embeding_in_sa = False,joints_pos_embed = None,aggregate_context_in_sa=True):
        super().__init__()
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        self.joints_pos_embed = joints_pos_embed
        self.aggregate_context_in_sa = aggregate_context_in_sa  
        self.hand_keypoint_layer = hand_keypoint                                                                                                                                                                                                             
        #query身份信息（32位）+ 位置信息
        if use_identify_embeding_in_sa:
            self.self_attn = SoftMultiHeadAttention_c_p_fix_bugs_v1( query_dim=p_d_model + c_d_model, key_dim=p_d_model+ c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        else:
            if aggregate_context_in_sa:
                self.self_attn = SoftMultiHeadAttention_c_p_fix_bugs_v1( c_d_model, p_d_model,key_dim=p_d_model, num_units=p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale, two_attention_map = False,sa = True)
            else:
                self.self_attn = SoftMultiHeadAttention_c_p_fix_bugs_v1( query_dim=p_d_model, key_dim=p_d_model, num_units=c_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)

        self.multihead_attn_content = SoftMultiHeadAttention_c_p_fix_bugs_v1(c_d_model, p_d_model, key_dim=c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale,two_attention_map = False,sa = False )
        self.multihead_attn_pos_content = SoftMultiHeadAttention_c_p_fix_bugs_v1(c_d_model, p_d_model, key_dim=c_d_model+p_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout, soft_sa_scale=soft_sa_scale,two_attention_map=True,sa = False )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(c_d_model+p_d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, c_d_model+p_d_model)

        self.norm1_c = nn.LayerNorm(c_d_model)
        self.norm2_c = nn.LayerNorm(c_d_model)
        self.norm3_c = nn.LayerNorm(c_d_model)
        self.norm4_c= nn.LayerNorm(c_d_model)

        
        self.norm1_p = nn.LayerNorm(p_d_model)
        self.norm2_p = nn.LayerNorm(p_d_model)
        self.norm3_p = nn.LayerNorm(p_d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.use_identify_embeding_in_sa = use_identify_embeding_in_sa
        self.d_model = c_d_model
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0] 
        # tgt = tgt + self.dropout1(tgt2)
        # #print(tgt-q)
        # tgt = self.norm1(tgt)
        # #print(tgt-q)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
        #                            key=self.with_pos_embed(memory, pos),
        #                            value = memory)[0]  #key
    
        # tgt = tgt + self.dropout2(tgt2)
    
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        # tgt = tgt + self.dropout3(tgt2)
       
        # tgt = self.norm3(tgt)
        
        if not self.use_pos_embeding_in_ca:
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2 = self.multihead_attn_content(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)[0]  #key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm1_c(tgt_content)     
            tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)       

            #仅仅以位置query作为q,k
            q = k = tgt[:,:,self.d_model:]
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
            tgt = tgt + self.dropout1(tgt2)
        
            ###do not modified
            tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
            tgt = tgt + self.dropout3(tgt2)
        
            tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        else:
            #说明是第一层
            if tgt.shape[-1] == self.d_model:
                tgt_content = tgt[:,:,:self.d_model] #取content query
                tgt2,att_map = self.multihead_attn_content(query=tgt_content,
                                        key=memory,
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
                tgt_content = self.norm1_c(tgt_content)     
                tgt = torch.cat((tgt_content,self.norm1_p(tgt2[:,:,self.d_model:])),dim=-1)  #add norm1_p
            #非第一层
            else:
                tgt = tgt #取content query
                tgt2,att_map = self.multihead_attn_pos_content(query=tgt,
                                        key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt = tgt + self.dropout1(tgt2)
                tgt = torch.cat((self.norm1_c(tgt[:,:,:self.d_model]),self.norm1_p(tgt[:,:,self.d_model:])),dim=-1)   #bug!

        if  self.use_identify_embeding_in_sa:
            #q = k = torch.cat((tgt[:,:,self.d_model:],tgt[:,:,self.d_model-32:self.d_model]),dim=-1)
            q = k = tgt
            #tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
        else:
        #仅仅以位置query作为q,k
            if self.joints_pos_embed is None :
                q = k = v = tgt[:,:,self.d_model:]
            ##Add pos embeding?
            else:
                q = k = tgt[:,:,self.d_model:] + self.joints_pos_embed.unsqueeze(0)


        tgt2 = self.self_attn(q, k, value=v, segms = None)[0]

  
        tgt[:,:,self.d_model:] = tgt[:,:,self.d_model:] + self.dropout2(tgt2)

        # #0~1 -> 0-256
        # keypoints = self.hand_keypoint_layer(tgt[:,:,self.d_model:])*256
        # spatial_dim = int(memory.shape[1]**0.5)
        # memory = self.norm4_c(memory)
        # memory = memory.reshape(-1,spatial_dim,spatial_dim,self.d_model).permute(0,3,1,2)
        # sample_context = get_context_bone_embeding_from_feature_use_initial_pred_sample_point(keypoints,[memory]).squeeze()
        # tgt[...,:self.d_model] =  tgt[...,:self.d_model] + sample_context
        # tgt_pos = tgt[:,:,self.d_model:]



        #这些最后一层没用到
        ###do not modified
        tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    
        tgt = tgt + self.dropout3(tgt2)
        
        tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 

        #0~1 -> 0-256
        keypoints = self.hand_keypoint_layer(tgt[:,:,self.d_model:])*256
        spatial_dim = int(memory.shape[1]**0.5)
        memory = self.norm4_c(memory)
        memory = memory.reshape(-1,spatial_dim,spatial_dim,self.d_model).permute(0,3,1,2)
        sample_context = get_context_bone_embeding_from_feature_use_initial_pred_sample_point(keypoints,[memory]).squeeze()
        tgt[...,:self.d_model] =  tgt[...,:self.d_model] + sample_context
        tgt_pos = tgt[:,:,self.d_model:]



        return tgt,att_map,keypoints,tgt_pos

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
    

class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim_fix_bugs_first_sa(nn.Module):
    def __init__(self, c_d_model, p_d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False,use_identify_embeding_in_sa = False,joints_pos_embed = None,aggregate_context_in_sa=True):
        super().__init__()
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        self.joints_pos_embed = joints_pos_embed
        self.aggregate_context_in_sa = aggregate_context_in_sa
        nhead = 9
        #query身份信息（32位）+ 位置信息
        if use_identify_embeding_in_sa:
            self.self_attn = SoftMultiHeadAttention_c_p_fix_bugs( query_dim=p_d_model + c_d_model, key_dim=p_d_model + c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        else:
            if aggregate_context_in_sa:
                self.self_attn = SoftMultiHeadAttention_c_p_fix_bugs( c_d_model, p_d_model,key_dim=p_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale, two_attention_map = False,sa = True)
            else:
                self.self_attn = SoftMultiHeadAttention_c_p_fix_bugs( query_dim=p_d_model, key_dim=p_d_model, num_units=c_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)

        self.multihead_attn_content = SoftMultiHeadAttention_c_p_fix_bugs(c_d_model, p_d_model, key_dim=c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale,two_attention_map = False,sa = False )
        self.multihead_attn_pos_content = SoftMultiHeadAttention_c_p_fix_bugs(c_d_model, p_d_model, key_dim=c_d_model+p_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout, soft_sa_scale=soft_sa_scale,two_attention_map=True,sa = False )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(c_d_model+p_d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, c_d_model+p_d_model)

        self.norm1_c = nn.LayerNorm(c_d_model)
        self.norm2_c = nn.LayerNorm(c_d_model)
        self.norm3_c = nn.LayerNorm(c_d_model)
        
        self.norm1_p = nn.LayerNorm(p_d_model)
        self.norm2_p = nn.LayerNorm(p_d_model)
        self.norm3_p = nn.LayerNorm(p_d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.use_identify_embeding_in_sa = use_identify_embeding_in_sa
        self.d_model = c_d_model
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0] 
        # tgt = tgt + self.dropout1(tgt2)
        # #print(tgt-q)
        # tgt = self.norm1(tgt)
        # #print(tgt-q)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
        #                            key=self.with_pos_embed(memory, pos),
        #                            value = memory)[0]  #key
    
        # tgt = tgt + self.dropout2(tgt2)
    
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        # tgt = tgt + self.dropout3(tgt2)
       
        # tgt = self.norm3(tgt)
        #说明不是第一层
        if tgt.shape[-1] != self.d_model:
            if  self.use_identify_embeding_in_sa:
                #q = k = torch.cat((tgt[:,:,self.d_model:],tgt[:,:,self.d_model-32:self.d_model]),dim=-1)
                q = k = tgt
                #tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
            else:
            #仅仅以位置query作为q,k
                if self.joints_pos_embed is None :
                    q = k = tgt[:,:,self.d_model:]
                ##Add pos embeding?
                else:
                    q = k = tgt[:,:,self.d_model:] + self.joints_pos_embed.unsqueeze(0)
            ##SA时是否汇聚内容信息
            if self.aggregate_context_in_sa:
                ##TODO 尝试加入可见性segms约束
                #segms = gradient_mask.unsqueeze(0).unsqueeze(0).to(q.device)
                #tgt2 = self.self_attn(q, k, value=tgt, segms = None,mask=(segms==0))[0]
                tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0]

                #tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0]
            else:
                value = tgt[:,:,self.d_model:]  #fix bugs
                ##TODO 尝试加入可见性segms约束
                tgt2 = self.self_attn(q, k, value=value, segms = None)[0]

            ##SA时是否汇聚内容信息
            if self.aggregate_context_in_sa:
                tgt = tgt + self.dropout2(tgt2)
                tgt = torch.cat((self.norm1_c(tgt[:,:,:self.d_model]),self.norm1_p(tgt[:,:,self.d_model:])),dim=-1)   #bug!
            else:
                tgt[:,:,:self.d_model] = tgt[:,:,:self.d_model] + self.dropout2(tgt2)
                tgt = torch.cat(tgt[:,:,:self.d_model],self.norm1_p(tgt[:,:,self.d_model:]),dim=-1)   #bug!




    
        #说明是第一层
        if tgt.shape[-1] == self.d_model:
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2,att_map = self.multihead_attn_content(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)  #key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm2_c(tgt_content)     
            tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)    
        #非第一层
        else:
            tgt = tgt #取content query
            tgt2,att_map = self.multihead_attn_pos_content(query=tgt,
                                    key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)  #key
            tgt = tgt + self.dropout1(tgt2)

            tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  )  
            tgt_content = tgt[...,:self.d_model]
            
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    
        tgt = tgt + self.dropout3(tgt2)
    
        tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        return tgt,att_map,tgt_content

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
    

class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim_reliability_weight(nn.Module):

    def __init__(self, c_d_model, p_d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False):
        super().__init__()
        self.c_d_model = c_d_model
        self.p_d_model = p_d_model
        self.self_attn = SoftMultiHeadAttention( query_dim=p_d_model, key_dim=p_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)

        self.multihead_attn_content = SoftMultiHeadAttention(query_dim=c_d_model, key_dim=c_d_model, num_units=c_d_model+p_d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        self.multihead_attn_pos_content = SoftMultiHeadAttention(query_dim=c_d_model+p_d_model, key_dim=c_d_model+p_d_model, num_units=c_d_model+p_d_model, hidden_dim=c_d_model+p_d_model,num_heads=nhead,dropout=dropout, soft_sa_scale=soft_sa_scale)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(c_d_model+p_d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, c_d_model+p_d_model)

        self.norm1_c = nn.LayerNorm(c_d_model)
        self.norm2_c = nn.LayerNorm(c_d_model)
        self.norm3_c = nn.LayerNorm(c_d_model)
        
        self.norm1_p = nn.LayerNorm(p_d_model)
        self.norm2_p = nn.LayerNorm(p_d_model)
        self.norm3_p = nn.LayerNorm(p_d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.d_model = c_d_model
        ##new 
        self.linear_relibility_reg = nn.Linear(c_d_model, 1)
        #self.linear_relibility_reg = nn.Linear(32, 1)

        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos,init_tgt): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0] 
        # tgt = tgt + self.dropout1(tgt2)
        # #print(tgt-q)
        # tgt = self.norm1(tgt)
        # #print(tgt-q)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
        #                            key=self.with_pos_embed(memory, pos),
        #                            value = memory)[0]  #key
    
        # tgt = tgt + self.dropout2(tgt2)
    
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        # tgt = tgt + self.dropout3(tgt2)
       
        # tgt = self.norm3(tgt)
        
        if not self.use_pos_embeding_in_ca:
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2 = self.multihead_attn_content(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)[0]  #key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm1_c(tgt_content)     
            tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)       

            #仅仅以位置query作为q,k
            q = k = tgt[:,:,self.d_model:]
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
            tgt = tgt + self.dropout1(tgt2)
        
            ###do not modified
            tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
            tgt = tgt + self.dropout3(tgt2)
        
            tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        else:
            #说明是第一层
            if tgt.shape[-1] == self.d_model:
                tgt_content = tgt[:,:,:self.d_model] #取content query
                tgt2,att_map = self.multihead_attn_content(query=tgt_content,
                                        key=memory,
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
                tgt_content = self.norm1_c(tgt_content)     
                tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)  
            #非第一层
            else:
                tgt = tgt #取content query
                tgt2,att_map = self.multihead_attn_pos_content(query=tgt,
                                        key=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                        segms = segms)  #key
                tgt = tgt + self.dropout1(tgt2)
                tgt = torch.cat((self.norm1_c(tgt[:,:,:self.d_model]),self.norm1_p(tgt[:,:,self.d_model:])),dim=-1)   #bug!

        
        ##尝试加入可见性segms约束
        #q_content = tgt[:,:,:self.d_model]
        #TODO 可以探索更多种差异表示方法，这里先尝试直接相减
        #option 1 直接相减
        #context_diff = (init_tgt - q_content)[:,:,256:]
        #context_diff = (init_tgt - q_content)
        #option 2 使用线性层回归
        # context_diff = torch.cat( [init_tgt,q_content],dim=-1 )
        # relibility = self.linear_relibility_reg(context_diff).squeeze().softmax(-1).unsqueeze(-2).repeat(1,42,1) #288->1
        #option 3 每层相对于前一层算差异
        context_diff = tgt2[:,:,:self.d_model]
        relibility = self.linear_relibility_reg(context_diff).squeeze().softmax(-1).unsqueeze(-2).repeat(1,42,1) #288->1

        #仅仅以位置query作为q,k
        q = k = tgt[:,:,self.d_model:]
        tgt2 = self.self_attn(q, k, value=tgt, segms = relibility)[0] 
        tgt = tgt + self.dropout1(tgt2)
    
        ###do not modified
        tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    
        tgt = tgt + self.dropout3(tgt2)
    
        tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        return tgt,att_map,relibility[:,0]

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos,init_tgt):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos,init_tgt=init_tgt)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos,init_tgt=init_tgt)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
    

    
class multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_old(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False):
        super().__init__()
        self.self_attn = SoftMultiHeadAttention( query_dim=d_model, key_dim=d_model, num_units=d_model*2, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        self.multihead_attn = SoftMultiHeadAttention(query_dim=d_model, key_dim=d_model, num_units=d_model*2, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model*2, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model*2)

        # self.norm1_c = nn.LayerNorm(d_model)
        # self.norm2_c = nn.LayerNorm(d_model)
        # self.norm3_c = nn.LayerNorm(d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self, tgt, memory, segms,   #query, key, value, segms,
                            pos): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0] 
        # tgt = tgt + self.dropout1(tgt2)
        # #print(tgt-q)
        # tgt = self.norm1(tgt)
        # #print(tgt-q)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
        #                            key=self.with_pos_embed(memory, pos),
        #                            value = memory)[0]  #key
    
        # tgt = tgt + self.dropout2(tgt2)
    
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        # tgt = tgt + self.dropout3(tgt2)
       
        # tgt = self.norm3(tgt)
        
        if self.use_pos_embeding_in_ca:
            pass
        else:
            #TODO 除了第一层外去q，v尝试是否加入位置编码
            tgt_content = tgt[:,:,:self.d_model] #取content query
            tgt2 = self.multihead_attn(query=tgt_content,
                                    key=memory,
                                    value=self.with_pos_embed(memory,pos,add_pos_method='concat'),
                                    segms = segms)[0]  #key
            tgt_content = tgt_content + self.dropout1(tgt2[:,:,:self.d_model])
            tgt_content = self.norm1_c(tgt_content)     
            tgt = torch.cat((tgt_content,tgt2[:,:,self.d_model:]),dim=-1)       

            #仅仅以位置query作为q,k
            q = k = tgt[:,:,self.d_model:]
            ##TODO 尝试加入可见性segms约束
            tgt2 = self.self_attn(q, k, value=tgt, segms = None)[0] 
            tgt = tgt + self.dropout1(tgt2)
        
            ###do not modified
            tgt = torch.cat(  [self.norm2_c((tgt[:,:,:self.d_model])) , self.norm2_p((tgt[:,:,self.d_model:]))], dim=-1  ) 
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
            tgt = tgt + self.dropout3(tgt2)
        
            tgt = torch.cat(  [self.norm3_c((tgt[:,:,:self.d_model])) , self.norm3_p((tgt[:,:,self.d_model:]))] , dim=-1  ) 
    
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


    def forward(self, output, memory, segms,   #query, key, value, segms,
                            pos):
        if self.normalize_before:
            return self.forward_pre(output, memory, segms,   #query, key, value, segms,
                            pos=pos)
        return self.forward_post(output, memory, segms,   #query, key, value, segms,
                            pos=pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query



class multi_TransformerDecoderLayer_Soft_Attention(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,sa_pre = True,soft_sa_method = 'multiply',
                 activation="relu", normalize_before=False,use_pos_embeding_in_ca = False):
        super().__init__()
        self.self_attn = SoftMultiHeadAttention( query_dim=d_model, key_dim=d_model, num_units=d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale,sa_pre =sa_pre, soft_sa_method = soft_sa_method)
        self.multihead_attn = SoftMultiHeadAttention(query_dim=d_model, key_dim=d_model, num_units=d_model, num_heads=nhead,dropout=dropout , soft_sa_scale=soft_sa_scale,sa_pre =sa_pre, soft_sa_method = soft_sa_method)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.use_pos_embeding_in_ca = use_pos_embeding_in_ca
        self.d_model = d_model
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)


    def with_pos_embed(self, tensor, pos,add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)


    def forward_post(self,tgt, memory, segms,   #query, key, value, segms,
                            pos,query_pos): 
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt,segms=None)[0] 
        tgt = tgt + self.dropout1(tgt2)
        #print(tgt-q)
        tgt = self.norm1(tgt)
        #print(tgt-q)
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,segms = segms)[0]  #key
    
        tgt = tgt + self.dropout2(tgt2)
    
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        tgt = tgt + self.dropout3(tgt2)
       
        tgt = self.norm3(tgt)
    
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, segms,   #query, key, value, segms,
                            pos,query_pos):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, segms,   #query, key, value, segms,
                            pos,query_pos)
        return self.forward_post(tgt, memory, segms,   #query, key, value, segms,
                            pos,query_pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

class multi_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class multi_TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor], add_pos_method = 'add'):
        if add_pos_method== 'add':
            return tensor if pos is None else tensor + pos
        elif add_pos_method== 'concat':
            return tensor if pos is None else torch.cat((tensor,pos),dim=-1)

    def forward_post(self, tgt, memory,                
                     tgt_mask: Optional[Tensor] = None,  
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None): #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0] 
        tgt = tgt + self.dropout1(tgt2)
        #print(tgt-q)
        tgt = self.norm1(tgt)
        #print(tgt-q)
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #query没有位置编码
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]  #key
    
        tgt = tgt + self.dropout2(tgt2)
    
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        tgt = tgt + self.dropout3(tgt2)
       
        tgt = self.norm3(tgt)
    
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)#output:分别为形状同quries的0向量;memory:backbone出来的特征;tgt_mask:None;memory_mask:手物query数量的mask（[4, 8, 59, 16384] -> [32, 59, 16384]）;tgt_key_padding_mask:None;memory_key_padding_mask:backbone的mask,pos:backbone特征的位置编码,query_pos:query的权重
#hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query


class all_global_transformer_two_hand(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()


        decoder_layer = multi_TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm_global = nn.LayerNorm(d_model)
        decoder_norm_decoder_left_hand = nn.LayerNorm(d_model)
        decoder_norm_decoder_right_hand = nn.LayerNorm(d_model)
        # decoder_norm = None
        num_decoder_layers = 1
        self.decoder_global = TransformerDecoderROI(decoder_layer, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        num_decoder_layers = 3
        self.decoder_local = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm_decoder_left_hand,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed,mask_left_hands,memory_mask,mask_right_hands,single_hand_num_queries, use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape #16, 256, 64, 64
        ###todo
        src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])

        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) #[59, 256] -> [59, bs,256]
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]

        if tgt is None:
            tgt = torch.zeros_like(query_embed)
        memory = src

        hs_global_layer1 = self.decoder_global(tgt, memory, memory_key_padding_mask=mask,memory_mask = memory_mask,
                          pos=pos_embed, query_pos=query_embed) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        hs_global_layer2_5 = self.decoder_local(hs_global_layer1[0], memory, memory_key_padding_mask=mask_left_hands,
                          pos=pos_embed, query_pos=query_embed)  #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

                          
        if use_all:
            hs_left_hand = torch.cat((hs_global_layer1[:,0:single_hand_num_queries],hs_global_layer2_5[:,0:single_hand_num_queries]),dim = 0)
            hs_right_hand = torch.cat((hs_global_layer1[:,single_hand_num_queries:single_hand_num_queries*2],hs_global_layer2_5[:,single_hand_num_queries:single_hand_num_queries*2]),dim=0)
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
        return hs_left_hand.transpose(1, 2),hs_right_hand.transpose(1, 2),None

class all_global_transformer_two_hand_multi_scale(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()


        decoder_layer = multi_TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm_global = nn.LayerNorm(d_model)
        decoder_norm_decoder_left_hand = nn.LayerNorm(d_model)
        decoder_norm_decoder_right_hand = nn.LayerNorm(d_model)
        # decoder_norm = None
        num_decoder_layers = 1
        self.decoder_global = TransformerDecoderROI(decoder_layer, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        num_decoder_layers = 3
        self.decoder_local = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm_decoder_left_hand,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_list, mask, query_embed, pos_embed_list,mask_left_hands_list,memory_mask_list,mask_right_hands_list,single_hand_num_queries, use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        bs, _,_,_ = src_list[0].shape #16, 256, 64, 64
        ###todo\
        src_list_flatten = []
        pos_embed_list_flatten = []
        #src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        for src_sample in src_list:
            src_sample = src_sample.flatten(2).permute(2, 0, 1) 
            src_list_flatten.append(src_sample)
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])
        for pos_embed_sample in pos_embed_list:
            pos_embed_sample = pos_embed_sample.flatten(2).permute(2, 0, 1)
            pos_embed_list_flatten.append(pos_embed_sample)

        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) #[59, 256] -> [59, bs,256]
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]
        
        mask_left_hands_list_flatten  = []
        mask_right_hands_list_flatten  = []
        for  i in range(len(mask_left_hands_list)):
            mask_left_hands = mask_left_hands_list[i].flatten(1) #[4, 128, 128] -> [4, 16384]
            mask_right_hands = mask_right_hands_list[i].flatten(1) #4,32,32->[4, 1024]
            mask_left_hands_list_flatten.append(mask_left_hands)
            mask_right_hands_list_flatten.append(mask_right_hands)
            
        if tgt is None:
            tgt = torch.zeros_like(query_embed)

        hs_global_layer1 = self.decoder_global(tgt, src_list_flatten[-1], memory_key_padding_mask=None,memory_mask = memory_mask_list[-1],
                          pos=pos_embed_list_flatten[-1], query_pos=query_embed) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        hs_global_layer2_5 = self.decoder_local(hs_global_layer1[0], src_list_flatten[:-1], memory_key_padding_mask=mask_left_hands_list_flatten[:-1],
                          pos=pos_embed_list_flatten[:-1], query_pos=query_embed)  #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

                          
        if use_all:
            hs_left_hand = torch.cat((hs_global_layer1[:,0:single_hand_num_queries],hs_global_layer2_5[:,0:single_hand_num_queries]),dim = 0)
            hs_right_hand = torch.cat((hs_global_layer1[:,single_hand_num_queries:single_hand_num_queries*2],hs_global_layer2_5[:,single_hand_num_queries:single_hand_num_queries*2]),dim=0)
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
        return hs_left_hand.transpose(1, 2),hs_right_hand.transpose(1, 2),None

class all_global_transformer_two_hand_multi_scale_sa_variant(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()


        decoder_layer_only_content_query = multi_TransformerDecoderLayer_Soft_Attention_variant(d_model, nhead, dim_feedforward,
                                                dropout, soft_sa_scale, activation, normalize_before,use_pos_embeding_in_ca= False)
        decoder_norm_global = nn.LayerNorm(d_model*2)
        # decoder_norm = None
        num_decoder_layers = 4
        self.decoder_layer_only_content_query = TransformerDecoderLayer_Soft_SA_variant(decoder_layer_only_content_query, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        # num_decoder_layers = 1

        # self.decoder_layer_both_query = TransformerDecoderLayer_Soft_SA(decoder_layer, num_decoder_layers, decoder_norm_global,
        #                                   return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                



    def get_soft_weight_for_all_query_from_pred_segms(self,pred_segms):
        all_soft_weight  = []
        right_joints_idx_to_segms = [12,[12,9],[9,7],[7,8],11,[11,1],[1,10],[10,8],4,[4,14],[14,3],[3,8],5,[5,15],[15,0],[0,8],6,[6,13],[13,2],[2,8],8]
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1 for i in segms]]
                one_part = two_part.sum(1)/2 #这里取消了.softmax(1)
            else:
                one_part = pred_segms[:,segms+1]
            all_soft_weight.append(one_part)
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1+16 for i in segms]]
                one_part = two_part.sum(1)/2#.softmax(-1)
            else:
                one_part = pred_segms[:,segms+1+16]
            all_soft_weight.append(one_part)
        return torch.stack(all_soft_weight,dim=1)
        
    def forward(self, src_list, query_embed, pos_embed_list,single_hand_num_queries, pred_segms_dict_softmax_list, use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        bs, _,_,_ = src_list[0].shape #16, 256, 64, 64
        ###todo\
        src_list_flatten = []
        pos_embed_list_flatten = []
        soft_sa_segms_list = []
        #src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        for src_sample in src_list:
            src_sample = src_sample.flatten(2).permute(0,2,1)
            src_list_flatten.append(src_sample)
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])
        for pos_embed_sample in pos_embed_list:
            pos_embed_sample = pos_embed_sample.flatten(2).permute(0,2,1)
            pos_embed_list_flatten.append(pos_embed_sample)

        for pred_segms_dict_softmax_sample in pred_segms_dict_softmax_list:
            pred_segms_dict_softmax_sample = pred_segms_dict_softmax_sample.flatten(2)
            soft_sa_segms = self.get_soft_weight_for_all_query_from_pred_segms(pred_segms_dict_softmax_sample)
            soft_sa_segms_list.append(soft_sa_segms)
        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(0).repeat(bs,1, 1) #(bs,query_num,256)
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]
        
            
        if tgt is None:
            tgt = query_embed

        hs =  self.decoder_layer_only_content_query(tgt,src_list_flatten,soft_sa_segms_list,pos_embed_list_flatten) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
                          
        if use_all:
            hs_left_hand = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,self.d_model:]
            hs_right_hand = hs[:,:,0:single_hand_num_queries,self.d_model:]
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
        return hs_left_hand,hs_right_hand, None

class all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,use_pos_embeding_in_ca= False,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,return_content_embeding = False):
        super().__init__()

        self.return_content_embeding = return_content_embeding
        decoder_layer_only_content_query = multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map(d_model, nhead, dim_feedforward,
                                                dropout, soft_sa_scale, activation, normalize_before,use_pos_embeding_in_ca= use_pos_embeding_in_ca)
        decoder_norm_global_c = nn.LayerNorm(d_model)        
        decoder_norm_global_p = nn.LayerNorm(d_model)
        decoder_norm_global = nn.ModuleList([decoder_norm_global_c,decoder_norm_global_p])
        #decoder_norm = None
        num_decoder_layers = 4
        self.decoder_layer_only_content_query = TransformerDecoderLayer_Soft_SA_variant_vis_att_map(decoder_layer_only_content_query, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        # num_decoder_layers = 1

        # self.decoder_layer_both_query = TransformerDecoderLayer_Soft_SA(decoder_layer, num_decoder_layers, decoder_norm_global,
        #                                   return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                



    def get_soft_weight_for_all_query_from_pred_segms(self,pred_segms):
        all_soft_weight  = []
        right_joints_idx_to_segms = [12,[12,9],[9,7],[7,8],11,[11,1],[1,10],[10,8],4,[4,14],[14,3],[3,8],5,[5,15],[15,0],[0,8],6,[6,13],[13,2],[2,8],8]
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1 for i in segms]]
                one_part = two_part.sum(1)/2 #这里取消了.softmax(1)
            else:
                one_part = pred_segms[:,segms+1]
            all_soft_weight.append(one_part)
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1+16 for i in segms]]
                one_part = two_part.sum(1)/2#.softmax(-1)
            else:
                one_part = pred_segms[:,segms+1+16]
            all_soft_weight.append(one_part)
        return torch.stack(all_soft_weight,dim=1)
        
    def forward(self, src_list, query_embed, pos_embed_list,single_hand_num_queries, pred_segms_dict_softmax_list, imgs_tensor = None,use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        bs, _,_,_ = src_list[0].shape #16, 256, 64, 64
        ###todo\
        src_list_flatten = []
        pos_embed_list_flatten = []
        soft_sa_segms_list = []
        #src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        for src_sample in src_list:
            src_sample = src_sample.flatten(2).permute(0,2,1)
            src_list_flatten.append(src_sample)
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])
        for pos_embed_sample in pos_embed_list:
            pos_embed_sample = pos_embed_sample.flatten(2).permute(0,2,1)
            pos_embed_list_flatten.append(pos_embed_sample)

        for pred_segms_dict_softmax_sample in pred_segms_dict_softmax_list:
            pred_segms_dict_softmax_sample = pred_segms_dict_softmax_sample.flatten(2)
            soft_sa_segms = self.get_soft_weight_for_all_query_from_pred_segms(pred_segms_dict_softmax_sample)
            soft_sa_segms_list.append(soft_sa_segms)
        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(0).repeat(bs,1, 1) #(bs,query_num,256)
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]
        
            
        if tgt is None:
            tgt = query_embed

        hs,att_map_list =  self.decoder_layer_only_content_query(tgt,src_list_flatten,soft_sa_segms_list,pos_embed_list_flatten) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        vis = False
        if vis:
            from utils.vis import visulize_attention_ratio
            import numpy as np
            import os
            import matplotlib.pyplot as plt
            from PIL import Image

            last_layer_att_map = att_map_list[-1]
            for i in range(bs):
                att_mask1 = last_layer_att_map[i][0:21].reshape(21,64,64).cpu().numpy()
                att_mask2 = last_layer_att_map[i][0:21].reshape(21,64,64).cpu().numpy()

                # att_mask2 = att_weights[i][21:22].reshape(1,128,128).cpu().numpy()
                # att_mask3 = att_weights[i][22:38].reshape(16,128,128).cpu().numpy()
                # att_mask4 = att_weights[i][38:].reshape(21,128,128).cpu().numpy()
                
                #img = imgs[i].permute(1,2,0).cpu().numpy()
                save_img_path1 = os.path.join("/data0/huanyao/code/a2-j_based-detr/output/detr_muti_scale_supervise_segms_ca_w_pos_sa_wo_weight_wo_soft_sa_wo_activate/vis_2d", "{}_att_mask_joint.png".format(0))
                save_img_path2 = os.path.join("/data0/huanyao/code/a2-j_based-detr/output/detr_muti_scale_supervise_segms_ca_w_pos_sa_wo_weight_wo_soft_sa_wo_activate/vis_2d", "{}_att_mask_joint.png".format(1))

                # save_img_path2 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_shape.png".format(img))
                # save_img_path3 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_pose.png".format(img))
                # save_img_path4 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_keypoint.png".format(img))
                # save_img_path = os.path.join("/home/zengsheng/code/Semi-Hand-Object/vis", str(k) + "test_hand_mask.png".format(img))
                #img = np.uint8(img*255.)
                # img = Image.open('/data0/huanyao/code/a2-j_based-detr/image0599.jpg', mode='r')
                # plt.imshow(img) # 显示图片
                # plt.axis('off') # 不显示坐标轴
                # plt.show()

                # save_raw_img_path = "/data0/huanyao/code/a2-j_based-detr/ttt.png"
                # plt.savefig(save_raw_img_path) 
                visulize_attention_ratio(None, att_mask1, save_img_path1, ratio=0.5)
                visulize_attention_ratio(None, att_mask2, save_img_path2, ratio=0.5)

                exit()

        vis = False
        #t-sne
        if vis:
            from sklearn import manifold, datasets
            import numpy as np
            import matplotlib.pyplot as plt
            from utils.vis import tensor_to_PIL
            bs_index = 0
            X = np.concatenate( (tgt[bs_index].cpu().numpy() ,hs[:,bs_index,:,0:256].cpu().reshape(4*42,256).numpy()),axis=0 )
            y = list(range(42))*5
            '''t-SNE'''
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(X)
            
            print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
            
            '''嵌入空间可视化'''
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            plt.figure(figsize=(8, 8))
            for i in range(X_norm.shape[0]):
                plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                        fontdict={'weight': 'bold', 'size': 9})
            plt.xticks([])
            plt.yticks([])
            plt.show()
            plt.savefig('./t-sne.png')
            img_pil = tensor_to_PIL((imgs_tensor[bs_index]*255).permute(1,2,0))
            img_pil.save("./origin.jpg")


            
                          
        if use_all:
            if not  self.return_content_embeding:
                hs_left_hand = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,self.d_model:]
                hs_right_hand = hs[:,:,0:single_hand_num_queries,self.d_model:]
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
                return hs_left_hand,hs_right_hand, None
            else:
                hs_left_hand = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,self.d_model:]
                hs_right_hand = hs[:,:,0:single_hand_num_queries,self.d_model:]
                hs_left_hand_context = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,:self.d_model]
                hs_right_hand_context= hs[:,:,0:single_hand_num_queries,:self.d_model]

            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
                return hs_left_hand,hs_right_hand, hs_left_hand_context,hs_right_hand_context
            

class all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp(nn.Module):
    def __init__(self, c_d_model=512,p_d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,use_pos_embeding_in_ca= False,use_identify_embeding_in_sa = False,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,return_content_embeding = False,joints_pos_embed = None,aggregate_context_in_sa =True,return_heatmap=False ):
        super().__init__()
        self.has_encoder_layer = False
        
        self.return_content_embeding = return_content_embeding
        self.return_heatmap = return_heatmap
        decoder_layer_only_content_query = multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim(c_d_model,p_d_model, nhead, dim_feedforward,
                                                dropout, soft_sa_scale, activation, normalize_before,use_pos_embeding_in_ca = use_pos_embeding_in_ca,
                                                use_identify_embeding_in_sa = use_identify_embeding_in_sa,joints_pos_embed = joints_pos_embed,aggregate_context_in_sa=aggregate_context_in_sa)
        decoder_norm_global_c = nn.LayerNorm(c_d_model)        
        decoder_norm_global_p = nn.LayerNorm(p_d_model)
        decoder_norm_global = nn.ModuleList([decoder_norm_global_c,decoder_norm_global_p])
        #decoder_norm = None
        num_decoder_layers = 4
        self.decoder_layer_only_content_query = TransformerDecoderLayer_Soft_SA_variant_vis_att_map_diff_dim(decoder_layer_only_content_query, num_decoder_layers, c_d_model, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        
        if self.has_encoder_layer:
            encoder_layer = TransformerEncoderLayer_Soft_SA_variant_vis_att_map_diff_dim(c_d_model, p_d_model,nhead, dim_feedforward,dropout, activation, normalize_before)
            self.encoder = TransformerEncoder_Soft_SA_variant_vis_att_map_diff_dim(encoder_layer, 2)
        # num_decoder_layers = 1

        # self.decoder_layer_both_query = TransformerDecoderLayer_Soft_SA(decoder_layer, num_decoder_layers, decoder_norm_global,
        #                                   return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = c_d_model
        self.p_d_model = p_d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                



    def get_soft_weight_for_all_query_from_pred_segms(self,pred_segms):
        all_soft_weight  = []
        right_joints_idx_to_segms = [12,[12,9],[9,7],[7,8],11,[11,1],[1,10],[10,8],4,[4,14],[14,3],[3,8],5,[5,15],[15,0],[0,8],6,[6,13],[13,2],[2,8],8]
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1 for i in segms]]
                one_part = two_part.sum(1)/2 #这里取消了.softmax(1)
            else:
                one_part = pred_segms[:,segms+1]
            all_soft_weight.append(one_part)
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1+16 for i in segms]]
                one_part = two_part.sum(1)/2#.softmax(-1)
            else:
                one_part = pred_segms[:,segms+1+16]
            all_soft_weight.append(one_part)
        return torch.stack(all_soft_weight,dim=1)
        
    def forward(self, src_list, query_embed, pos_embed_list,single_hand_num_queries, pred_segms_dict_softmax_list, imgs_tensor = None,use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        bs, _,_,_ = src_list[0].shape #16, 256, 64, 64
        ###todo\
        src_list_flatten = []
        pos_embed_list_flatten = []
        soft_sa_segms_list = []
        #src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        for src_sample in src_list:
            src_sample = src_sample.flatten(2).permute(0,2,1)
            src_list_flatten.append(src_sample)
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])
        for pos_embed_sample in pos_embed_list:
            pos_embed_sample = pos_embed_sample.flatten(2).permute(0,2,1)
            pos_embed_list_flatten.append(pos_embed_sample)

        # for pred_segms_dict_softmax_sample in pred_segms_dict_softmax_list:
        #     pred_segms_dict_softmax_sample = pred_segms_dict_softmax_sample.flatten(2)
        #     soft_sa_segms = self.get_soft_weight_for_all_query_from_pred_segms(pred_segms_dict_softmax_sample)
        #     soft_sa_segms_list.append(soft_sa_segms)
        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(0).repeat(bs,1, 1) #(bs,query_num,256)
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]
        
            
        if tgt is None:
            tgt = query_embed

        if self.has_encoder_layer:
            feature_list = self.encoder(src_list_flatten,pos_embed_list_flatten)
            src_list_flatten[-2:] = feature_list[::-1]
        hs,att_map_list,tgt_content_list =  self.decoder_layer_only_content_query(tgt,src_list_flatten,soft_sa_segms_list,pos_embed_list_flatten) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        vis = False
        if vis:
            from utils.vis import visulize_attention_ratio
            import numpy as np
            import os
            import matplotlib.pyplot as plt
            from PIL import Image

            #yangshi = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
            

            all_scale_p = {}
            all_scale_c = {}
            all_scale_all = {}

            for i in range(4):
                all_scale_c[(2**(i))*8] = att_map_list[i][0:4].mean(0)
                all_scale_p[(2**(i))*8] = att_map_list[i][5:].mean(0)
                all_scale_all[(2**(i))*8] = att_map_list[i][:].mean(0)
            for b in range(bs):
                # att_mask1 = last_layer_att_map[i][0:21].reshape(21,64,64).cpu().numpy()
                # att_mask2 = last_layer_att_map[i][0:21].reshape(21,64,64).cpu().numpy()
                image_np =  ((imgs_tensor[b]*255).permute(1,2,0)).cpu().numpy().astype(np.uint8)

                for j in range(42):
                    for scale in all_scale_p:
                        att_mask1 = all_scale_c[scale][b][j].reshape(1,scale,scale).cpu().numpy()
                        att_mask2 = all_scale_p[scale][b][j].reshape(1,scale,scale).cpu().numpy()
                        att_mask3 = all_scale_all[scale][b][j].reshape(1,scale,scale).cpu().numpy()

                # att_mask2 = att_weights[i][21:22].reshape(1,128,128).cpu().numpy()
                # att_mask3 = att_weights[i][22:38].reshape(16,128,128).cpu().numpy()
                # att_mask4 = att_weights[i][38:].reshape(21,128,128).cpu().numpy()
                
                #img = imgs[i].permute(1,2,0).cpu().numpy()
                        save_img_path1 = os.path.join("./vis", "c_{}_{}att_mask_joint.png".format(scale,j))
                        save_img_path2 = os.path.join("./vis", "p_{}_{}att_mask_joint.png".format(scale,j))
                        save_img_path3 = os.path.join("./vis", "all_{}_{}_att_mask_joint.png".format(scale,j))


                        visulize_attention_ratio(image_np, att_mask1, save_img_path1, ratio=0.5)
                        visulize_attention_ratio(image_np, att_mask2, save_img_path2, ratio=0.5)
                        visulize_attention_ratio(image_np, att_mask3, save_img_path3, ratio=0.5)
                    pass

                # save_img_path2 = os.path.join("/data0/huanyao/code/a2-j_based-detr/output/detr_muti_scale_supervise_segms_ca_w_pos_sa_wo_weight_wo_soft_sa_wo_activate/vis_2d", "{}_att_mask_joint.png".format(1))

                # save_img_path2 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_shape.png".format(img))
                # save_img_path3 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_pose.png".format(img))
                # save_img_path4 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_keypoint.png".format(img))
                # save_img_path = os.path.join("/home/zengsheng/code/Semi-Hand-Object/vis", str(k) + "test_hand_mask.png".format(img))
                #img = np.uint8(img*255.)
                # img = Image.open('/data0/huanyao/code/a2-j_based-detr/image0599.jpg', mode='r')
                # plt.imshow(img) # 显示图片
                # plt.axis('off') # 不显示坐标轴
                # plt.show()

                # save_raw_img_path = "/data0/huanyao/code/a2-j_based-detr/ttt.png"
                # plt.savefig(save_raw_img_path) 

                #gnuplot，jet


                # visulize_attention_ratio(None, att_mask2, save_img_path2, ratio=0.5)

        vis = False
        #t-sne
        if vis:
            from sklearn import manifold, datasets
            import numpy as np
            import matplotlib.pyplot as plt
            from utils.vis import tensor_to_PIL
            bs_index = 0
            X = np.concatenate( (tgt[bs_index].cpu().numpy() ,hs[:,bs_index,:,0:256].cpu().reshape(4*42,256).numpy()),axis=0 )
            y = list(range(42))*5
            '''t-SNE'''
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(X)
            
            print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
            
            '''嵌入空间可视化'''
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            plt.figure(figsize=(8, 8))
            for i in range(X_norm.shape[0]):
                plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                        fontdict={'weight': 'bold', 'size': 9})
            plt.xticks([])
            plt.yticks([])
            plt.show()
            plt.savefig('./t-sne.png')
            img_pil = tensor_to_PIL((imgs_tensor[bs_index]*255).permute(1,2,0))
            img_pil.save("./origin.jpg")


            
                          
        if use_all:
            if not  self.return_content_embeding:
                hs_left_hand = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,self.d_model:]
                hs_right_hand = hs[:,:,0:single_hand_num_queries,self.d_model:]
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
                return hs_left_hand,hs_right_hand, None
            else:
                hs_left_hand = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,self.d_model:]
                hs_right_hand = hs[:,:,0:single_hand_num_queries,self.d_model:]
                hs_left_hand_context = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,:self.d_model]
                hs_right_hand_context= hs[:,:,0:single_hand_num_queries,:self.d_model]
                hs_weak_cam_param = hs[:,:,single_hand_num_queries*2:]
                if not self.return_heatmap:

            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
                    return hs_left_hand,hs_right_hand, hs_left_hand_context,hs_right_hand_context,tgt_content_list,hs_weak_cam_param
                else:
                    return hs_left_hand,hs_right_hand, hs_left_hand_context,hs_right_hand_context,tgt_content_list,hs_weak_cam_param,att_map_list

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.01)
        if not m.bias is None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        if not m.bias is None:
            nn.init.constant_(m.bias, 0)

            
class all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp_v1(nn.Module):
    def __init__(self, c_d_model=512,p_d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,use_pos_embeding_in_ca= False,use_identify_embeding_in_sa = False,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,return_content_embeding = False):
        super().__init__()

        self.return_content_embeding = return_content_embeding
        hand_keypoint = MLP(p_d_model, 128, 2,  3)

        decoder_layer_only_content_query = multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim_fix_bugs_v1(c_d_model,p_d_model, nhead, hand_keypoint,dim_feedforward,
                                                dropout, soft_sa_scale, activation,normalize_before,use_pos_embeding_in_ca = use_pos_embeding_in_ca,
                                                use_identify_embeding_in_sa = use_identify_embeding_in_sa)
        decoder_norm_global_c = nn.LayerNorm(c_d_model)        
        decoder_norm_global_p = nn.LayerNorm(p_d_model)
        decoder_norm_global = nn.ModuleList([decoder_norm_global_c,decoder_norm_global_p])
        #decoder_norm = None
        num_decoder_layers = 4
        self.decoder_layer_only_content_query = TransformerDecoderLayer_Soft_SA_variant_vis_att_map_diff_dim_v1(decoder_layer_only_content_query, num_decoder_layers, c_d_model, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        # num_decoder_layers = 1

        # self.decoder_layer_both_query = TransformerDecoderLayer_Soft_SA(decoder_layer, num_decoder_layers, decoder_norm_global,
        #                                   return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = c_d_model
        self.p_d_model = p_d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for name,p in self.named_parameters():
            if  ("pos_inter_net" in name) or ("c2p_c_scale" in name) or ("p2c_c_scale" in name) or ("c2p_scale" in name):
                continue
            
            if "hand_keypoint" in name:
                init_weights(p)
                continue

            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                



    def get_soft_weight_for_all_query_from_pred_segms(self,pred_segms):
        all_soft_weight  = []
        right_joints_idx_to_segms = [12,[12,9],[9,7],[7,8],11,[11,1],[1,10],[10,8],4,[4,14],[14,3],[3,8],5,[5,15],[15,0],[0,8],6,[6,13],[13,2],[2,8],8]
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1 for i in segms]]
                one_part = two_part.sum(1)/2 #这里取消了.softmax(1)
            else:
                one_part = pred_segms[:,segms+1]
            all_soft_weight.append(one_part)
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1+16 for i in segms]]
                one_part = two_part.sum(1)/2#.softmax(-1)
            else:
                one_part = pred_segms[:,segms+1+16]
            all_soft_weight.append(one_part)
        return torch.stack(all_soft_weight,dim=1)
        
    def forward(self, src_list, query_embed, pos_embed_list,single_hand_num_queries, pred_segms_dict_softmax_list,imgs_tensor = None,use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        bs, _,_,_ = src_list[0].shape #16, 256, 64, 64
        ###todo\
        src_list_flatten = []
        pos_embed_list_flatten = []
        soft_sa_segms_list = []
        #src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        for src_sample in src_list:
            src_sample = src_sample.flatten(2).permute(0,2,1)
            src_list_flatten.append(src_sample)
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])
        for pos_embed_sample in pos_embed_list:
            pos_embed_sample = pos_embed_sample.flatten(2).permute(0,2,1)
            pos_embed_list_flatten.append(pos_embed_sample)

        # for pred_segms_dict_softmax_sample in pred_segms_dict_softmax_list:
        #     pred_segms_dict_softmax_sample = pred_segms_dict_softmax_sample.flatten(2)
        #     soft_sa_segms = self.get_soft_weight_for_all_query_from_pred_segms(pred_segms_dict_softmax_sample)
        #     soft_sa_segms_list.append(soft_sa_segms)
        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(0).repeat(bs,1, 1) #(bs,query_num,256)
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]
        
            
        if tgt is None:
            tgt = query_embed

        hs,att_map_list,keypoints_list,tgt_pos_list =  self.decoder_layer_only_content_query(tgt,src_list_flatten,soft_sa_segms_list,pos_embed_list_flatten) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        vis = False
        if vis:
            from utils.vis import visulize_attention_ratio
            import numpy as np
            import os
            import matplotlib.pyplot as plt
            from PIL import Image

            last_layer_att_map = att_map_list[-1]
            for i in range(bs):
                att_mask1 = last_layer_att_map[i][0:21].reshape(21,64,64).cpu().numpy()
                att_mask2 = last_layer_att_map[i][0:21].reshape(21,64,64).cpu().numpy()

                # att_mask2 = att_weights[i][21:22].reshape(1,128,128).cpu().numpy()
                # att_mask3 = att_weights[i][22:38].reshape(16,128,128).cpu().numpy()
                # att_mask4 = att_weights[i][38:].reshape(21,128,128).cpu().numpy()
                
                #img = imgs[i].permute(1,2,0).cpu().numpy()
                save_img_path1 = os.path.join("/data0/huanyao/code/a2-j_based-detr/output/detr_muti_scale_supervise_segms_ca_w_pos_sa_wo_weight_wo_soft_sa_wo_activate/vis_2d", "{}_att_mask_joint.png".format(0))
                save_img_path2 = os.path.join("/data0/huanyao/code/a2-j_based-detr/output/detr_muti_scale_supervise_segms_ca_w_pos_sa_wo_weight_wo_soft_sa_wo_activate/vis_2d", "{}_att_mask_joint.png".format(1))

                # save_img_path2 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_shape.png".format(img))
                # save_img_path3 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_pose.png".format(img))
                # save_img_path4 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_keypoint.png".format(img))
                # save_img_path = os.path.join("/home/zengsheng/code/Semi-Hand-Object/vis", str(k) + "test_hand_mask.png".format(img))
                #img = np.uint8(img*255.)
                # img = Image.open('/data0/huanyao/code/a2-j_based-detr/image0599.jpg', mode='r')
                # plt.imshow(img) # 显示图片
                # plt.axis('off') # 不显示坐标轴
                # plt.show()

                # save_raw_img_path = "/data0/huanyao/code/a2-j_based-detr/ttt.png"
                # plt.savefig(save_raw_img_path) 
                visulize_attention_ratio(None, att_mask1, save_img_path1, ratio=0.5)
                visulize_attention_ratio(None, att_mask2, save_img_path2, ratio=0.5)

                exit()

        vis = False
        #t-sne
        if vis:
            from sklearn import manifold, datasets
            import numpy as np
            import matplotlib.pyplot as plt
            from utils.vis import tensor_to_PIL
            bs_index = 0
            X = np.concatenate( (tgt[bs_index].cpu().numpy() ,hs[:,bs_index,:,0:256].cpu().reshape(4*42,256).numpy()),axis=0 )
            y = list(range(42))*5
            '''t-SNE'''
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(X)
            
            print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
            
            '''嵌入空间可视化'''
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            plt.figure(figsize=(8, 8))
            for i in range(X_norm.shape[0]):
                plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                        fontdict={'weight': 'bold', 'size': 9})
            plt.xticks([])
            plt.yticks([])
            plt.show()
            plt.savefig('./t-sne.png')
            img_pil = tensor_to_PIL((imgs_tensor[bs_index]*255).permute(1,2,0))
            img_pil.save("./origin.jpg")


            
                          
        if use_all:
            if not  self.return_content_embeding:
                hs_left_hand = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,self.d_model:]
                hs_right_hand = hs[:,:,0:single_hand_num_queries,self.d_model:]
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
                return hs_left_hand,hs_right_hand, None
            else:
                hs_left_hand = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,self.d_model:]
                hs_right_hand = hs[:,:,0:single_hand_num_queries,self.d_model:]
                hs_left_hand_context = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,:self.d_model]
                hs_right_hand_context= hs[:,:,0:single_hand_num_queries,:self.d_model]

            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
                return hs_left_hand,hs_right_hand, hs_left_hand_context,hs_right_hand_context,keypoints_list,tgt_pos_list
            
class all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp_reliability_weight(nn.Module):
    def __init__(self, c_d_model=512,p_d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,use_pos_embeding_in_ca= False,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,return_content_embeding = False):
        super().__init__()

        self.return_content_embeding = return_content_embeding
        decoder_layer_only_content_query = multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm_vis_att_map_diff_dim_reliability_weight(c_d_model,p_d_model, nhead, dim_feedforward,
                                                dropout, soft_sa_scale, activation, normalize_before,use_pos_embeding_in_ca= use_pos_embeding_in_ca)
        decoder_norm_global_c = nn.LayerNorm(c_d_model)        
        decoder_norm_global_p = nn.LayerNorm(p_d_model)
        decoder_norm_global = nn.ModuleList([decoder_norm_global_c,decoder_norm_global_p])
        #decoder_norm = None
        num_decoder_layers = 4
        self.decoder_layer_only_content_query = TransformerDecoderLayer_Soft_SA_variant_vis_att_map_diff_dim_reliability_weight(decoder_layer_only_content_query, num_decoder_layers, c_d_model, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        # num_decoder_layers = 1

        # self.decoder_layer_both_query = TransformerDecoderLayer_Soft_SA(decoder_layer, num_decoder_layers, decoder_norm_global,
        #                                   return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = c_d_model
        self.p_d_model = p_d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                



    def get_soft_weight_for_all_query_from_pred_segms(self,pred_segms):
        all_soft_weight  = []
        right_joints_idx_to_segms = [12,[12,9],[9,7],[7,8],11,[11,1],[1,10],[10,8],4,[4,14],[14,3],[3,8],5,[5,15],[15,0],[0,8],6,[6,13],[13,2],[2,8],8]
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1 for i in segms]]
                one_part = two_part.sum(1)/2 #这里取消了.softmax(1)
            else:
                one_part = pred_segms[:,segms+1]
            all_soft_weight.append(one_part)
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1+16 for i in segms]]
                one_part = two_part.sum(1)/2#.softmax(-1)
            else:
                one_part = pred_segms[:,segms+1+16]
            all_soft_weight.append(one_part)
        return torch.stack(all_soft_weight,dim=1)
        
    def forward(self, src_list, query_embed, pos_embed_list,single_hand_num_queries, pred_segms_dict_softmax_list, imgs_tensor = None,use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        bs, _,_,_ = src_list[0].shape #16, 256, 64, 64
        ###todo\
        src_list_flatten = []
        pos_embed_list_flatten = []
        soft_sa_segms_list = []
        #src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]

        for src_sample in src_list:
            src_sample = src_sample.flatten(2).permute(0,2,1)
            src_list_flatten.append(src_sample)

        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])
        for pos_embed_sample in pos_embed_list:
            pos_embed_sample = pos_embed_sample.flatten(2).permute(0,2,1)
            pos_embed_list_flatten.append(pos_embed_sample)

        #注释加速
        # for pred_segms_dict_softmax_sample in pred_segms_dict_softmax_list:
        #     pred_segms_dict_softmax_sample = pred_segms_dict_softmax_sample.flatten(2)
        #     soft_sa_segms = self.get_soft_weight_for_all_query_from_pred_segms(pred_segms_dict_softmax_sample)
        #     soft_sa_segms_list.append(soft_sa_segms)

        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(0).repeat(bs,1, 1) #(bs,query_num,256)
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]
        
            
        if tgt is None:
            tgt = query_embed

        hs,att_map_list,relibility =  self.decoder_layer_only_content_query(tgt,src_list_flatten,soft_sa_segms_list,pos_embed_list_flatten) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        vis = False
        if vis:
            from utils.vis import visulize_attention_ratio
            import numpy as np
            import os
            import matplotlib.pyplot as plt
            from PIL import Image

            last_layer_att_map = att_map_list[-1]
            for i in range(bs):
                att_mask1 = last_layer_att_map[i][0:21].reshape(21,64,64).cpu().numpy()
                att_mask2 = last_layer_att_map[i][0:21].reshape(21,64,64).cpu().numpy()

                # att_mask2 = att_weights[i][21:22].reshape(1,128,128).cpu().numpy()
                # att_mask3 = att_weights[i][22:38].reshape(16,128,128).cpu().numpy()
                # att_mask4 = att_weights[i][38:].reshape(21,128,128).cpu().numpy()
                
                #img = imgs[i].permute(1,2,0).cpu().numpy()
                save_img_path1 = os.path.join("/data0/huanyao/code/a2-j_based-detr/output/detr_muti_scale_supervise_segms_ca_w_pos_sa_wo_weight_wo_soft_sa_wo_activate/vis_2d", "{}_att_mask_joint.png".format(0))
                save_img_path2 = os.path.join("/data0/huanyao/code/a2-j_based-detr/output/detr_muti_scale_supervise_segms_ca_w_pos_sa_wo_weight_wo_soft_sa_wo_activate/vis_2d", "{}_att_mask_joint.png".format(1))

                # save_img_path2 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_shape.png".format(img))
                # save_img_path3 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_pose.png".format(img))
                # save_img_path4 = os.path.join("/data0/kzs/ICCV_2023/HO3D_vis1", id[i] + "_att_mask_keypoint.png".format(img))
                # save_img_path = os.path.join("/home/zengsheng/code/Semi-Hand-Object/vis", str(k) + "test_hand_mask.png".format(img))
                #img = np.uint8(img*255.)
                # img = Image.open('/data0/huanyao/code/a2-j_based-detr/image0599.jpg', mode='r')
                # plt.imshow(img) # 显示图片
                # plt.axis('off') # 不显示坐标轴
                # plt.show()

                # save_raw_img_path = "/data0/huanyao/code/a2-j_based-detr/ttt.png"
                # plt.savefig(save_raw_img_path) 
                visulize_attention_ratio(None, att_mask1, save_img_path1, ratio=0.5)
                visulize_attention_ratio(None, att_mask2, save_img_path2, ratio=0.5)

                exit()

        vis = False
        #t-sne
        if vis:
            from sklearn import manifold, datasets
            import numpy as np
            import matplotlib.pyplot as plt
            from utils.vis import tensor_to_PIL
            bs_index = 0
            X = np.concatenate( (tgt[bs_index].cpu().numpy() ,hs[:,bs_index,:,0:256].cpu().reshape(4*42,256).numpy()),axis=0 )
            y = list(range(42))*5
            '''t-SNE'''
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(X)
            
            print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
            
            '''嵌入空间可视化'''
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            plt.figure(figsize=(8, 8))
            for i in range(X_norm.shape[0]):
                plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                        fontdict={'weight': 'bold', 'size': 9})
            plt.xticks([])
            plt.yticks([])
            plt.show()
            plt.savefig('./t-sne.png')
            img_pil = tensor_to_PIL((imgs_tensor[bs_index]*255).permute(1,2,0))
            img_pil.save("./origin.jpg")


            
                          
        if use_all:
            if not  self.return_content_embeding:
                hs_left_hand = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,self.d_model:]
                hs_right_hand = hs[:,:,0:single_hand_num_queries,self.d_model:]
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
                return hs_left_hand,hs_right_hand, None
            else:
                hs_left_hand = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,self.d_model:]
                hs_right_hand = hs[:,:,0:single_hand_num_queries,self.d_model:]
                hs_left_hand_context = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,:self.d_model]
                hs_right_hand_context= hs[:,:,0:single_hand_num_queries,:self.d_model]

            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
                return hs_left_hand,hs_right_hand, hs_left_hand_context,hs_right_hand_context,relibility


class all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_two_feature(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,use_pos_embeding_in_ca= False,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()


        decoder_layer_only_content_query = multi_TransformerDecoderLayer_Soft_Attention_variant_layer_norm(d_model, nhead, dim_feedforward,
                                                dropout, soft_sa_scale, activation, normalize_before,use_pos_embeding_in_ca= use_pos_embeding_in_ca)
        decoder_norm_global_c = nn.LayerNorm(d_model)        
        decoder_norm_global_p = nn.LayerNorm(d_model)
        decoder_norm_global = nn.ModuleList([decoder_norm_global_c,decoder_norm_global_p])
        #decoder_norm = None
        num_decoder_layers = 4
        self.decoder_layer_only_content_query_l = TransformerDecoderLayer_Soft_SA_variant(decoder_layer_only_content_query, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        num_decoder_layers = 4
        self.decoder_layer_only_content_query_r = TransformerDecoderLayer_Soft_SA_variant(decoder_layer_only_content_query, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        # num_decoder_layers = 1

        # self.decoder_layer_both_query = TransformerDecoderLayer_Soft_SA(decoder_layer, num_decoder_layers, decoder_norm_global,
        #                                   return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                



    def get_soft_weight_for_all_query_from_pred_segms(self,pred_segms):
        all_soft_weight  = []
        right_joints_idx_to_segms = [12,[12,9],[9,7],[7,8],11,[11,1],[1,10],[10,8],4,[4,14],[14,3],[3,8],5,[5,15],[15,0],[0,8],6,[6,13],[13,2],[2,8],8]
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1 for i in segms]]
                one_part = two_part.sum(1)/2 #这里取消了.softmax(1)
            else:
                one_part = pred_segms[:,segms+1]
            all_soft_weight.append(one_part)
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1+16 for i in segms]]
                one_part = two_part.sum(1)/2#.softmax(-1)
            else:
                one_part = pred_segms[:,segms+1+16]
            all_soft_weight.append(one_part)
        return torch.stack(all_soft_weight,dim=1)
        
    def forward(self, src_list, query_embed, pos_embed_list,single_hand_num_queries, pred_segms_dict_softmax_list, use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        bs, _,_,_ = src_list[0][0].shape #16, 256, 64, 64
        ###todo\
        src_list_flatten_l = []
        src_list_flatten_r = []

        pos_embed_list_flatten = []
        soft_sa_segms_list = []
        #src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        src_list_l = src_list[0]
        src_list_r = src_list[1] #bugs

        for src_sample in src_list_l:
            src_sample = src_sample.flatten(2).permute(0,2,1)
            src_list_flatten_l.append(src_sample)
        for src_sample in src_list_r:
            src_sample = src_sample.flatten(2).permute(0,2,1)
            src_list_flatten_r.append(src_sample)

        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])
        for pos_embed_sample in pos_embed_list:
            pos_embed_sample = pos_embed_sample.flatten(2).permute(0,2,1)
            pos_embed_list_flatten.append(pos_embed_sample)

        soft_sa_segms_list_l = []
        soft_sa_segms_list_r = []

        for pred_segms_dict_softmax_sample in pred_segms_dict_softmax_list:
            pred_segms_dict_softmax_sample = pred_segms_dict_softmax_sample.flatten(2)
            soft_sa_segms = self.get_soft_weight_for_all_query_from_pred_segms(pred_segms_dict_softmax_sample)
            soft_sa_segms_list_l.append(soft_sa_segms[:,single_hand_num_queries:])
            soft_sa_segms_list_r.append(soft_sa_segms[:,:single_hand_num_queries])

        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(0).repeat(bs,1, 1) #(bs,query_num,256)
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]
        
            
        if tgt is None:
            tgt = query_embed

        hs_l =  self.decoder_layer_only_content_query_l(tgt[:,single_hand_num_queries:],src_list_flatten_l,soft_sa_segms_list_l,pos_embed_list_flatten) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        hs_r =  self.decoder_layer_only_content_query_r(tgt[:,:single_hand_num_queries],src_list_flatten_r,soft_sa_segms_list_r,pos_embed_list_flatten) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重

                          
        if use_all:
            hs_left_hand = hs_l[:,:,0:single_hand_num_queries,self.d_model:]
            hs_right_hand = hs_r[:,:,0:single_hand_num_queries,self.d_model:]
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
        return hs_left_hand,hs_right_hand, None
    
class all_global_transformer_two_hand_multi_scale_sa(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,soft_sa_scale = 100,sa_pre = True,soft_sa_method = 'multiply',
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()


        decoder_layer = multi_TransformerDecoderLayer_Soft_Attention(d_model, nhead, dim_feedforward,
                                                dropout, soft_sa_scale,sa_pre,soft_sa_method,activation, normalize_before,use_pos_embeding_in_ca= False)
        decoder_norm_global = nn.LayerNorm(d_model)
        # decoder_norm = None
        num_decoder_layers = 4
        self.decoder = TransformerDecoderLayer_Soft_SA(decoder_layer, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        # num_decoder_layers = 1

        # self.decoder_layer_both_query = TransformerDecoderLayer_Soft_SA(decoder_layer, num_decoder_layers, decoder_norm_global,
        #                                   return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                



    def get_soft_weight_for_all_query_from_pred_segms(self,pred_segms):
        all_soft_weight  = []
        right_joints_idx_to_segms = [12,[12,9],[9,7],[7,8],11,[11,1],[1,10],[10,8],4,[4,14],[14,3],[3,8],5,[5,15],[15,0],[0,8],6,[6,13],[13,2],[2,8],8]
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1 for i in segms]]
                one_part = two_part.sum(1)/2 #这里取消了.softmax(1),而是直接/2
            else:
                one_part = pred_segms[:,segms+1]
            all_soft_weight.append(one_part)
        for segms in right_joints_idx_to_segms:
            if isinstance(segms,list):
                two_part = pred_segms[:,[i+1+16 for i in segms]]
                one_part = two_part.sum(1)/2#.softmax(-1)
            else:
                one_part = pred_segms[:,segms+1+16]
            all_soft_weight.append(one_part)
        return torch.stack(all_soft_weight,dim=1)
        
    def forward(self, src_list, query_embed, pos_embed_list,single_hand_num_queries, pred_segms_dict_softmax_list, use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        bs, _,_,_ = src_list[0].shape #16, 256, 64, 64
        ###todo\
        src_list_flatten = []
        pos_embed_list_flatten = []
        soft_sa_segms_list = []
        #src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        for src_sample in src_list:
            src_sample = src_sample.flatten(2).permute(0,2,1)
            src_list_flatten.append(src_sample)
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])
        for pos_embed_sample in pos_embed_list:
            pos_embed_sample = pos_embed_sample.flatten(2).permute(0,2,1)
            pos_embed_list_flatten.append(pos_embed_sample)

        for pred_segms_dict_softmax_sample in pred_segms_dict_softmax_list:
            pred_segms_dict_softmax_sample = pred_segms_dict_softmax_sample.flatten(2)
            soft_sa_segms = self.get_soft_weight_for_all_query_from_pred_segms(pred_segms_dict_softmax_sample)
            soft_sa_segms_list.append(soft_sa_segms)
        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(0).repeat(bs,1, 1) #(bs,query_num,256)
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]
        
            
        if tgt is None: 
            tgt = torch.zeros_like(query_embed)

        hs =  self.decoder(tgt,src_list_flatten,soft_sa_segms_list,pos_embed_list_flatten,query_embed) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
                          
        if use_all:
            hs_left_hand = hs[:,:,single_hand_num_queries:single_hand_num_queries*2,:self.d_model]
            hs_right_hand = hs[:,:,0:single_hand_num_queries,:self.d_model:]
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
        #return hs_left_hand.transpose(1, 2),hs_right_hand.transpose(1, 2), None
        return hs_left_hand,hs_right_hand, None


class all_global_transformer_two_hand_add_encoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = multi_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        
        decoder_layer = multi_TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm_global = nn.LayerNorm(d_model)
        decoder_norm_decoder_left_hand = nn.LayerNorm(d_model)
        # decoder_norm = None
        num_decoder_layers = 1
        self.decoder_global = TransformerDecoderROI(decoder_layer, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        num_decoder_layers = 3
        self.decoder_local = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm_decoder_left_hand,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed,mask_left_hands,memory_mask,mask_right_hands,single_hand_num_queries, use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape #16, 256, 64, 64
        ###todo
        src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])

        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) #[59, 256] -> [59, bs,256]
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]

        if tgt is None:
            tgt = torch.zeros_like(query_embed)
        #memory = src 

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) #torch.Size([squence_length, 2, 256]) 
        
        hs_global_layer1 = self.decoder_global(tgt, memory, memory_key_padding_mask=mask,memory_mask = memory_mask,   #和hs_global_layer2_5差别在于两者使用了不同的mask
                          pos=pos_embed, query_pos=query_embed) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        hs_global_layer2_5 = self.decoder_local(hs_global_layer1[0], memory, memory_key_padding_mask=mask_left_hands,
                          pos=pos_embed, query_pos=query_embed)  #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

                          
        if use_all:
            hs_left_hand = torch.cat((hs_global_layer1[:,0:single_hand_num_queries],hs_global_layer2_5[:,0:single_hand_num_queries]),dim = 0)
            hs_right_hand = torch.cat((hs_global_layer1[:,single_hand_num_queries:single_hand_num_queries*2],hs_global_layer2_5[:,single_hand_num_queries:single_hand_num_queries*2]),dim=0)
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
        return hs_left_hand.transpose(1, 2),hs_right_hand.transpose(1, 2),None,memory
    
class all_global_transformer_two_hand_add_encoder_two_feature(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = multi_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        
        decoder_layer = multi_TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm_global = nn.LayerNorm(d_model)
        decoder_norm_decoder_left_hand = nn.LayerNorm(d_model)
        # decoder_norm = None
        num_decoder_layers = 1
        self.decoder_global_left_hand = TransformerDecoderROI(decoder_layer, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        num_decoder_layers = 3
        self.decoder_local_left_hand = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm_decoder_left_hand,
                                          return_intermediate=return_intermediate_dec)
        
        num_decoder_layers = 1
        self.decoder_global_right_hand = TransformerDecoderROI(decoder_layer, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        num_decoder_layers = 3
        self.decoder_local_right_hand = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm_decoder_left_hand,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_list, mask, query_embed, pos_embed,mask_left_hands,memory_mask,mask_right_hands,single_hand_num_queries, use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        src_left_hand = src_list[0]
        src_right_hand = src_list[1]
        
        src = torch.cat((src_left_hand,src_right_hand),dim=-2)

        bs, c, h, w = src.shape #16, 256, 64, 64
        ###todo
        src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        pos_embed = pos_embed.repeat(1,1,2,1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])

        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) #[59, 256] -> [59, bs,256]
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        mask_left_hands = mask_left_hands. repeat(1,2,1)
        mask_right_hands = mask_right_hands. repeat(1,2,1)

        mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]

        if tgt is None:
            tgt = torch.zeros_like(query_embed)
        #memory = src 

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) #torch.Size([squence_length, 2, 256]) 
        
        hs_global_layer1_left_hand = self.decoder_global_left_hand(tgt[:single_hand_num_queries], memory[:int(h/2*w)], memory_key_padding_mask=mask,memory_mask = memory_mask[:,:single_hand_num_queries],   #和hs_global_layer2_5差别在于两者使用了不同的mask
                          pos=pos_embed[:int(h/2*w)], query_pos=query_embed[:single_hand_num_queries]) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        hs_global_layer2_5_left_hand = self.decoder_local_left_hand(hs_global_layer1_left_hand[0], memory[:int(h/2*w)], memory_key_padding_mask=mask_left_hands[:,:int(h/2*w)],
                          pos=pos_embed[:int(h/2*w)], query_pos=query_embed[:single_hand_num_queries])  #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

        hs_global_layer1_right_hand = self.decoder_global_right_hand(tgt[single_hand_num_queries:], memory[int(h/2*w):], memory_key_padding_mask=mask,memory_mask = memory_mask[:,single_hand_num_queries:],   #和hs_global_layer2_5差别在于两者使用了不同的mask
                          pos=pos_embed[int(h/2*w):], query_pos=query_embed[single_hand_num_queries:]) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        hs_global_layer2_5_right_hand = self.decoder_local_right_hand(hs_global_layer1_right_hand[0], memory[int(h/2*w):], memory_key_padding_mask=mask_left_hands[:,int(h/2*w):],
                          pos=pos_embed[int(h/2*w):], query_pos=query_embed[single_hand_num_queries:])  #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query                  
        
        if use_all:
            hs_left_hand = torch.cat((hs_global_layer1_left_hand[:,0:single_hand_num_queries],hs_global_layer2_5_left_hand[:,0:single_hand_num_queries]),dim = 0)
            hs_right_hand = torch.cat((hs_global_layer1_right_hand[:,0:single_hand_num_queries],hs_global_layer2_5_right_hand[:,0:single_hand_num_queries]),dim=0)
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
        return hs_left_hand.transpose(1, 2),hs_right_hand.transpose(1, 2),None
    

class all_global_transformer_two_hand_add_encoder_two_feature_separate(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer_l = multi_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        encoder_norm_l = nn.LayerNorm(d_model) if normalize_before else None


        self.encoder_left_hand= TransformerEncoder(encoder_layer_l, num_encoder_layers, encoder_norm_l)
        

        encoder_layer_r = multi_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        encoder_norm_r = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder_right_hand= TransformerEncoder(encoder_layer_r, num_encoder_layers, encoder_norm_r)

        
        decoder_layer = multi_TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm_global = nn.LayerNorm(d_model)
        decoder_norm_decoder_left_hand = nn.LayerNorm(d_model)
        # decoder_norm = None
        num_decoder_layers = 1
        self.decoder_global_left_hand = TransformerDecoderROI(decoder_layer, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        num_decoder_layers = 3
        self.decoder_local_left_hand = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm_decoder_left_hand,
                                          return_intermediate=return_intermediate_dec)
        
        num_decoder_layers = 1
        self.decoder_global_right_hand = TransformerDecoderROI(decoder_layer, num_decoder_layers, decoder_norm_global,
                                          return_intermediate=return_intermediate_dec)
        num_decoder_layers = 3
        self.decoder_local_right_hand = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm_decoder_left_hand,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_list, mask, query_embed, pos_embed,mask_left_hands,memory_mask,mask_right_hands,single_hand_num_queries, use_all = False,tgt = None):
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32)
        # flatten NxCxHxW to HWxNxC
        src_left_hand = src_list[0]
        src_right_hand = src_list[1]
        
        #src = torch.cat((src_left_hand,src_right_hand),dim=-2)

        bs, c, h, w = src_left_hand.shape #16, 256, 64, 64
        ###todo
        #src = src.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256]  #（K_n,bs,dims）
        src_left_hand = src_left_hand.flatten(2).permute(2, 0, 1)
        src_right_hand = src_right_hand.flatten(2).permute(2, 0, 1)

        #obj = obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed_obj = pos_embed_obj.flatten(2).permute(2, 0, 1) #[4, 256, 32, 32] -> [1024, 4, 256]
        #pos_embed = pos_embed.repeat(1,1,2,1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[4, 256, 128, 128] -> [16384, 4, 256])

        if len(query_embed.shape) < 3:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) #[59, 256] -> [59, bs,256]
        elif len(query_embed.shape)  >= 4:
            query_embed = query_embed[:,:,0]

        #mask = mask.flatten(1) #[4, 128, 128] -> [4, 16384]
        # mask_left_hands = mask_left_hands. repeat(1,2,1)
        # mask_right_hands = mask_right_hands. repeat(1,2,1)

        mask_left_hands = mask_left_hands.flatten(1) #[4, 128, 128] -> [4, 16384]
        mask_right_hands = mask_right_hands.flatten(1) #4,32,32->[4, 1024]

        if tgt is None:
            tgt = torch.zeros_like(query_embed)
        #memory = src 

        memory_l = self.encoder_left_hand(src_left_hand, src_key_padding_mask=mask, pos=pos_embed) #torch.Size([squence_length, 2, 256]) 
        memory_r = self.encoder_right_hand(src_right_hand, src_key_padding_mask=mask, pos=pos_embed) #torch.Size([squence_length, 2, 256]) 

        hs_global_layer1_left_hand = self.decoder_global_left_hand(tgt[:single_hand_num_queries], memory_l, memory_key_padding_mask=mask,memory_mask = memory_mask[:,:single_hand_num_queries],   #和hs_global_layer2_5差别在于两者使用了不同的mask
                          pos=pos_embed, query_pos=query_embed[:single_hand_num_queries]) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        hs_global_layer2_5_left_hand = self.decoder_local_left_hand(hs_global_layer1_left_hand[0], memory_l, memory_key_padding_mask=mask_left_hands,
                          pos=pos_embed, query_pos=query_embed[:single_hand_num_queries])  #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query

        hs_global_layer1_right_hand = self.decoder_global_right_hand(tgt[single_hand_num_queries:], memory_r, memory_key_padding_mask=mask,memory_mask = memory_mask[:,single_hand_num_queries:],   #和hs_global_layer2_5差别在于两者使用了不同的mask
                          pos=pos_embed, query_pos=query_embed[single_hand_num_queries:]) #分别为形状同quries的0向量，backbone出来的特征，backbone特征的mask（全部为0），手物query数量的mask,backbone特征的位置编码，query的权重
        
        hs_global_layer2_5_right_hand = self.decoder_local_right_hand(hs_global_layer1_right_hand[0], memory_r, memory_key_padding_mask=mask_left_hands,
                          pos=pos_embed, query_pos=query_embed[single_hand_num_queries:])  #hs_global；memory:backnone特征的mask;memory_key_padding_mask:mask_hands;pos_embed:backnone的位置编码。最初的手的query                  
        
        if use_all:
            hs_left_hand = torch.cat((hs_global_layer1_left_hand[:,0:single_hand_num_queries],hs_global_layer2_5_left_hand[:,0:single_hand_num_queries]),dim = 0)
            hs_right_hand = torch.cat((hs_global_layer1_right_hand[:,0:single_hand_num_queries],hs_global_layer2_5_right_hand[:,0:single_hand_num_queries]),dim=0)
            #other_embeddings =  torch.cat((hs_global_layer1[:,2*single_hand_num_queries:],hs_global_layer2_5[:,2*single_hand_num_queries:]),dim = 0)
        return hs_left_hand.transpose(1, 2),hs_right_hand.transpose(1, 2),None 
    

def bulid_all_global_transformer_two_hand(args):
        return all_global_transformer_two_hand(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
    
def bulid_all_global_transformer_two_hand_multi_scale(args):
    return all_global_transformer_two_hand_multi_scale(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
    
def bulid_all_global_transformer_two_hand_multi_scale_sa(args):
    return all_global_transformer_two_hand_multi_scale_sa(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        soft_sa_scale = args.soft_sa_scale,
        sa_pre = args.sa_pre, 
        soft_sa_method = args.soft_sa_method,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

def bulid_all_global_transformer_two_hand_multi_scale_sa_variant(args):
    return all_global_transformer_two_hand_multi_scale_sa_variant(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        soft_sa_scale = args.soft_sa_scale,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
    
def bulid_all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm(args,return_content_embeding = False):
    return all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        soft_sa_scale = args.soft_sa_scale,
        use_pos_embeding_in_ca = args.use_pos_embeding_in_ca,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        return_content_embeding = return_content_embeding,
    )


def bulid_all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp(args,return_content_embeding = False,joints_pos_embed = None,return_heatmap=False):
    return all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp(
        c_d_model=256+32,
        p_d_model=256,
        dropout=args.dropout,
        soft_sa_scale = args.soft_sa_scale,
        use_pos_embeding_in_ca = args.use_pos_embeding_in_ca,
        use_identify_embeding_in_sa = args.use_identify_embeding_in_sa,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        return_content_embeding = return_content_embeding,
        return_heatmap = return_heatmap,
        joints_pos_embed = joints_pos_embed,
        aggregate_context_in_sa = args.aggregate_context_in_sa,
    )

def bulid_all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp_v1(args,return_content_embeding = False):
    return all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp_v1(
        c_d_model=256+32,
        p_d_model=288,
        dropout=args.dropout,
        soft_sa_scale = args.soft_sa_scale,
        use_pos_embeding_in_ca = args.use_pos_embeding_in_ca,
        use_identify_embeding_in_sa = args.use_identify_embeding_in_sa,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        return_content_embeding = return_content_embeding,
    )

def bulid_all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp_reliability_weight(args,return_content_embeding = False):
    return all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp_reliability_weight(
        c_d_model=256+32,
        p_d_model=256,
        dropout=args.dropout,
        soft_sa_scale = args.soft_sa_scale,
        use_pos_embeding_in_ca = args.use_pos_embeding_in_ca,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        return_content_embeding = return_content_embeding,
    )


def bulid_all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_two_feature(args):
    return all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_two_feature(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        soft_sa_scale = args.soft_sa_scale,
        use_pos_embeding_in_ca = args.use_pos_embeding_in_ca,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
    
def bulid_all_global_transformer_two_hand_add_encoder(args):
        return all_global_transformer_two_hand_add_encoder(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

def bulid_all_global_transformer_two_hand_add_encoder_two_feature(args):
        return all_global_transformer_two_hand_add_encoder_two_feature(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )



def bulid_all_global_transformer_two_hand_add_encoder_two_feature_separate(args):
        return all_global_transformer_two_hand_add_encoder_two_feature_separate(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
        
        