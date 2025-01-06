import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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
 
    def __init__(self, query_dim, key_dim, num_units, num_heads, hidden_dim = None,soft_sa_scale = 1, dropout=0.1,sa_pre = True,soft_sa_method = 'multiply'):
 
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
            self.W_value = nn.Linear(in_features=num_units, out_features=hidden_dim, bias= True) 
            self.hidden_dim = hidden_dim

        
        self.out_proj = nn.Linear(in_features=num_units, out_features=query_dim, bias= True)
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

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0) == 0, -1e9)
        
        if self.sa_pre:
            ## mask
            if segms is not None:
                ## mask:  [N, T_k] --> [h, N, T_q, T_k]
                ## new segms [N,T_q,T_K] --> [h, N, T_q, T_k]
                segms = segms.unsqueeze(0).repeat(self.num_heads,1,1,1)
                #scores = scores.masked_fill(segms, -np.inf)
                scores.masked_fill(segms,-1e9)

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
 
        return out,scores.mean(0)

class adaption_layer(nn.Module):
    def __init__(self,query_dim,key_dim,v_dim,dropout = 0.1):
        super().__init__()
        #CA
        self.attention_layer  = SoftMultiHeadAttention(query_dim, key_dim, v_dim, 8, hidden_dim = key_dim)
        #SA
        self.attention_layer_sa  = SoftMultiHeadAttention(query_dim, query_dim, query_dim, 8, hidden_dim = query_dim)

        self.q_layerNorm = nn.LayerNorm(query_dim)
        self.k_v_layerNorm = nn.LayerNorm(key_dim)
        self.q1_layer_norm = nn.LayerNorm(query_dim)
        self.linear = nn.Linear(query_dim, query_dim)
        self.linear2 = nn.Linear(query_dim, query_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) 
        self.dropout2 = nn.Dropout(dropout) 
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.linear.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)


    def forward(self,q,kv,mask=None):
        q = self.q_layerNorm(q)
        k = v =   self.k_v_layerNorm(kv)
        if mask is None:
            q_1 = self.attention_layer(q,k,v,segms = None)[0]
        else:
            q_1 = self.attention_layer(q,k,v,segms = None,mask = mask)[0]

        q = self.q1_layer_norm(q + q_1)
        q_2 = self.linear2(self.dropout(self.activation(self.linear(q))))
        q = q + self.dropout2(self.linear(q_2))

        return q


        #CA
        # q = self.q_layerNorm(q)
        # k = v =   self.k_v_layerNorm(kv)
        # if mask is None:
        #     q_1 = self.attention_layer(q,k,v,segms = None)[0]
        # else:
        #     q_1 = self.attention_layer(q,k,v,segms = None,mask = mask)[0]
        # q = self.q1_layer_norm(q + q_1)
        # #SA
        # q = self.attention_layer_sa(q,q,q,segms = None)[0]

        # q_2 = self.linear2(self.dropout(self.activation(self.linear(q))))
        # q = q + self.dropout2(self.linear(q_2))

        return q





class crossmodal_Transformer(nn.Module):
    def __init__(self,query_dim,key_dim,v_dim):
        super().__init__()
        crossmodal_Transformer_layer = adaption_layer(query_dim,key_dim,v_dim)
        self.crossmodal_Transformer = _get_clones(crossmodal_Transformer_layer,4)
        self.layer_norm = nn.LayerNorm(query_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,all_joints_pos_embeding,features_list,pos_feature_list,keypoint_mask=None):
        H = 64
        output = all_joints_pos_embeding
        for i,layer in enumerate(self.crossmodal_Transformer):
            feature = features_list[-i-1].flatten(2).permute(0,2,1)
            bs,pixel_num,_ = feature.shape
            H_ds = int(pixel_num**0.5)
            downsample_radio = H_ds / H
            pos = pos_feature_list[-i-1].flatten(2).permute(0,2,1)
            feature[:,:,:256] += pos               #fix bugs!
            if keypoint_mask is None:
                output = layer(output, feature) 
            else:
                ds_mask = F.interpolate(keypoint_mask.float(), scale_factor = downsample_radio , mode="nearest").bool().flatten(-2)
                output = layer(output, feature,ds_mask) 

        return self.layer_norm(output)

        

class  vis_j_inter_w_whole_map(nn.Module):
    def __init__(self,query_dim,key_dim,v_dim,dropout = 0.1):
        super().__init__()
        self.query_dim = query_dim
        soft_attention_layer = SoftMultiHeadAttention(query_dim,key_dim,v_dim,num_heads=8)
        self.soft_attention_ca = _get_clones(soft_attention_layer,4)
        self.layer_norm = nn.LayerNorm(query_dim)
        self.layer_norm_list = _get_clones(self.layer_norm,4)
        self.layer_norm_list1 = _get_clones(self.layer_norm,4)
        # self.dropout = nn.Dropout(dropout) 
        # self.dropout_list = _get_clones(self.dropout,4)
        self.linear = nn.Linear(query_dim, query_dim)
        self._reset_parameters()

        self.linear_list = _get_clones(self.linear,4)
        self.activation = nn.ReLU()






    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.linear.bias, 0.)

    def forward(self,hs_hand_j_feature,features_list_with_vis_info_hidden_dim_256,segms = None):
        j_inter_w_whole_map = []
        for i,layer in enumerate(features_list_with_vis_info_hidden_dim_256):
            #segms 
            # hs_hand_j_feature_l = self.layer_norm_list[i](hs_hand_j_feature[i][:,:self.query_dim].permute(0,2,1))
            # features_list_with_vis_info_hidden_dim_256 = features_list_with_vis_info_hidden_dim_256[i]
            # features_list_with_vis_info__hidden_dim_256_l =  features_list_with_vis_info_hidden_dim_256[:,:self.query_dim].flatten(-2).permute(0,2,1)
            # if segms is not None:
            #     segms = segms.unsqueeze(1).repeat(1,42,1,1).flatten(-2)
            # hs_hand_j_feature_avg = (self.soft_attention_ca[i])(hs_hand_j_feature_l,features_list_with_vis_info__hidden_dim_256_l,features_list_with_vis_info__hidden_dim_256_l,segms=segms)[0]
            # res = self.activation(self.linear_list[i](self.layer_norm_list1[i](hs_hand_j_feature_avg+hs_hand_j_feature_l)))
            # j_inter_w_whole_map.append(res)

            hs_hand_j_feature_l = self.layer_norm_list[i](hs_hand_j_feature[i].permute(0,2,1))
            features_list_with_vis_info__hidden_dim_256_l = features_list_with_vis_info_hidden_dim_256[i].flatten(-2).permute(0,2,1)
            hs_hand_j_feature_avg = (self.soft_attention_ca[i])(hs_hand_j_feature_l,features_list_with_vis_info__hidden_dim_256_l,features_list_with_vis_info__hidden_dim_256_l,segms=None)[0]
            res = self.activation(self.linear_list[i](self.layer_norm_list1[i](hs_hand_j_feature_avg+hs_hand_j_feature_l)))
            j_inter_w_whole_map.append(res)

        return j_inter_w_whole_map
    

class  j_feature_inter_selfsa(nn.Module):
    def __init__(self,query_dim,key_dim,v_dim):
        super().__init__()
        soft_attention_layer = SoftMultiHeadAttention(query_dim,key_dim,v_dim,num_heads=8)
        self.soft_attention_ca = _get_clones(soft_attention_layer,1)
        self.layer_norm = nn.LayerNorm(query_dim)
        self.layer_norm_list = _get_clones(self.layer_norm,1)
        self.layer_norm_list1 = _get_clones(self.layer_norm,1)

        # self.dropout = nn.Dropout(dropout) 
        # self.dropout_list = _get_clones(self.dropout,4)
        self.linear = nn.Linear(query_dim, query_dim)
        self._reset_parameters()

        self.linear_list = _get_clones(self.linear,4)
        self.activation = nn.ReLU()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.linear.bias, 0.)


    def forward(self,hs_hand_j_feature):
        j_inter_sa = []
        for i,layer in enumerate(hs_hand_j_feature):
            q = k = v = self.layer_norm_list[i](hs_hand_j_feature[i])
            attention_softmax = (self.soft_attention_ca[i])(q,k,v,segms=None)[1]
            j_inter_sa.append(attention_softmax)
        j_inter_sa = torch.stack(j_inter_sa,dim=1)
        return j_inter_sa
    