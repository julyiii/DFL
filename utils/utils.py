import math

import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F



def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.5):
    # """
    # Create a beta schedule that discretizes the given alpha_t_bar
    # function, which defines the cumulative product of (1-beta) over time
    # from t = [0,1].
    # :param num_diffusion_timesteps: the number of betas to produce.
    # :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
    #                   produces the cumulative product of (1-beta) up to
    #                   that part of the diffusion process.
    # :param max_beta: the maximum beta to use; use values lower than 1 to
    #                  prevent singularities.
    # """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def compute_noise_scheduling(schedule, beta_start, beta_end, num_steps):
    if schedule == "quad":
        beta = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_steps,
            )
            ** 2
        )
    elif schedule == "linear":
        beta = np.linspace(
            beta_start,
            beta_end,
            num_steps,
        )
    elif schedule == "cosine":
        beta = betas_for_alpha_bar(
            num_steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            max_beta=beta_end,
        )

    alpha_hat = 1 - beta
    alpha = np.cumprod(alpha_hat)

    sigma = ((1.0 - alpha[:-1]) / (1.0 - alpha[1:]) * beta[1:]) ** 0.5
    return beta, alpha, alpha_hat, sigma



#########################################################################conditions guidance###################################################
def mix_data_with_condition(
    noisy_data,
    data_2d,
    mix_mode="concat",
    p_uncond=0.0,
):
    # For classifier-free guidance
    data_2d = apply_guidance(data_2d, p_uncond)

    if mix_mode == "concat" or mix_mode == "z_only":
        # Concatenate over channel dim
        return torch.cat([noisy_data, data_2d], dim=1)
    elif mix_mode == "sum":
        assert torch.is_same_size(noisy_data, data_2d), (
            "noisy_data and data_2d need to have the same size to be summed."
            f"Got {noisy_data.shape} and {data_2d.shape}."
        )
        # Sum
        return noisy_data + data_2d
    else:
        raise ValueError(
            "Accepted mix_mode values are 'sum', 'concat' and 'z_only'."
            f"Got {mix_mode}."
        )

def apply_guidance(x, p_uncond, miss_rate=1.0):
    if p_uncond > 0.0:
        if np.random.rand() < p_uncond:
            if miss_rate == 1.0:
                x = torch.zeros_like(x)
            else:
                # Guidance with random joint masking instead of complete
                # masking
                B, _, J, L = x.shape
                mask = torch.zeros((B, J, L), device=x.device)
                u = np.random.uniform(0.0, 1.0, size=(B, J, L))
                mask[u > miss_rate] = 1.0
                x *= mask[:, None, ...]
    return x

class detr_condition(nn.Module):
    def __init__(self,encode_3d=True,cond_mix_mode='concat',embed_dim = 64,condition_out_dim = 64):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, condition_out_dim),
        )

        # remove 3D projection head (keep only normalization)
        self.out_dim = condition_out_dim
        self.mix_mode = cond_mix_mode

        if cond_mix_mode == "sum":
            assert encode_3d != self.keep_pt_head, (
                "When summing the condition the 3D noisy data, they need to "
                "have the same dimension. If you encode the 3D data, the "
                "PTMixSTE head should be removed and vice-versa."
            )
        else:
            if encode_3d:
                self.out_dim += condition_out_dim
            else:
                self.out_dim += 3
        # 3D encoder
        if encode_3d:
            self.data_enc = nn.Linear(in_features=3, out_features=condition_out_dim)
        else:
            self.data_enc = nn.Identity()
    def forward(self,noisy_data,data_2d,p_uncond=0.0):
        encoded_2d_data =  rearrange(self.head(data_2d).unsqueeze(-1),"B J C L ->  B C J L ")       #(B J C) -> B C J L
        encoded_3d_data = self.data_enc(
            noisy_data.permute(0, 3, 2, 1)
        ).permute(0, 3, 2, 1)
        
        total_input = mix_data_with_condition(
            encoded_3d_data,
            encoded_2d_data,
            mix_mode=self.mix_mode,
            p_uncond=p_uncond,
        )
        return total_input


def get_position_bone_embeding_from_keypoint_embeding(keypoint_refine_embeding_3d): #input bs*joints*hidden_embeding
    bone_parent = torch.Tensor([1,2,3,20,5,6,7,20,9,10,11,20,13,14,15,20,17,18,19,20]).long() #大拇指指尖到小拇指
    bone_child = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).long()
    bone_refine_embeding_3d = torch.cat((keypoint_refine_embeding_3d[:,bone_child],keypoint_refine_embeding_3d[:,bone_parent]),dim=-1) #dim=512

    return bone_refine_embeding_3d

def get_position_bone_and_vaild_label_from_keypoint(keypoints_coord,valid):#  input layer_num*bs*joints*3
    bone_parent = torch.Tensor([1,2,3,20,5,6,7,20,9,10,11,20,13,14,15,20,17,18,19,20]).long() #
    bone_parent = torch.cat((bone_parent,bone_parent+21))
    bone_child = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).long()
    bone_child = torch.cat((bone_child,bone_child+21))
    bone_vector = keypoints_coord[:,:,bone_child] - keypoints_coord[:,:,bone_parent]
    bone_valid = valid[:,:,bone_child] * valid[:,:,bone_parent]
    return bone_vector,bone_valid


def get_final_keypoint(initial_keypoint,bone_vertor_l):
    """get final 2.5D prediction use predict bone_vertor and initial 2.5D from intial lift gcn net

    Args:
        initial_keypoint (torch.tensor): layer_num*bs*init_keypoint_num*3
        bone_vertor_l (torch.tensor): layer_num*bs*bone_num*3

    Returns:
        refine_keypoint: final 2.5D prediction
    """
    layer_num, bs, _, _ = initial_keypoint.shape
    parents_keypoint = torch.Tensor([1,2,3,20,5,6,7,20,9,10,11,20,13,14,15,20,17,18,19,20]).long()
    wrist_joints = initial_keypoint[:,:,20:]
    

    #直接相加
    # initial_parents_keypoint = initial_keypoint[:,:,parents_keypoint]
    # refine_keypoint = initial_parents_keypoint + bone_vertor_l
    # refine_keypoint = torch.cat((refine_keypoint,wrist_joints),dim=-2)

    #按运动学树逐层累积
    kinematics_tree = [[3,2,1,0],[7,6,5,4],[11,10,9,8],[15,14,13,12],[19,18,17,16]]
    kinematics_wo_wrist = bone_vertor_l[:,:,kinematics_tree,:]
    kinematics = torch.cat([wrist_joints.unsqueeze(2).repeat(1,1,5,1,1),kinematics_wo_wrist],dim=3)
    kinematics_add = kinematics.cumsum(dim=3)
    kinematics_order_origin = ((kinematics_add.flip(-2))[:,:,:,:-1]).reshape(layer_num,bs,5*4,3)
    refine_keypoint = torch.cat((kinematics_order_origin,wrist_joints),dim=-2)
    return refine_keypoint

def get_context_bone_embeding_from_feature_use_initial_pred_sample_point(initial_2d_pred,features_list):
    # bone_parent_idx = torch.Tensor([1,2,3,20,5,6,7,20,9,10,11,20,13,14,15,20,17,18,19,20]).long() #大拇指指尖到小拇指
    # bone_child_idx = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).long()
    # bone_parent = initial_2d_pred[:,bone_parent_idx]
    # bone_child = initial_2d_pred[:,bone_child_idx]
    # bone_vector = bone_child - bone_parent
    # bs,bone_num,_ = bone_vector.shape
    # bone_k = bone_vector[...,1] / bone_vector[...,0]
    bs,joints_num,_  = initial_2d_pred.shape

    points_coords = 2 * (initial_2d_pred)/256 - 1 #transform initial_2d_pred 0-1 to -1,1  

    joint_feature_list = []
    for feature in features_list:
        joint_feature = F.grid_sample(feature,points_coords.unsqueeze(1),align_corners=True)
        joint_feature_list.append(joint_feature.squeeze())
    #all_joints_feature_cat = torch.cat(joint_feature_list,dim=-2).permute(0,2,1)
    joint_feature_list.reverse()
    all_joints_feature_cat = torch.stack(joint_feature_list,dim=0).permute(0,1,3,2)
    return all_joints_feature_cat  #bs*joints_num*channel

def get_context_bone_embeding_from_feature_use_initial_pred_sample_point_multi_scale(initial_2d_pred,features_list):
    # bone_parent_idx = torch.Tensor([1,2,3,20,5,6,7,20,9,10,11,20,13,14,15,20,17,18,19,20]).long() #大拇指指尖到小拇指
    # bone_child_idx = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).long()
    # bone_parent = initial_2d_pred[:,bone_parent_idx]
    # bone_child = initial_2d_pred[:,bone_child_idx]
    # bone_vector = bone_child - bone_parent
    # bs,bone_num,_ = bone_vector.shape
    # bone_k = bone_vector[...,1] / bone_vector[...,0]
    layer_num,bs,joints_num,_  = initial_2d_pred.shape
    points_coords = 2 * (initial_2d_pred)/256 - 1 #transform initial_2d_pred 0-1 to -1,1  
    
    joint_feature_list = []
    for i,feature in enumerate(features_list):
        pred_2d = points_coords[-i-1]
        joint_feature = F.grid_sample(feature,pred_2d.unsqueeze(1),align_corners=True)
        joint_feature_list.append(joint_feature.squeeze())
    all_joints_feature_cat = torch.stack(joint_feature_list,dim=0)
    return all_joints_feature_cat  #bs*joints_num*channel

###################
#define 平行线宽度
line_width = 3
###################
#define 平行线宽度
line_width = 0
def get_context_bone_embeding_from_feature_use_initial_pred(initial_2d_pred,features_list):
    bone_parent_idx = torch.Tensor([1,2,3,20,5,6,7,20,9,10,11,20,13,14,15,20,17,18,19,20]).long() #大拇指指尖到小拇指
    bone_child_idx = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).long()
    bone_parent = initial_2d_pred[:,bone_parent_idx] / 4  #/4 #fix bugs bone_parent-256 而start_x的范围以及转换为0-64了，因此应该除以4
    bone_child = initial_2d_pred[:,bone_child_idx] / 4  #/4 #fix bugs 这个时候还是bone_child的范围还是0-256 而start_x的范围以及转换为0-64了，因此应该除以4
    bone_vector = bone_child - bone_parent
    bs,bone_num,_ = bone_vector.shape
    bone_k = bone_vector[...,1] / (bone_vector[...,0]+1e-9) #TODO 在2d平面的骨骼向量，可以用来进行内容编码 

    _,feature_channel,H,_= features_list[0].shape
    feature_num = len(features_list)

    #all_bone_feature = torch.empty(bs,bone_num,feature_channel*4,device=initial_2d_pred.device) #last dim:feature dim * feature num

    #串行
    # for b in range(bs):
    #     for i in range(bone_num):
    #         bone_parent_x = bone_parent[b,i,0].int()
    #         bone_child_x = bone_child[b,i,0].int()
    #         start_x = min(bone_parent_x,bone_child_x)
    #         end_x = max(bone_parent_x,bone_child_x)
    #         if start_x == bone_parent_x:
    #             start_y = bone_parent[b,i,1].int()
    #         else:
    #             start_y = bone_child[b,i,1].int()
    #         points_x = torch.range(start=start_x, end=end_x,device = start_x.device)
    #         points_y = (points_x - start_x) * bone_k[b,i] + start_y
    #         mask_lt_zero = (torch.bitwise_and(points_x>=0 ,points_y>=0))
    #         mask_mr_256  = (torch.bitwise_and(points_x<=256,points_y<=256))
    #         mask = torch.bitwise_and(mask_lt_zero,mask_mr_256)
    #         points_x = points_x[mask]
    #         points_y = points_y[mask]
    #         b_i_all_feature_scale_list =  []  
    #         if len(points_x) == 0:
    #             all_bone_feature[b,i] = torch.zeros(256*4,device=start_x.device)
    #         else:
    #             points_coords =  2 * (torch.stack([points_x,points_y],dim=-1)/256) - 1 #transform initial_2d_pred 0-1 to -1,1  
    #             for feature in features_list:
    #                 b_i_feature = F.grid_sample(feature[b:b+1],points_coords.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(1)
    #                 #TODO 尝试不同的fusion方法 
    #                 b_i_feature_fusion = b_i_feature.sum(dim=-1)  
    #                 b_i_all_feature_scale_list.append(b_i_feature_fusion)
    #             b_i_all_feature_scale = torch.cat(b_i_all_feature_scale_list,dim=-1)
    #             all_bone_feature[b,i] = b_i_all_feature_scale

    #并行
    bone_parent_x = bone_parent[:,:,0].int()
    bone_child_x = bone_child[:,:,0].int()
    start_x,min_idx = torch.stack([bone_parent_x,bone_child_x],dim=-1).min(-1) #bs*bone_num
    end_x,max_idx = torch.stack([bone_parent_x,bone_child_x],dim=-1).max(-1)
    start_y = torch.where(min_idx==0,bone_parent[:,:,1].int(),bone_child[:,:,1].int())
    points_x = torch.range(start=0, end=H-1,device = start_x.device).view(1,1,H).repeat(bs,bone_num,1) #bs*bone_num*256
    
    #略微增加长度
    mask_in_bone = torch.bitwise_and( ( points_x >= torch.clamp(start_x.unsqueeze(-1),0,H-1) ),(points_x <=  torch.clamp(end_x.unsqueeze(-1),0,H-1) ))

    points_y = torch.where( mask_in_bone, ((points_x - start_x.unsqueeze(-1)) * bone_k.unsqueeze(-1) ) + start_y.unsqueeze(-1), torch.zeros_like(points_x) - 1 ) #将不在骨骼上的点y赋值为-1，这样采样之后的特征全0.不影响特征fusion之后的最终的结果。这里也可以不赋值为-1，因为mask也可以实现相同的效果
    
    #增加多平行线之前的
    #points_coords = 2 * (torch.stack([points_x,points_y],dim=-1)/H) - 1 #transform initial_2d_pred 0-1 to -1,1  
    #增加多平行线之后的
    ##得到宽度为2*line_width的平行线
    points_y_prallal_line = points_y.unsqueeze(-1)
    points_y_prallal_line = points_y_prallal_line + torch.range(-line_width,line_width,device = initial_2d_pred.device,dtype = initial_2d_pred.dtype).view(1,1,1,-1)
    points_y_prallal_line = torch.where(mask_in_bone.unsqueeze(-1).repeat(1,1,1,2*line_width+1),points_y_prallal_line,torch.zeros_like(points_y_prallal_line) - 1 )
    points_coords = 2 * (torch.stack([points_x.unsqueeze(-1).repeat(1,1,1,points_y_prallal_line.shape[-1]).flatten(-2),points_y_prallal_line.flatten(-2)],dim=-1)/H) - 1 #transform initial_2d_pred 0-1 to -1,1 

    
    ## mask
    ## mask_lt_zero = (torch.bitwise_and(points_x>=0 ,points_y>=0))
    ## mask_mr_256  = (torch.bitwise_and(points_x<=256,points_y<=256))

    # #增加多平行线之前的
    # mask_y =  torch.bitwise_and(points_y>0,points_y<H) #fix bugs 应该是<H
    # mask = torch.bitwise_and(mask_in_bone,mask_y) #在骨骼上且位于平面内
    # vaild_num = (mask.sum(-1, keepdim=True) + 1) #TODO 在2d平面的骨骼长度，可以用来进行内容编码
    #增加多平行线之后的
    mask_y_prallal_line =  torch.bitwise_and(points_y_prallal_line>0,points_y_prallal_line<H) #fix bugs 应该是<H
    mask = torch.bitwise_and(mask_in_bone.unsqueeze(-1),mask_y_prallal_line).flatten(-2) 
    vaild_num = (mask.sum(-1, keepdim=True) + 1) #TODO 在2d平面的骨骼长度，可以用来进行内容编码 fix bugs * points_y_prallal_line.shape[-1] 




    bone_feature_list = []
    for feature in features_list:
        bone_feature = F.grid_sample(feature,points_coords,padding_mode='border',align_corners=True)
        bone_feature_list.append(bone_feature)
    ##增加多平行线之前的
    # bone_feature = torch.stack(bone_feature_list,dim=0) * (mask.view(1,bs,1,bone_num,H)) #feature_dim*bs*c*bone_num*256
    # bone_feature = bone_feature.permute(1,0,2,3,4).reshape(bs,feature_num*feature_channel,bone_num,H)
    ##增加多平行线之后的
    bone_feature = torch.stack(bone_feature_list,dim=0) * (mask.view(1,bs,1,bone_num,H*(points_y_prallal_line.shape[-1]))) #feature_dim*bs*c*bone_num*256
    bone_feature = bone_feature.permute(1,0,2,3,4).reshape(bs,feature_num*feature_channel,bone_num,H*(points_y_prallal_line.shape[-1]))
    #TODO 尝试不同的fusion方法 
    bone_feature_fusion = bone_feature.sum(-1).permute(0,2,1)/vaild_num  #return bs*channel*keypoint_num

    #bone vector and length feature
    #bone_feature_fusion = torch.cat((bone_vector,torch.norm(bone_vector,p=2,dim=-1,keepdim=True)),dim=-1)

    return bone_feature_fusion




def restruct_joints_rel_root_from_bone(bone_coord_3d):
    layer_num, bs, _, _ = bone_coord_3d.shape
    #按运动学树逐层累积
    kinematics_tree = [[3,2,1,0],[7,6,5,4],[11,10,9,8],[15,14,13,12],[19,18,17,16]]
    kinematics_wo_wrist = bone_coord_3d[:,:,kinematics_tree,:]
    # wrist_joints = torch.zeros(layer_num,bs,5,1,3,device=bone_coord_3d.device)
    # kinematics = torch.cat([wrist_joints,kinematics_wo_wrist],dim=3)
    kinematics_add = kinematics_wo_wrist.cumsum(dim=3)
    kinematics_order_origin = ((kinematics_add.flip(-2))).reshape(layer_num,bs,5*4,3)
    wrist_joints = torch.zeros(layer_num,bs,1,3,device=bone_coord_3d.device)
    kinematics_order_origin = torch.cat((kinematics_order_origin,wrist_joints),dim=-2)
    return kinematics_order_origin

##FROM DIR
#该函数计算当点位于骨骼包围的范围时，与骨骼的垂直距离，当位于骨骼位置之外时，计算与端点的距离
def lineseg_dists(p, a, b):
    device = p.device
    d_ba = b - a
    d = torch.divide(d_ba, ( torch.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)+1e-6) ) #20*像素点*bs个单位向量 20*bs个骨骼单位向量

    s = torch.multiply(a - p, d).sum(dim=1) #20*像素点*bs个由像素指向父关节点的向量 然后与上面单位向量求点积
    t = torch.multiply(p - b, d).sum(dim=1) #20*像素点*bs个由子关节点指向像素的向量 然后与上面单位向量求点积

    h = torch.maximum(torch.maximum(s, t), torch.zeros(s.size(0)).to(device))
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0] #叉乘

    #return torch.hypot(h, c)
    return (h**2 + c**2+1e-6)**1/2


def get_each_joint_context_bone_context(keypoints_coord,bone_embeding_c,Joints_bone_value):
    B,keypoints_num,_ = keypoints_coord.shape
    bone_num = keypoints_num - 2
    bone_parent_idx_r = torch.Tensor([1,2,3,20,5,6,7,20,9,10,11,20,13,14,15,20,17,18,19,20]).long() #大拇指指尖到小拇指
    bone_parent_idx = torch.cat((bone_parent_idx_r,bone_parent_idx_r+keypoints_num//2)) #大拇指指尖到小拇指
    bone_child_idx_r = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).long()
    bone_child_idx = torch.cat((bone_child_idx_r,bone_child_idx_r+keypoints_num//2)) #大拇指指尖到小拇指
    bone_a = keypoints_coord[:,bone_parent_idx].reshape([B, 1, bone_num, 2]).repeat(1,keypoints_num, 1, 1).reshape([-1, 2])
    bone_b = keypoints_coord[:,bone_child_idx].reshape([B, 1, bone_num, 2]).repeat(1,keypoints_num, 1, 1).reshape([-1, 2])
    keypoints_coord = keypoints_coord.unsqueeze(2).repeat(1, 1,bone_num, 1).reshape(-1, 2)

    distance = lineseg_dists(keypoints_coord, bone_a, bone_b).reshape([B, keypoints_num, bone_num])

    Joints_bone_mask = (Joints_bone_value != 0)
    distance_rm_joint_link_bone = (1 / (distance.masked_fill(Joints_bone_mask,value=1e9)+1e-9)).softmax(-1).unsqueeze(-1) #fix bugs 修正为距离越大，关系越小
    bone_embeding_c = bone_embeding_c.reshape([B, 1, bone_num, -1])

    ##fusion_feature = bone_embeding_c * distance_rm_joint_link_bone + bone_embeding_c * Joints_bone_value.unsqueeze(-1) 
    fusion_feature = bone_embeding_c * Joints_bone_value.unsqueeze(-1) 

    fusion_feature =  bone_embeding_c * Joints_bone_value.unsqueeze(-1) 

    fusion_feature = fusion_feature.mean(-2) #bs*keypoint_num*288
    return fusion_feature





###################
#define 平行线宽度
line_width = 3
def get_context_bone_embeding_from_feature_use_initial_pred_two_hand_mutiscale(initial_2d_pred,features_list,pos_feature_list):
    B,keypoints_num,_ = initial_2d_pred.shape
    bone_parent_idx_r = torch.Tensor([1,2,3,20,5,6,7,20,9,10,11,20,13,14,15,20,17,18,19,20]).long() #大拇指指尖到小拇指
    bone_parent_idx = torch.cat((bone_parent_idx_r,bone_parent_idx_r+keypoints_num//2)) #大拇指指尖到小拇指
    bone_child_idx_r = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).long()
    bone_child_idx = torch.cat((bone_child_idx_r,bone_child_idx_r+keypoints_num//2)) #大拇指指尖到小拇指

    bone_parent = initial_2d_pred[:,bone_parent_idx] / 4  #/4 #fix bugs bone_parent-256 而start_x的范围以及转换为0-64了，因此应该除以4
    bone_child = initial_2d_pred[:,bone_child_idx] / 4  #/4 #fix bugs 这个时候还是bone_child的范围还是0-256 而start_x的范围以及转换为0-64了，因此应该除以4
    bone_vector = bone_child - bone_parent
    bs,bone_num,_ = bone_vector.shape
    bone_k = bone_vector[...,1] / (bone_vector[...,0]+1e-9) #TODO 在2d平面的骨骼向量，可以用来进行内容编码 

    _,feature_channel,H,_= features_list[0].shape
    feature_num = len(features_list)

    #all_bone_feature = torch.empty(bs,bone_num,feature_channel*4,device=initial_2d_pred.device) #last dim:feature dim * feature num

    #串行
    # for b in range(bs):
    #     for i in range(bone_num):
    #         bone_parent_x = bone_parent[b,i,0].int()
    #         bone_child_x = bone_child[b,i,0].int()
    #         start_x = min(bone_parent_x,bone_child_x)
    #         end_x = max(bone_parent_x,bone_child_x)
    #         if start_x == bone_parent_x:
    #             start_y = bone_parent[b,i,1].int()
    #         else:
    #             start_y = bone_child[b,i,1].int()
    #         points_x = torch.range(start=start_x, end=end_x,device = start_x.device)
    #         points_y = (points_x - start_x) * bone_k[b,i] + start_y
    #         mask_lt_zero = (torch.bitwise_and(points_x>=0 ,points_y>=0))
    #         mask_mr_256  = (torch.bitwise_and(points_x<=256,points_y<=256))
    #         mask = torch.bitwise_and(mask_lt_zero,mask_mr_256)
    #         points_x = points_x[mask]
    #         points_y = points_y[mask]
    #         b_i_all_feature_scale_list =  []  
    #         if len(points_x) == 0:
    #             all_bone_feature[b,i] = torch.zeros(256*4,device=start_x.device)
    #         else:
    #             points_coords =  2 * (torch.stack([points_x,points_y],dim=-1)/256) - 1 #transform initial_2d_pred 0-1 to -1,1  
    #             for feature in features_list:
    #                 b_i_feature = F.grid_sample(feature[b:b+1],points_coords.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(1)
    #                 #TODO 尝试不同的fusion方法 
    #                 b_i_feature_fusion = b_i_feature.sum(dim=-1)  
    #                 b_i_all_feature_scale_list.append(b_i_feature_fusion)
    #             b_i_all_feature_scale = torch.cat(b_i_all_feature_scale_list,dim=-1)
    #             all_bone_feature[b,i] = b_i_all_feature_scale

    #并行
    bone_parent_x = bone_parent[:,:,0].int()
    bone_child_x = bone_child[:,:,0].int()
    start_x,min_idx = torch.stack([bone_parent_x,bone_child_x],dim=-1).min(-1) #bs*bone_num
    end_x,max_idx = torch.stack([bone_parent_x,bone_child_x],dim=-1).max(-1)
    start_y = torch.where(min_idx==0,bone_parent[:,:,1].int(),bone_child[:,:,1].int())
    points_x = torch.range(start=0, end=H-1,device = start_x.device).view(1,1,H).repeat(bs,bone_num,1) #bs*bone_num*256

    #增加一点x可行的长度
    mask_in_bone = torch.bitwise_and((points_x >= (start_x.unsqueeze(-1)-5)),(points_x <= (end_x.unsqueeze(-1)+5)))

    points_y = torch.where( mask_in_bone, ((points_x - start_x.unsqueeze(-1)) * bone_k.unsqueeze(-1) ) + start_y.unsqueeze(-1), torch.zeros_like(points_x) - 1 ) #将不在骨骼上的点y赋值为-1，这样采样之后的特征全0.不影响特征fusion之后的最终的结果。这里也可以不赋值为-1，因为mask也可以实现相同的效果
    # mask
    # mask_lt_zero = (torch.bitwise_and(points_x>=0 ,points_y>=0))
    # mask_mr_256  = (torch.bitwise_and(points_x<=256,points_y<=256))
    ##得到宽度为2*line_width的平行线
    points_y_prallal_line = points_y.unsqueeze(-1)
    points_y_prallal_line = points_y_prallal_line + torch.range(-line_width,line_width,device = initial_2d_pred.device,dtype = initial_2d_pred.dtype).view(1,1,1,-1)
    points_y_prallal_line = torch.where(mask_in_bone.unsqueeze(-1).repeat(1,1,1,2*line_width+1),points_y_prallal_line,torch.zeros_like(points_y_prallal_line) - 1 )

    # mask_y =  torch.bitwise_and(points_y>0,points_y<H) #fix bugs 应该是<H
    # mask = torch.bitwise_and(mask_in_bone,mask_y) #

    mask_y_prallal_line =  torch.bitwise_and(points_y_prallal_line>0,points_y_prallal_line<H) #fix bugs 应该是<H
    mask = torch.bitwise_and(mask_in_bone.unsqueeze(-1),mask_y_prallal_line).flatten(-2) #

    vaild_num = (mask.sum(-1, keepdim=True).shape[-1]  + 1) #TODO 在2d平面的骨骼长度，可以用来进行内容编码  #fixbugs 这里mask已经包含平行线上的可行的素有的点，不需要再乘以平行线的数量

    #points_coords = 2 * (torch.stack([points_x,points_y],dim=-1)/H) - 1 #transform initial_2d_pred 0-1 to -1,1 
    points_coords = 2 * (torch.stack([points_x.unsqueeze(-1).repeat(1,1,1,points_y_prallal_line.shape[-1]).flatten(-2),points_y_prallal_line.flatten(-2)],dim=-1)/H) - 1 #transform initial_2d_pred 0-1 to -1,1 

    bone_feature_list = []
    bone_pos_feature_list = []
    for idx,feature in enumerate(features_list):
        bone_feature = F.grid_sample(features_list[idx],points_coords,padding_mode='border') #应该false,同插值函数保持统一
        pos_feature = F.grid_sample(pos_feature_list[idx],points_coords,padding_mode='border')
        bone_feature_list.append(bone_feature)
        bone_pos_feature_list.append(pos_feature)

    bone_feature = torch.stack(bone_feature_list,dim=0) * (mask.view(1,bs,1,bone_num,H*(points_y_prallal_line.shape[-1]))) #feature_dim*bs*c*bone_num*256
    bone_pos_feature = torch.stack(bone_pos_feature_list,dim=0) * (mask.view(1,bs,1,bone_num,H*(points_y_prallal_line.shape[-1])))
    
    #bone_feature = bone_feature.permute(1,0,2,3,4).reshape(bs,feature_num*feature_channel,bone_num,H*(points_y_prallal_line.shape[-1]))
    #TODO 尝试不同的fusion方法 
    #bone_feature_fusion = bone_feature.sum(-1).permute(0,2,1)/vaild_num  #return bs*channel*keypoint_num

    #bone vector and length feature
    #bone_feature_fusion = torch.cat((bone_vector,torch.norm(bone_vector,p=2,dim=-1,keepdim=True)),dim=-1)

    return bone_feature,bone_pos_feature,mask   #feature_layer_num*bs*c*bone_num*sample_point_num, bs*bone_num*sample_point_num

###################
#define 平行线宽度
line_width = 3
def get_context_bone_mask_from_feature_use_initial_pred_two_hand_mutiscale(initial_2d_pred):
    H = 64
    device=initial_2d_pred.device
    B,keypoints_num,_ = initial_2d_pred.shape
    bone_parent_idx_r = torch.Tensor([1,2,3,20,5,6,7,20,9,10,11,20,13,14,15,20,17,18,19,20]).long() #大拇指指尖到小拇指
    bone_parent_idx = torch.cat((bone_parent_idx_r,bone_parent_idx_r+keypoints_num//2)) #大拇指指尖到小拇指
    bone_child_idx_r = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).long()
    bone_child_idx = torch.cat((bone_child_idx_r,bone_child_idx_r+keypoints_num//2)) #大拇指指尖到小拇指

    bone_parent = initial_2d_pred[:,bone_parent_idx] / 4  #/4 #fix bugs bone_parent-256 而start_x的范围以及转换为0-64了，因此应该除以4
    bone_child = initial_2d_pred[:,bone_child_idx] / 4  #/4 #fix bugs 这个时候还是bone_child的范围还是0-256 而start_x的范围以及转换为0-64了，因此应该除以4
    bone_vector = bone_child - bone_parent
    bs,bone_num,_ = bone_vector.shape
    bone_k = bone_vector[...,1] / (bone_vector[...,0]+1e-9) #TODO 在2d平面的骨骼向量，可以用来进行内容编码 

    #all_bone_feature = torch.empty(bs,bone_num,feature_channel*4,device=initial_2d_pred.device) #last dim:feature dim * feature num

    #串行
    # for b in range(bs):
    #     for i in range(bone_num):
    #         bone_parent_x = bone_parent[b,i,0].int()
    #         bone_child_x = bone_child[b,i,0].int()
    #         start_x = min(bone_parent_x,bone_child_x)
    #         end_x = max(bone_parent_x,bone_child_x)
    #         if start_x == bone_parent_x:
    #             start_y = bone_parent[b,i,1].int()
    #         else:
    #             start_y = bone_child[b,i,1].int()
    #         points_x = torch.range(start=start_x, end=end_x,device = start_x.device)
    #         points_y = (points_x - start_x) * bone_k[b,i] + start_y
    #         mask_lt_zero = (torch.bitwise_and(points_x>=0 ,points_y>=0))
    #         mask_mr_256  = (torch.bitwise_and(points_x<=256,points_y<=256))
    #         mask = torch.bitwise_and(mask_lt_zero,mask_mr_256)
    #         points_x = points_x[mask]
    #         points_y = points_y[mask]
    #         b_i_all_feature_scale_list =  []  
    #         if len(points_x) == 0:
    #             all_bone_feature[b,i] = torch.zeros(256*4,device=start_x.device)
    #         else:
    #             points_coords =  2 * (torch.stack([points_x,points_y],dim=-1)/256) - 1 #transform initial_2d_pred 0-1 to -1,1  
    #             for feature in features_list:
    #                 b_i_feature = F.grid_sample(feature[b:b+1],points_coords.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(1)
    #                 #TODO 尝试不同的fusion方法 
    #                 b_i_feature_fusion = b_i_feature.sum(dim=-1)  
    #                 b_i_all_feature_scale_list.append(b_i_feature_fusion)
    #             b_i_all_feature_scale = torch.cat(b_i_all_feature_scale_list,dim=-1)
    #             all_bone_feature[b,i] = b_i_all_feature_scale

    #并行
    bone_parent_x = bone_parent[:,:,0].int()
    bone_child_x = bone_child[:,:,0].int()
    start_x,min_idx = torch.stack([bone_parent_x,bone_child_x],dim=-1).min(-1) #bs*bone_num
    end_x,max_idx = torch.stack([bone_parent_x,bone_child_x],dim=-1).max(-1)
    start_y = torch.where(min_idx==0,bone_parent[:,:,1].int(),bone_child[:,:,1].int())
    points_x = torch.range(start=0, end=H-1,device = device).view(1,1,H).repeat(bs,bone_num,1) #bs*bone_num*256

    #增加一点x可行的长度
    mask_in_bone = torch.bitwise_and( ( points_x >= torch.clamp(start_x.unsqueeze(-1),0,H-1) ),(points_x <=  torch.clamp(end_x.unsqueeze(-1),0,H-1) ))

    points_y = torch.where( mask_in_bone, ((points_x - start_x.unsqueeze(-1)) * bone_k.unsqueeze(-1) ) + start_y.unsqueeze(-1), torch.zeros_like(points_x) - 1 )#将不在骨骼上的点y赋值为-1，这样采样之后的特征全0.不影响特征fusion之后的最终的结果。这里也可以不赋值为-1，因为mask也可以实现相同的效果
    # mask
    # mask_lt_zero = (torch.bitwise_and(points_x>=0 ,points_y>=0))
    # mask_mr_256  = (torch.bitwise_and(points_x<=256,points_y<=256))
    ##得到宽度为2*line_width的平行线
    points_y_prallal_line = points_y.unsqueeze(-1)
    points_y_prallal_line = points_y_prallal_line + torch.range(-line_width,line_width,device = initial_2d_pred.device,dtype = initial_2d_pred.dtype).view(1,1,1,-1)
    points_y_prallal_line = torch.where(mask_in_bone.unsqueeze(-1).repeat(1,1,1,2*line_width+1),points_y_prallal_line,torch.zeros_like(points_y_prallal_line) - 1 )

    # mask_y =  torch.bitwise_and(points_y>0,points_y<H) #fix bugs 应该是<H
    # mask = torch.bitwise_and(mask_in_bone,mask_y) #

    mask_y_prallal_line =  torch.bitwise_and(points_y_prallal_line>=0,points_y_prallal_line<H) #fix bugs 应该是<H
    mask = torch.bitwise_and(mask_in_bone.unsqueeze(-1),mask_y_prallal_line).flatten(-2) #

    vaild_num = (mask.sum(-1, keepdim=True).shape[-1]  + 1) #TODO 在2d平面的骨骼长度，可以用来进行内容编码  #fixbugs 这里mask已经包含平行线上的可行的素有的点，不需要再乘以平行线的数量

    #points_coords = 2 * (torch.stack([points_x,points_y],dim=-1)/H) - 1 #transform initial_2d_pred 0-1 to -1,1 
    bone_mask = (torch.full((B,bone_num,H,H),False,device=device))
    points_coords = torch.stack([points_x.unsqueeze(-1).repeat(1,1,1,points_y_prallal_line.shape[-1]).flatten(-2),points_y_prallal_line.flatten(-2)],dim=-1)#transform initial_2d_pred 0-1 to -1,1 
    valid_coords = (mask.unsqueeze(-1) * points_coords).long() #x y 落在图像上且在骨骼上的为索引值，其余为0
    bs_index,bone_num_index = torch.meshgrid(torch.range(0,bs-1),torch.range(0,bone_num-1))
    bs_index = bs_index.view(bs,bone_num,1).expand(bs,bone_num,H*(2*line_width+1)).long().to(device)
    bone_num_index = bone_num_index.view(bs,bone_num,1).expand(bs,bone_num,H*(2*line_width+1)).long().to(device)

    bone_mask.index_put_((bs_index.flatten(),bone_num_index.flatten(),valid_coords[:,:,:,1].flatten(),valid_coords[:,:,:,0].flatten()),torch.tensor(True,device=device)) #注意此处下图像下的下x,y与tensor索引的顺序正好相反。而grid sample的顺序同像素坐标系。       
    bone_mask[:,:,0,:] = bone_mask[:,:,0,:] * mask[:,:,::(2*line_width+1)]


    return bone_mask   #feature_layer_num*bs*c*bone_num*sample_point_num, bs*bone_num*sample_point_num

def keypoints_mask_from_bone_info(mask):
    bs,bone_num,H,W =  mask.shape
    #仅仅和关节点相邻的手指
    # keypoints_coord_rel_bone_r = torch.Tensor([[0,0],[0,1],[1,2],[2,3],[4,4],[4,5],[5,6],[6,7],[8,8],[8,9],[9,10],[10,11],[12,12],[12,13],[13,14],[14,15],[16,16],[16,17],[17,18],[18,19],[3,19]]).long() #大拇指指尖到小拇指
    # keypoints_coord_rel_bone = torch.cat((keypoints_coord_rel_bone_r,keypoints_coord_rel_bone_r+bone_num//2)) #大拇指指尖到小拇指

    #整根手指
    keypoints_coord_rel_bone_r = torch.Tensor([[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],
                                               [4,5,6,7],[4,5,6,7],[4,5,6,7],[4,5,6,7],
                                               [8,9,10,11],[8,9,10,11],[8,9,10,11],[8,9,10,11],
                                               [12,13,14,15],[12,13,14,15],[12,13,14,15],[12,13,14,15],
                                               [16,17,18,19],[16,17,18,19],[16,17,18,19],[16,17,18,19],
                                               [3,7,15,19]]).long() #大拇指指尖到小拇指
    keypoints_coord_rel_bone = torch.cat((keypoints_coord_rel_bone_r,keypoints_coord_rel_bone_r+bone_num//2)) #大拇指指尖到小拇指
    
    j_mask =  mask[:,keypoints_coord_rel_bone]
    j_mask = torch.bitwise_or(torch.bitwise_or(j_mask[:,:,0],j_mask[:,:,1]),torch.bitwise_or(j_mask[:,:,2],j_mask[:,:,3]))

    # add_mask_for_root_r = torch.Tensor([[7,11,15]]).long()
    # add_mask_for_root = torch.cat((add_mask_for_root_r,add_mask_for_root_r+bone_num//2))

    # add_mask = mask[:,add_mask_for_root]
    # add_mask = torch.bitwise_or(torch.bitwise_or(add_mask[:,:,0],add_mask[:,:,1]),add_mask[:,:,2])
    
    # j_mask[:,20] = torch.bitwise_or(j_mask[:,20],add_mask[:,0])
    # j_mask[:,41] = torch.bitwise_or(j_mask[:,41],add_mask[:,1])

    return j_mask

def restore_weak_cam_param(joints_gt,joint_loc_2d_gt,joint_valid_in):
    """get gt weak cam param

        joints_gt (torch.tensor): bs*num_keypoints*3 mm
        joint_loc_2d_gt (torch.tensor): bs*num_keypoint*2 pixel
        joint_valid (torch.tensor): N*num_keypoint

    Returns:
        gt_cam_param: 1(layer_num)*N*3 
        cam_valid:1(layer_num) x N x 1(unsqueeze for three dim cam param)
    """
    bs,keypoint_num = joints_gt.shape[0],joints_gt.shape[1]

    col1 = torch.zeros((keypoint_num*2,), dtype=torch.float32).to(joints_gt.device)
    col1[0::2] = 1
    col1 = col1.unsqueeze(0).repeat(bs,1)
    col2 = torch.zeros((keypoint_num * 2,), dtype=torch.float32).to(joints_gt.device)
    col2[1::2] = 1
    col2 = col2.unsqueeze(0).repeat(bs, 1)
    joint_valid = joint_valid_in.detach().clone()
    
    # Compute the orthographic projection matrix
    with torch.no_grad():
        b = joint_loc_2d_gt.reshape(bs, -1)  # N x 21*2
        b[:, ::2] *= joint_valid
        b[:, 1::2] *= joint_valid
        cam_param_gt_list = []
        for i in range(1):
            mat_A = torch.stack([joints_gt[:,:,:2].reshape(bs, -1), col1, col2], dim=2) # N x 21*2 x 3

            mat_A[:, ::2] *= joint_valid.unsqueeze(2)
            mat_A[:, 1::2] *= joint_valid.unsqueeze(2)

            tmp = torch.inverse(torch.matmul(mat_A.transpose(2,1), mat_A)+torch.eye(3, device=mat_A.device, dtype=mat_A.dtype)*1e-4)
            cam_param_gt_curr = torch.matmul(tmp, torch.matmul(mat_A.transpose(2,1), b.unsqueeze(2))).squeeze(2).unsqueeze(0) # 1 x N x 3
            cam_param_gt_list.append(cam_param_gt_curr)

        cam_param_gt = torch.cat(cam_param_gt_list, dim=0) # 6 x N x 3
        cam_valid = torch.sum(joint_valid, dim=1) > 0 # N
        cam_valid = cam_valid.unsqueeze(0).unsqueeze(2) # 1 x N x 1
        
        
        
    return cam_param_gt, cam_valid