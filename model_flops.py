import torch
import torch.nn as nn
# from nets.meshreg import MeshRegNet
# from nets.module import BackboneNet, PoseNet
from config import cfg
# from nets.loss import calculate_loss
import torchvision.transforms as transforms
from nets.backbone import build_backbone
from nets.transformer import bulid_all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp
# import smplx
import copy
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from utils.utils import compute_noise_scheduling,apply_guidance,detr_condition,get_context_bone_embeding_from_feature_use_initial_pred_sample_point,get_context_bone_embeding_from_feature_use_initial_pred,get_each_joint_context_bone_context,get_context_bone_mask_from_feature_use_initial_pred_two_hand_mutiscale,keypoints_mask_from_bone_info,get_context_bone_embeding_from_feature_use_initial_pred_sample_point_multi_scale,restore_weak_cam_param

# from utils.misc import rot6d2mat, mat2aa
import torch.nn.functional as F
import numpy as np
import os.path as osp
from nets.lifting_ct_gcn import LiftCTGCN
from nets.mix_ste import DiffMixSTE
from utils.skeleton import Skeleton
import nets.segm_net as segm_net
import segmentation_models_pytorch as smp
from einops import rearrange,parse_shape
from nets.model_IGANet import Model as IGANet
from nets.model_IGANet import IGANet_attention_guide
from nets.pose_dformer import PoseTransformer
import math
from nets.crossmodal_Transformer import crossmodal_Transformer,vis_j_inter_w_whole_map,j_feature_inter_selfsa
import gc


from utils.vis import visulize_attention_ratio,compare_visulize_attention_ratio

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


def _get_clones(module, N):
    #init_weights(module)
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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

def init_weights_like_dir(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=0.001)           
            
def detr_loss(total_keypoints_coord,annotations,joint_valid,spatialFactor=0.5):
    lvl_num, batch_size, j, _ = total_keypoints_coord.shape
    total_annotations = torch.unsqueeze(annotations,0).expand(lvl_num,-1,-1,-1)  ## l,b,j,3 
    total_joint_valid = joint_valid[None,:,:,None].expand(lvl_num,-1,-1, 3)  ## l,b,j,3 
    
    
    # Count the number of valid and set zero items to one
    per_batch_valid_num = torch.sum(joint_valid, dim = 1)
    per_batch_valid_num = torch.where(
                torch.le(per_batch_valid_num, 0.5),
                per_batch_valid_num+1,
                per_batch_valid_num
        )
    per_batch_valid_num = torch.unsqueeze(per_batch_valid_num,0).expand(lvl_num,-1) * 3  ## l,b ,*3 indicates how many 3d coordinates are available in l and 
    
    regression_diff_xy = torch.abs(total_keypoints_coord[:,:,:,0:2] - total_annotations[:,:,:,0:2])
    regression_loss_xy = torch.where(
                torch.le(regression_diff_xy, 1),
                0.5 * 1 * torch.pow(regression_diff_xy, 2),
                regression_diff_xy - 0.5 / 1
                )
    regression_diff_z = torch.abs(total_keypoints_coord[:,:,:,2] - total_annotations[:,:,:,2])
    regression_loss_z = torch.where(
                torch.le(regression_diff_z, 3),
                0.5 * (1/3) * torch.pow(regression_diff_z, 2),
                regression_diff_z - 0.5 / (1/3)
                )
    regression_loss = torch.cat((regression_loss_xy * spatialFactor, regression_loss_z.unsqueeze(3)), dim = 3)
    regression_loss_valid = (regression_loss * total_joint_valid).reshape(lvl_num, batch_size, -1)  ## l,b,j*3
    regression_loss_mean_batch = torch.mean(torch.sum(regression_loss_valid, 2)/ per_batch_valid_num, 1)  ## l,b 
    regression_loss_output = torch.mean(regression_loss_mean_batch)

    return regression_loss_output


def detr_loss_2d(total_keypoints_coord,annotations,joint_valid,spatialFactor=0.5):
    lvl_num, batch_size, j, _ = total_keypoints_coord.shape
    total_annotations = torch.unsqueeze(annotations,0).expand(lvl_num,-1,-1,-1)  ## l,b,j,3 
    total_joint_valid = joint_valid[None,:,:,None].expand(lvl_num,-1,-1, 2)  ## l,b,j,3 
    
    
    # Count the number of valid and set zero items to one
    per_batch_valid_num = torch.sum(joint_valid, dim = 1)
    per_batch_valid_num = torch.where(
                torch.le(per_batch_valid_num, 0.5),
                per_batch_valid_num+1,
                per_batch_valid_num
        )
    per_batch_valid_num = torch.unsqueeze(per_batch_valid_num,0).expand(lvl_num,-1) * 3  ## l,b ,*3 indicates how many 3d coordinates are available in l and 
    
    regression_diff_xy = torch.abs(total_keypoints_coord[:,:,:,0:2] - total_annotations[:,:,:,0:2])
    regression_loss_xy = torch.where(
                torch.le(regression_diff_xy, 1),
                0.5 * 1 * torch.pow(regression_diff_xy, 2),
                regression_diff_xy - 0.5 / 1
                )
    # regression_diff_z = torch.abs(total_keypoints_coord[:,:,:,2] - total_annotations[:,:,:,2])
    # regression_loss_z = torch.where(
    #             torch.le(regression_diff_z, 3),
    #             0.5 * (1/3) * torch.pow(regression_diff_z, 2),
    #             regression_diff_z - 0.5 / (1/3)
    #             )
    #regression_loss = torch.cat((regression_loss_xy * spatialFactor, regression_loss_z.unsqueeze(3)), dim = 3)
    regression_loss = regression_loss_xy
    regression_loss_valid = (regression_loss * total_joint_valid).reshape(lvl_num, batch_size, -1)  ## l,b,j*3
    regression_loss_mean_batch = torch.mean(torch.sum(regression_loss_valid, 2)/ per_batch_valid_num, 1)  ## l,b 
    regression_loss_output = torch.mean(regression_loss_mean_batch)

    return regression_loss_output


def detr_loss_depth(total_keypoints_coord,annotations,joint_valid,spatialFactor=0.5):
    lvl_num, batch_size, j, _ = total_keypoints_coord.shape
    total_annotations = torch.unsqueeze(annotations,0).expand(lvl_num,-1,-1,-1)  ## l,b,j,3 
    total_joint_valid = joint_valid[None,:,:,None].expand(lvl_num,-1,-1, 1)  ## l,b,j,3 
    
    
    # Count the number of valid and set zero items to one
    per_batch_valid_num = torch.sum(joint_valid, dim = 1)
    per_batch_valid_num = torch.where(
                torch.le(per_batch_valid_num, 0.5),
                per_batch_valid_num+1,
                per_batch_valid_num
        )
    per_batch_valid_num = torch.unsqueeze(per_batch_valid_num,0).expand(lvl_num,-1) * 1  ## l,b ,*3 indicates how many 3d coordinates are available in l and 
    
    # regression_diff_xy = torch.abs(total_keypoints_coord[:,:,:,0:2] - total_annotations[:,:,:,0:2])
    # regression_loss_xy = torch.where(
    #             torch.le(regression_diff_xy, 1),
    #             0.5 * 1 * torch.pow(regression_diff_xy, 2),
    #             regression_diff_xy - 0.5 / 1
    #             )
    regression_diff_z = torch.abs(total_keypoints_coord[:,:,:,0:] - total_annotations[:,:,:,2:])
    regression_loss_z = torch.where(
                torch.le(regression_diff_z, 3),
                0.5 * (1/3) * torch.pow(regression_diff_z, 2),
                regression_diff_z - 0.5 / (1/3)
                )
    regression_loss = regression_loss_z
    regression_loss_valid = (regression_loss * total_joint_valid).reshape(lvl_num, batch_size, -1)  ## l,b,j*3
    regression_loss_mean_batch = torch.mean(torch.sum(regression_loss_valid, 2)/ per_batch_valid_num, 1)  ## l,b 
    regression_loss_output = torch.mean(regression_loss_mean_batch)
    return regression_loss_output

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.2):
        super().__init__()
        #self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j,joint_valid):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        l,batch_size,keypoint_num,hidden_dim = emb_i.shape
        joints_mask = joint_valid.view(1,1,batch_size,keypoint_num)

        all_num = joint_valid.sum()
        emb_i = emb_i.reshape(l,batch_size*keypoint_num,hidden_dim)
        emb_j = emb_j.reshape(l,batch_size*keypoint_num,hidden_dim)
        z_i = F.normalize(emb_i, dim=2)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=2)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=1)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(2), representations.unsqueeze(1), dim=-1)      # simi_mat: (2*bs, 2*bs)
        
        # sim_ij = torch.diag(similarity_matrix, batch_size*keypoint_num)         # bs
        # sim_ji = torch.diag(similarity_matrix, - batch_size*keypoint_num)       # bs
        sim_ij_list = []
        sim_ji_list = [] 
        for layer in range(l):
            similarity_matrix_layer = similarity_matrix[layer]
            sim_ij_list.append(torch.diag(similarity_matrix_layer,batch_size*keypoint_num))
            sim_ji_list.append(torch.diag(similarity_matrix_layer,batch_size*keypoint_num))
        sim_ij = torch.stack(sim_ij_list,dim=0)    
        sim_ji = torch.stack(sim_ji_list,dim=0)    
        positives = torch.cat([sim_ij, sim_ji], dim=1)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask.unsqueeze(0) * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=2))        # 2*bs
        loss = torch.sum(loss_partial.reshape(l,2,batch_size,keypoint_num)*joints_mask) / (2 * all_num)
        return loss

def detr_context_loss(total_keypoints_coord,annotations,joint_valid):
    lvl_num, batch_size, j, _ = total_keypoints_coord.shape
    total_annotations = annotations  ## l,b,j,3 
    total_joint_valid = joint_valid[None,:,:,None].expand(lvl_num,-1,-1, 288)  ## l,b,j,3 
    
    
    # Count the number of valid and set zero items to one
    per_batch_valid_num = torch.sum(joint_valid, dim = 1)
    per_batch_valid_num = torch.where(
                torch.le(per_batch_valid_num, 0.5),
                per_batch_valid_num+1,
                per_batch_valid_num
        )
    per_batch_valid_num = torch.unsqueeze(per_batch_valid_num,0).expand(lvl_num,-1) * 256  ## l,b ,*3 indicates how many 3d coordinates are available in l and 
    
    regression_diff_xy = torch.abs(total_keypoints_coord[:,:,:,:] - total_annotations[:,:,:,:])**2

    regression_loss = regression_diff_xy

    regression_loss_valid = (regression_loss * total_joint_valid).reshape(lvl_num, batch_size, -1)  ## l,b,j*3
    regression_loss_mean_batch = torch.mean(torch.sum(regression_loss_valid, 2)/ per_batch_valid_num, 1)  ## l,b 
    regression_loss_output = torch.mean(regression_loss_mean_batch)
    return regression_loss_output

class dense_loss(nn.Module):
    def __ini__(self):
        super(dense_loss, self).__init__()

    def forward(self, dense_color_pred, dense_color, dense_color_valid):
        loss = ((dense_color_pred - dense_color)**2 * dense_color_valid[:,None,None,None]).mean()
        return loss
    

def two_hand_Joints_L1_loss(predict_joint,gt_joint,joint_valid):# -> Any:
    ##repective 3d joints loss
    joint_loss = torch.where(joint_valid[...,None].bool(),
                                predict_joint-gt_joint,
                                torch.zeros_like(predict_joint))

    valid_num = ((joint_valid).sum() + 1)
    return torch.abs(joint_loss).sum() / valid_num /2
#############################################################################################################
###############################################add###########################################################
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


class DETR(nn.Module):
    def __init__(self, backbone,position_embedding,hand_segm,global_interactive_localtransformer,
                args= None, joint_nb=21, stacks=1, channels=256, blocks=1,
                 transformer_depth=1, transformer_head=8,
                 mano_layer=None, mano_neurons=[1024, 512],
                pretrained=True):

        super(DETR, self).__init__()
        self.backbone = backbone
        self.hand_segm = hand_segm
        self.lift_net = IGANet(args)

        # if args.use_2d_queries:
        #     self.single_hand_num_queries = 17 + 21           #38*2=77 21+16+1 LEFT HAND 21+16+1  RIGTH HAND
        # else:
        #     self.single_hand_num_queries = 17   
        # self.num_queries = self.single_hand_num_queries*2 # 34 context decoder 16*2 (pose param) + 1*2(shape param)
        
        # self.scale_trans_embedding_position = 0
        # if cfg.pred_trans_no_nomalize or cfg.normalize_length :
        #     self.num_queries += 1                  #+1 (S and T ;4 dim)
        #     self.scale_trans_embedding_position += 1
        # if cfg.pred_cam_param:
        #     self.num_queries += 1                  #+1 (cam param S and Tx Ty)
        #     self.cam_param_embedding_position = self.scale_trans_embedding_position + 1
        self.single_hand_num_queries = 21 
        self.segm_channel = 16*2 + 1 
        self.multi_scale_feature_num = 4
        self.global_interactive_localtransformer = global_interactive_localtransformer
        # self.hand_transformer = hand_transformer
        self.position_embedding = position_embedding
        self.position_embedding_feature = copy.deepcopy(position_embedding)
        p_hidden_dim = global_interactive_localtransformer.p_d_model
        hidden_dim = global_interactive_localtransformer.d_model

        self.num_queries = self.single_hand_num_queries*2
        self.anisotropic = False
        
        #self.crossmodal_Transformer = crossmodal_Transformer(query_dim=p_hidden_dim,key_dim=hidden_dim,v_dim=hidden_dim)
        self.vis_j_inter_w_whole_map = vis_j_inter_w_whole_map(query_dim=hidden_dim,key_dim=hidden_dim,v_dim=hidden_dim)
        self.j_feature_inter_selfsa = j_feature_inter_selfsa(query_dim=hidden_dim,key_dim=hidden_dim,v_dim=hidden_dim)
        self.point_feature_deformer_fusion_module = PoseTransformer(in_chans=256)
        self.hand_keypoint_3d = MLP(p_hidden_dim, 128, 3,  3)
        init_weights(self.hand_keypoint_3d)




        #mano
        # self.pose_dim = args.pose_dim
        # self.mano_pose_size = 16 * 3
        # self.pose_param_size = 16

        ####keypoint
        # self.use_2d_queries = False
        # if args.use_2d_queries:
        #     #self.hand_keypoint = MLP(hidden_dim, 1024, 21*2, 3)
        #     self.left_hand_keypoint = MLP(hidden_dim, 128, 2,  3)
        #     self.right_hand_keypoint = MLP(hidden_dim, 128, 2,  3)
        #     self.use_2d_queries = True
        self.left_hand_keypoint = MLP(p_hidden_dim, 128, 2,  3)
        self.right_hand_keypoint = MLP(p_hidden_dim, 128, 2,  3)
        # self.cam_param_regressor = MLP(p_hidden_dim+hidden_dim, 128 ,6,2)
        # self.context_feature_ds = MLP(hidden_dim*4,hidden_dim*2,hidden_dim,3)


        # feature_dim = 256 + 32
        # self.context_bone_feature_ds_dir = nn.Sequential(
        #     nn.Conv1d(feature_dim, feature_dim, 1),
        #     #nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Conv1d(feature_dim, feature_dim, 1)
        # )

        # self.context_keypoint_feature_ds_dir = nn.Sequential(
        #     nn.Conv1d(feature_dim, feature_dim, 1),
        #     #nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Conv1d(feature_dim, feature_dim, 1),
        # )
        # self.context_keypoint_feature_layernorm = nn.LayerNorm(feature_dim)
        init_weights(self.left_hand_keypoint)
        init_weights(self.right_hand_keypoint)
        # init_weights(self.cam_param_regressor)

        #init_weights(self.context_feature_ds)

        #特征层可学习编码
        if cfg.use_layer_learnable_embeding:
            self.layer_pos_embed = nn.Parameter(torch.zeros(cfg.dec_layers,p_hidden_dim))


        
        Single_Joints_to_Bone_tensor = torch.zeros((21,20))
        #除根节点
        for i in range(20):
            Single_Joints_to_Bone_tensor[i,i] = 1
            if i % 4 != 0:
                Single_Joints_to_Bone_tensor[i,i] = 0.5
                Single_Joints_to_Bone_tensor[i,i-1] = 0.5
        Single_Joints_to_Bone_tensor[20,[3,7,11,15,19]] = 1/5
        Joints_to_Bone_tensor = torch.zeros((42,40))
        Joints_to_Bone_tensor[:21,:20] = Single_Joints_to_Bone_tensor
        Joints_to_Bone_tensor[21:,20:]  =  Single_Joints_to_Bone_tensor
        Joints_bone_value = Joints_to_Bone_tensor.unsqueeze(0)
        self.register_buffer('Joints_bone_value',Joints_bone_value)
        #self.register_buffer('Joints_bone_mask',Joints_bone_mask)


        # self.channel_ds_conv = nn.Sequential(
        #     nn.Conv2d(hidden_dim, hidden_dim-self.segm_channel, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d( hidden_dim-self.segm_channel),
        #     nn.ReLU(True),
        # )
        #self.channel_ds_conv_all = _get_clones(self.channel_ds_conv, self.multi_scale_feature_num)

        self.decoder_num = cfg.dec_layers
        # self.left_hand_keypoint = _get_clones(left_hand_keypoint,self.decoder_num)
        # self.right_hand_keypoint = _get_clones(right_hand_keypoint,self.decoder_num)


        #init 
        
        # #####shape 10
        # self.right_hand_shape = MLP(hidden_dim, hidden_dim // 2, 10, 3)
        # self.left_hand_shape = MLP(hidden_dim, hidden_dim // 2, 10, 3)
        #####pose 3
        # self.pose = MLP(hidden_dim + 21*2, 1024, 16*6, 3) 先只估计3维
        #self.right_hand_pose = MLP(hidden_dim,hidden_dim//2, 6, 3)
        # if args.pose_dim == 3:
        #     self.right_hand_pose = MLP(hidden_dim, hidden_dim//2, 3, 3)
        #     self.left_hand_pose = MLP(hidden_dim, hidden_dim//2, 3, 3)
        # elif args.pose_dim == 6:
        #     self.right_hand_pose = MLP(hidden_dim, hidden_dim//2, 6, 3)
        #     self.left_hand_pose = MLP(hidden_dim, hidden_dim//2, 6, 3)
        #####T and S 4
        # if cfg.normalize_length:
        #     self.scale_trans = MLP(hidden_dim, hidden_dim, 4, 3)
        # else:
        #     if args.pred_trans_no_nomalize:
        #         self.rel_trans = MLP(hidden_dim, hidden_dim, 3, 3)
        #     if args.pred_cam_param:
        #         if args.pred_like_ACR:
        #             self.cam_param_reg = MLP(hidden_dim, hidden_dim, 6, 3)  #前三维右手，后三维左手
        #         else:
        #             self.cam_param_reg = MLP(hidden_dim, hidden_dim, 3, 3)  #前三维右手，后三维左手

        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        

        # self.query_embed_local = nn.Embedding(num_queries, hidden_dim)
        self.backbone_name = 'fpn'
        # if args.backbone == 'resnet50':
        #     num_channels = 512
        #     self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
            
        ###mano branch from double hands （21 iccv）
        #self.mano_layer = mano_layer
        #J_regressor_mano_ih26m = np.load(osp.join(cfg.smplx_path, "mano","J_regressor_mano_ih26m.npy"))
        #self.register_buffer("J_regressor_mano_ih26m",torch.FloatTensor(J_regressor_mano_ih26m).unsqueeze(0)) # 1 x 21 x 778


        ##self.mano_branch = mano_regHead(mano_layer,coord_change_mat=coord_change_mat)

        #self.obj_keypoints = MLP(hidden_dim, 128, 2, 3)
        # self.obj_trans = MLP(hidden_dim, hidden_dim, 3, 3)
        # num_queries = 21
        #self.query_embed_object = nn.Embedding(num_queries, hidden_dim)

        self.inp_res = args.input_img_shape[0]
        assert args.input_img_shape[0] == args.input_img_shape[1] 
        
        

        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        self.joint_num = 21
        
        if cfg.pred_2d_depth:
            self.detrloss = detr_loss_depth
        else:
            self.detrloss = detr_loss

        self.detrloss_2d = detr_loss_2d

        self.detrloss_L1 = two_hand_Joints_L1_loss

        # #self.context_loss = ContrastiveLoss(16*42)
        # self.context_loss = detr_context_loss
        # self.context_norm = nn.LayerNorm(288)
        # # self.context_norm_list = _get_clones(self.context_norm,4)


        #self.activate = nn.LeakyReLU()
        self.activate = torch.nn.Identity()

        # self.dense_loss = dense_loss()

        ###################################################segm##################################################
        if cfg.use_seg_loss:
            self.loss1 = smp.losses.DiceLoss(mode='multiclass',from_logits = True)
            self.loss2 = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1,ignore_index = None)
            self.loss3 = smp.losses.FocalLoss(alpha=0.25,mode='multiclass',ignore_index = None)
        
        
        
        ###################################################diffision#############################################
        if cfg.use_diffusion:
            self.num_steps = cfg.num_steps
            (
                self.beta,
                self.alpha,
                self.alpha_hat,
                self.sigma,
            ) = compute_noise_scheduling(
                schedule=cfg.schedule,
                beta_start=cfg.beta_start,
                beta_end=cfg.beta_end,
                num_steps=cfg.num_steps,
            )
            
            self.alpha_torch = (
            torch.tensor(self.alpha)
            .float()
            #.to(self.device)
            .unsqueeze(1)
            .unsqueeze(1)
            )
            self.guidance_before_cond = False
            self.p_uncond = cfg.p_uncond
            
            self.conditioning = detr_condition(
                                encode_3d = cfg.encode_3d,
                                cond_mix_mode = cfg.cond_mix_mode,
                                embed_dim = hidden_dim,
                                condition_out_dim = cfg.condition_out_dim
                                
            )
            self._extra_input = None
            self.strategy=cfg.sample_strategy
            self.n_samples=cfg.n_samples

            
            
            ##############################skeleton####################################
            # Joints in InterHand3.6M -- data has 32 joints, but only 17 that move; these are the
            # indices.
            # H36M_NAMES = [""] * 42
            # H36M_NAMES[0] = "Hip"
            # H36M_NAMES[1] = "RHip"
            # H36M_NAMES[2] = "RKnee"
            # H36M_NAMES[3] = "RFoot"
            # H36M_NAMES[6] = "LHip"
            # H36M_NAMES[7] = "LKnee"
            # H36M_NAMES[8] = "LFoot"
            # H36M_NAMES[12] = "Spine"
            # H36M_NAMES[13] = "Thorax"
            # H36M_NAMES[14] = "Neck/Nose"
            # H36M_NAMES[15] = "Head"
            # H36M_NAMES[17] = "LShoulder"
            # H36M_NAMES[18] = "LElbow"
            # H36M_NAMES[19] = "LWrist"
            # H36M_NAMES[25] = "RShoulder"
            # H36M_NAMES[26] = "RElbow"
            # H36M_NAMES[27] = "RWrist"

            # Human3.6m Skeleton
            self.skeleton = Skeleton(
                parents=[
                    1,2,3,20,
                    5,6,7,20,
                    9,10,11,20,
                    13,14,15,20,
                    17,18,19,20,
                    -1,
                    22,23,24,41,
                    26,27,28,41,
                    30,31,32,41,
                    34,35,36,41,
                    38,39,40,41,
                    -1                       
                ],
                joints_left=[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],
                joints_right=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
            )
            skeleton_joints_group_hand = [
                [0,1,2,3],[4,5,6,7],[8,9,10,11,20],[12,13,14,15],[16,17,18,19],
                [21,22,23,24],[25,26,27,28],[29,30,31,32,41],[33,34,35,36],[37,38,39,40]
            ]
            self.skeleton._joints_group = skeleton_joints_group_hand

            
            
            self.n_output_coords = 3
            if cfg.arch == "gcn":
                self.diffmodel = LiftCTGCN(
                    cfg,
                    seq_len=1,
                    skeleton=self.skeleton,
                    n_input_coords=self.conditioning.out_dim,
                    n_output_coords=self.n_output_coords,
                )
            elif cfg.arch == "mixste":
                self.diffmodel = DiffMixSTE(
                    num_frame=1,
                    num_joints=self.skeleton.num_joints(),
                    in_chans=self.conditioning.out_dim,
                    out_dim=self.n_output_coords,
                    embed_dim=cfg.channels,
                    depth=cfg.layers,
                    num_heads=cfg.nheads,
                    drop_path_rate=cfg.drop_path_rate,
                )
            else:
                raise ValueError(
                    "Architecture param could not be recognized: "
                    f"{cfg['arch']}. Only gcn is implemented for lifting."
                )
        #self.single_hand_mano2inter_reorder = [16,15,14,13,17,3,2,1,18,6,5,4,19,12,11,10,20,9,8,7,0]  #mano顺序 -> interhand 需要在确认一下指尖的valid
        #self.double_hand_mano2inter_reorder = self.single_hand_mano2inter_reorder + [i+self.joint_num  for i in self.single_hand_mano2inter_reorder]
        # FPN-Res50 backbone
        # self.base_net = FPN(pretrained=pretrained)

        # # hand head
        # self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb,
        #                               stacks=stacks, channels=channels, blocks=blocks)
        # # hand encoder
        # self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
        #                                  size_input_feature=(roi_res, roi_res))
        # mano branch
        # self.mano_branch = mano_regHead(mano_layer, feature_size=self.hand_encoder.num_feat_out,
        #                                 mano_neurons=mano_neurons, coord_change_mat=coord_change_mat)
        # object head
        #self.reg_object = reg_object

        # if reg_object:
        #     # self.transformer_object = copy.deepcopy(transformer)
        #     # self.obj_transformer = obj_transformer
        #     self.input_proj_object = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        #     self.obj_keypoints = MLP(hidden_dim, 128, 2, 3)
            # self.obj_trans = MLP(hidden_dim, hidden_dim, 3, 3)
            # num_queries = 21
            # self.query_embed_object = nn.Embedding(num_queries, hidden_dim)
            # self.query_embed_object_rot = MLP(64, 1024, 3*3, 6)
        # self.query_embed_pos_hand_obj = MLP(4, hidden_dim, hidden_dim, 3)

    # def pose_shape2verts_joints(self,pose,shape,side):
    #     output = self.mano_layer[side](global_orient=pose[:,:3], hand_pose=pose[:,3:], betas=shape)
    #     #J = output.joints
    #     # fingertips = output.vertices[:,[745, 317, 444, 556, 673]]
    #     # joints = torch.cat([J,fingertips], dim=1) * 1000
    #     if (cfg.finger_tips_use_interhand) and (not cfg.pred_use_mano):            
    #         joints = torch.matmul(self.J_regressor_mano_ih26m,output.vertices)*1000  # N x 21 x 3 ,mm
    #     else:
    #         J = output.joints
    #         fingertips = output.vertices[:,[745, 317, 444, 556, 673]]
    #         joints = torch.cat([J,fingertips], dim=1) * 1000
            
    #     if  (cfg.pred_use_mano) and cfg.joint_use_gt_anno:
    #         joints = joints[:,self.single_hand_mano2inter_reorder]
    #     verts = output.vertices * 1000
    #     # T_pose = self.mano_layer[side].template_J.unsqueeze(0) + \
    #     #             (self.mano_layer[side].beta2J[None,:,:,:] * shape[:,None,None,:]).sum(dim = -1)
    #     # T_pose = T_pose * 1000
    #     T_pose = joints
    #     results = {"verts3d": verts, "joints3d": joints, "shape": shape, "pose": pose, "T_pose": T_pose}
    #     return results
        self.init_weights()

        # def hook_fn(module, input, output):
        #     print(module[0].weight)
        # self.context_keypoint_feature_ds_dir.register_forward_hook(hook_fn)



     
    def init_weights(self):
        for name, m in self.lift_net.named_modules():
            init_weights_like_dir(m)
        # for name, m in self.context_bone_feature_ds_dir.named_modules():
        #     init_weights_like_dir(m)
        # for name, m in self.context_keypoint_feature_ds_dir.named_modules():
        #     init_weights_like_dir(m)
        



    def apply_conditioning(
        self, target_3d, data_2d, cam, data_3d, p_uncond=0.0
    ):
        if self.guidance_before_cond:
            # Classifier-free guidance before conditioners
            data_2d = apply_guidance(
                data_2d, p_uncond, self.rand_guidance_rate
            )
            p_uncond = 0.0  # disable guidance after conditioning

        if self._extra_input is None:
            diff_input = self.conditioning(
                target_3d,
                data_2d,
                p_uncond=p_uncond,
            )
        elif self._extra_input == "cam":
            diff_input = self.conditioning(
                target_3d,
                data_2d,
                cam,
                p_uncond=p_uncond,
            )
        elif self._extra_input == "gt":
            diff_input = self.conditioning(
                target_3d,
                data_3d,
                p_uncond=p_uncond,
            )
        else:
            raise ValueError(
                "_extra_input can only take values None, 'cam' and "
                f"'gt'. Got {self._extra_input}."
            )
        return diff_input
    
    def impute(
        self,
        data_2d,
        n_samples,
        cam=None,
        data_3d=None,
        starting_pose=None,
        n_steps=None,
    ):
        B, J, D = data_2d.shape
        L = 1
        device = data_2d.device
        #assert D == 2, f"Expected input dimension 2, got {D}."

        if n_steps is None:
            n_steps = self.num_steps

        imputed_samples = torch.zeros(B, n_samples, 3, J, L).to(device)

        for i in range(n_samples):
            if starting_pose is None:
                target_3d = torch.randn(
                    (B, self.n_output_coords, J, L),
                    device=device,
                )

                if self.anisotropic:
                    target_3d = (
                        torch.mm(
                            self.err_sqrt_cov,
                            target_3d.permute(2, 1, 0, 3).reshape(
                                J * 3, B * L
                            ),
                        )
                        + self.err_mean
                    )
                    target_3d = target_3d.reshape(J, 3, B, L).permute(
                        2, 1, 0, 3
                    )
            else:
                assert isinstance(starting_pose, torch.Tensor), (
                    "stating_pose should be a torch tensor. "
                    f"Got {type(starting_pose)}."
                )
                assert starting_pose.shape == (B, self.n_output_coords, J, L)
                target_3d = starting_pose.to(device)

            for t in range(n_steps - 1, -1, -1):
                diff_input = self.apply_conditioning(
                    target_3d,
                    data_2d,
                    cam,
                    data_3d,
                    p_uncond=0.0,
                )

                # Conditional noise prediction
                predicted_noise = self.diffmodel(
                    diff_input, torch.tensor([t]).to(device)
                )  # (B,3,J,L)

                # Classifier-free guidance
                if self.p_uncond > 0 and self.guidance > 0.0:
                    # Unconditional input
                    diff_input_uncond = self.apply_conditioning(
                        target_3d,
                        data_2d,
                        cam,
                        data_3d,
                        p_uncond=1.0,
                    )
                    predicted_noise *= 1 + self.guidance
                    predicted_noise -= self.guidance * self.diffmodel(
                        diff_input_uncond, torch.tensor([t]).to(device)
                    )  # (B,3,J,L)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_mean = coeff1 * (target_3d - coeff2 * predicted_noise)

                if t > 0:
                    noise = torch.randn_like(current_mean)
                    target_3d = current_mean + self.sigma[t - 1] * noise

            if self.conditioning.mix_mode != "z_only":
                imputed_samples[:, i] = current_mean.detach()
            else:
                # Use input x,y and output z
                imputed_samples[:, i, 2:, :, :] = current_mean.detach()
                imputed_samples[:, i, :2, :, :] = diff_input[:, 1:].detach()
        return imputed_samples
    
    def evaluate(self, batch, n_samples, starting_pose=None, n_steps=None):
        get_3d = self._extra_input == "gt"
        data_3d, data_2d, cam = self.process_data(batch, get_3d=get_3d)

        with torch.no_grad():
            samples = self.impute(
                data_2d,
                n_samples,
                cam=cam,
                data_3d=data_3d,
                starting_pose=starting_pose,
                n_steps=n_steps,
            )
        return samples
    
    # def forward(self, inputs, targets, meta_info, mode):
    def forward(self, img, bbox_hand_left, bbox_hand_right):
        mode = 'test'
        imgs = self.normalize_img(img).detach()
        # imgs = self.normalize_img(inputs['img']).detach()
        # bbox_hand_left = meta_info["bbox_hand_left"]
        # bbox_hand_right = meta_info["bbox_hand_right"]

        batch = imgs.shape[0]

        if not isinstance(imgs, NestedTensor):
            samples = nested_tensor_from_tensor_list(imgs)
        #features = self.backbone(samples.tensors)[0]  #[1, 256, 128, 128] [16, 256, 64, 64]
        p2, p3, p4, p5 = self.backbone(samples.tensors)  # [16, 256, 64, 64] [16, 256, 32, 32] [16, 256, 16, 16] [16, 256, 8, 8]
        features_list = [p2, p3, p4, p5]
        features = p2
        if cfg.use_seg_loss:
            segm_dict = self.hand_segm(p2)
            segms_dict = segm_dict['segm_logits']
            #pred_dense_color = segm_dict['dense_color']

        
        #input project
        if self.backbone_name == 'resnet50':
            features = self.input_proj(features)
        
        bs,c,w,h = features.shape
        assert w==h,"特征图长宽相等"
        ##### bbox downsample
        #for left hand
        #downsample radio
        downsample_radio = self.inp_res / w
        left_hand_bbox = bbox_hand_left.clamp(0.,self.inp_res) #256
        left_hand_bbox /= downsample_radio  #0-256/4
        #for right hand
        right_hand_bbox = bbox_hand_right.clamp(0.,self.inp_res) #256
        right_hand_bbox /= downsample_radio


        # hand = bbox_hand.clamp(0.,inp_res)
        # for i in range(batch):  #右上角（x,y）+左下角(x,y)
        #     if hand[i,0] >= hand[i,2]:
        #         hand[i,2] = inp_res
        #         hand[i,0] = 0.
        #     if hand[i,1] >= hand[i,3]:
        #         hand[i,3] = inp_res
        #         hand[i,1] = 0.

        left_hand_bbox = np.round(left_hand_bbox.cpu().numpy())
        right_hand_bbox = np.round(right_hand_bbox.cpu().numpy())

        #####hand_mask
        mask_left_hand = ~ F.interpolate(samples.mask[None].float(), size=(features.shape[2], features.shape[3])).to(torch.bool)[0]  #需屏蔽的为1，全部为1
        mask_right_hand = mask_left_hand.clone()
        # mask_global = [] 
        mask_left_hands = []
        mask_right_hands = []
        batch_diff_size_mask_two_hand = []
        for i in range(batch):
            diff_size_mask_two_hand= []
            if int(left_hand_bbox[i,1]) >= int(left_hand_bbox[i,3]) or int(left_hand_bbox[i,0]) >= int(left_hand_bbox[i,2]):
                mask_left_hand[i] = False
            else:
                mask_left_hand[i,int(left_hand_bbox[i,1]):int(left_hand_bbox[i,3]),int(left_hand_bbox[i,0]):int(left_hand_bbox[i,2])] = False
            if int(right_hand_bbox[i,1]) >= int(right_hand_bbox[i,3]) or int(right_hand_bbox[i,0]) >= int(right_hand_bbox[i,2]):
                mask_right_hand[i] = False
            else:
                mask_right_hand[i,int(right_hand_bbox[i,1]):int(right_hand_bbox[i,3]),int(right_hand_bbox[i,0]):int(right_hand_bbox[i,2])] = False
            
            mask_left_hands.append(mask_left_hand[i].clone())
            mask_right_hands.append(mask_right_hand[i].clone())
            
            mask_left_hand[i] = False  #全部置为false
            diff_size_mask_two_hand.append(mask_left_hand[i].clone().unsqueeze(0).repeat(self.num_queries,1,1))
            batch_diff_size_mask_two_hand.append(torch.cat(diff_size_mask_two_hand,dim = 0))
        #     print( torch.unique(mask_hands[i],return_counts=True))
        #     print( torch.unique(diff_size_mask_hand_objs[i][0],return_counts=True))
        #     print( torch.unique(diff_size_mask_hand_objs[i][22],return_counts=True))
        # exit()
        batch_diff_size_mask_two_hand = torch.stack(batch_diff_size_mask_two_hand)
        mask_left_hands = torch.stack(mask_left_hands)
        mask_right_hands = torch.stack(mask_right_hands)

        

        #backbone and pos embedding
        mask_feature = F.interpolate(samples.mask[None].float(), size=(features.shape[2], features.shape[3])).to(torch.bool)[0]  #全部为0,不需要进行mask
        #Nest_feature = NestedTensor(features, mask_feature) #backbone出来的特征进行位置编码
        #pos_feature = (self.position_embedding_feature(Nest_feature.tensors,Nest_feature.mask)[0]).to(Nest_feature.tensors.dtype)

        batch_diff_size_mask_two_hand_list = []
        mask_left_hands_list = []
        mask_right_hands_list = []
        pos_feature_list = []
        pred_segms_dict_softmax_list = []
        pred_segms_dict_softmax_list_all = []
        features_list_with_vis_info__hidden_dim_256 = [] 

        for i in range(len(features_list)):
            downsample_radio = features_list[i].shape[-1]/batch_diff_size_mask_two_hand.shape[-1]
            
            #batch_diff_size_mask_two_hand_list.append(F.interpolate(batch_diff_size_mask_two_hand.float(), scale_factor = downsample_radio , mode="nearest").bool())
            #mask_left_hands_list.append(F.interpolate(mask_left_hands.float()[:,None,:,:], scale_factor = downsample_radio , mode="nearest").bool().squeeze())
            #mask_right_hands_list.append(F.interpolate(mask_right_hands.float()[:,None,:,:], scale_factor = downsample_radio , mode="nearest").bool().squeeze())
            mask_feature_multi_scale = F.interpolate(mask_feature.float()[:,None,:,:], scale_factor = downsample_radio , mode="nearest").bool().squeeze()

            #pred_segms_dict_ds = F.interpolate(segms_dict,scale_factor = downsample_radio/2 , mode = "bilinear")
            #avg pooling
            pred_segms_dict_ds = F.avg_pool2d(segms_dict,kernel_size = int((1/downsample_radio))*2,stride = int((1/downsample_radio))*2)
            #max pooling
            #pred_segms_dict_ds = F.max_pool2d(segms_dict,kernel_size = int((1/downsample_radio))*2,stride = int((1/downsample_radio))*2)
            
            #通道以及空间上softmax
            #pred_segms_dict_softmax = rearrange( (rearrange(pred_segms_dict_ds.softmax(1),'b c h w -> b c (h w)').softmax(-1)) ,'b c (h w) -> b c h w', **parse_shape(pred_segms_dict_ds,'b c h w'))
            #仅仅通道维度上进行softmax
            pred_segms_dict_softmax = pred_segms_dict_ds.softmax(1)
            features_with_vis_info__hidden_dim_256 =  torch.cat((features_list[i],pred_segms_dict_softmax[:,1:]),dim=1)
            features_list_with_vis_info__hidden_dim_256.append(features_with_vis_info__hidden_dim_256)

            pred_segms_dict_softmax_list.append(pred_segms_dict_softmax[:,1:])
            pred_segms_dict_softmax_list_all.append(pred_segms_dict_softmax)

            pos_multi_scale =  self.position_embedding_feature(features_list[i], mask_feature_multi_scale)[0].to(features_list[i].dtype)
            if cfg.use_layer_learnable_embeding:
                pos_multi_scale = self.layer_pos_embed[i].view(1,-1,1,1) + pos_multi_scale
            pos_feature_list.append(pos_multi_scale)

        #preds_joints = None
        #gt_dense_color_ds_128 = F.avg_pool2d(targets['dense'],kernel_size = 2,stride = 2)

        ######obj
        # #idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        # roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)
        # y = ops.roi_align(features, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,  #[4, 256, 32, 32]
        #                       sampling_ratio=-1)  # obj
        # mask_object = F.interpolate(samples.mask[None].float(), size=(self.out_res, self.out_res)).to(torch.bool)[0]
        # Nest_feature_object = NestedTensor(y, mask_object) #物体的roi特征进行位置编码
        # pos_feature_obj = self.position_embedding(Nest_feature_object).to(Nest_feature_object.tensors.dtype)

        hs_left_hand,hs_right_hand, hs_left_hand_context,hs_right_hand_context,tgt_content_list,hs_weak_cam_param,att_map_list= self.global_interactive_localtransformer(features_list_with_vis_info__hidden_dim_256, self.query_embed.weight, pos_feature_list,self.single_hand_num_queries,pred_segms_dict_softmax_list,imgs_tensor=img,use_all = True)
        #分别是backbone出来的特征，backbone特征的mask（全部为0），query的权重，backbone特征的位置编码，手的mask，手物query数量的mask,物体roi后的特征，object的mask（32*32）
        # hs = self.hand_transformer(self.input_proj(features), mask_feature, self.query_embed.weight, pos_feature,mask_hands,diff_size_mask_hand_objs)
        # print(hs.shape)
        # exit()
        # hs = hs[-1].unsqueeze(0)
        # preds_hand_keypoint = self.hand_keypoint(hs[:,:,0:21]).sigmoid()
        # preds_hand_keypoint = preds_hand_keypoint.permute(1,0,2,3)
        # intput_hand = torch.cat([preds_hand_keypoint,hs],dim = -1)
        # preds_hand_keypoint = preds_hand_keypoint.flatten(2).reshape(batch,-1,21,2)

        #### shape reg
        # pred_left_hand_mano_shape = self.left_hand_shape(hs_left_hand[:,:,self.single_hand_num_queries-1]).flatten(2)
        # pred_right_hand_mano_shape = self.right_hand_shape(hs_right_hand[:,:,self.single_hand_num_queries-1]).flatten(2)

        #### pose reg
        # pred_left_hand_mano_pose = self.left_hand_pose(hs_left_hand[:,:,self.single_hand_num_queries-1-self.pose_param_size:self.single_hand_num_queries-1]).flatten(2)
        # pred_right_hand_mano_pose = self.right_hand_pose(hs_right_hand[:,:,self.single_hand_num_queries-1-self.pose_param_size:self.single_hand_num_queries-1]).flatten(2)

        #### scale_trans reg | cam param reg
        # if cfg.normalize_length:
        #     scale_trans_embedding =  other_embeddings[:,:,self.scale_trans_embedding_position - 1]
        #     pred_scale_trans = self.scale_trans(scale_trans_embedding.squeeze(0)).flatten(1)
        #     pred_cam_param = None
        # else:
        #     if cfg.pred_trans_no_nomalize:
        #         scale_trans_embedding =  other_embeddings[:,:,self.scale_trans_embedding_position - 1]
        #         pred_scale_trans = self.rel_trans(scale_trans_embedding.squeeze(0)).flatten(1)  #pred_rel_trans_cam
        #     else:
        #         pred_scale_trans = None
        #     if cfg.pred_cam_param:
        #         cam_param_embedding = other_embeddings[:,:,self.cam_param_embedding_position - 1]
        #         pred_cam_param = self.cam_param_reg(cam_param_embedding.squeeze(0)).flatten(2).transpose(1,0)
        #     else:
        #         pred_cam_param = None
        #### 2d keypoints reg 
        # if self.use_2d_queries:
        #     pred_left_hand_2d_keypoints = self.left_hand_keypoint(hs_left_hand[:,:,0:21]).sigmoid().permute(1,0,2,3) #bs,layer_num,21,2
        #     pred_right_hand_2d_keypoints = self.right_hand_keypoint(hs_right_hand[:,:,0:21]).sigmoid().permute(1,0,2,3) 
        # else:
        #     pred_left_hand_2d_keypoints,pred_right_hand_2d_keypoints = None,None

        #pred_mano_results, gt_mano_results = self.mano_branch(pred_mano_pose,pred_mano_shape, mano_params=mano_params, roots3d=roots3d)

        # preds_obj = {}
        
        # left_mano_para_list = {"verts3d": [], "joints3d": [], "shape": [], "pose": [], "T_pose": []}
        # right_mano_para_list = {"verts3d": [], "joints3d": [], "shape": [], "pose": [], "T_pose": []}

        # for layer_index in range(len(pred_left_hand_mano_shape)):
        #     ##left hand
        #     pred_left_hand_mano_shape_layer = pred_left_hand_mano_shape[layer_index]
        #     pred_left_hand_mano_pose_layer = pred_left_hand_mano_pose[layer_index]
        #     if self.pose_dim == 6:
        #         #pred_left_mano_pose_aa_list = []
        #         #for bs_left_hand in range(len(pred_left_hand_mano_pose_layer)):
        #         pred_left_mano_pose_rotmat = rot6d2mat(pred_left_hand_mano_pose_layer.view(-1, 6)).view(-1, 16, 3, 3).contiguous()
        #         pred_left_mano_pose_aa = mat2aa(pred_left_mano_pose_rotmat.view(-1, 3, 3)).contiguous().view(-1,self.mano_pose_size)
        #             #pred_left_mano_pose_aa_list.append(pred_left_mano_pose_aa)
        #         #left_mano_para = self.pose_shape2verts_joints(torch.stack(pred_left_mano_pose_aa_list,dim=0),pred_left_hand_mano_shape_layer,"l")
        #         left_mano_para = self.pose_shape2verts_joints(pred_left_mano_pose_aa,pred_left_hand_mano_shape_layer,"l")
        #     elif self.pose_dim == 3:
        #         left_mano_para = self.pose_shape2verts_joints(pred_left_hand_mano_pose_layer,pred_left_hand_mano_shape_layer,"l")

        #     ##right hand
        #     pred_right_hand_mano_shape_layer = pred_right_hand_mano_shape[layer_index]
        #     pred_right_hand_mano_pose_layer = pred_right_hand_mano_pose[layer_index]
        #     if self.pose_dim == 6:
        #         #pred_right_mano_pose_aa_list = []
        #         #for bs_right_hand in range(len(pred_right_hand_mano_pose_layer)):
        #         pred_right_mano_pose_rotmat = rot6d2mat(pred_right_hand_mano_pose_layer.view(-1, 6)).view(-1, 16, 3, 3).contiguous()
        #         pred_right_mano_pose_aa = mat2aa(pred_right_mano_pose_rotmat.view(-1, 3, 3)).contiguous().view(-1,self.mano_pose_size)
        #             #pred_right_mano_pose_aa_list.append(pred_right_mano_pose_aa) 
        #         #$right_mano_para = self.pose_shape2verts_joints(torch.stack(pred_right_mano_pose_aa_list,dim=0),pred_right_hand_mano_shape_layer,"r")
        #         right_mano_para = self.pose_shape2verts_joints(pred_right_mano_pose_aa,pred_right_hand_mano_shape_layer,"r")
        #     elif self.pose_dim == 3:
        #         right_mano_para = self.pose_shape2verts_joints(pred_right_hand_mano_pose_layer,pred_right_hand_mano_shape_layer,"r")
          
        #     for k,v in left_mano_para.items():
        #         left_mano_para_list[k].append(v)
        #     for k,v in right_mano_para.items():
        #         right_mano_para_list[k].append(v)
        # for k in left_mano_para_list.keys():
        #     left_mano_para_list[k]=torch.stack(left_mano_para_list[k],dim=1)
        # for k in right_mano_para_list.keys():
        #     right_mano_para_list[k]=torch.stack(right_mano_para_list[k],dim=1)
        ##TODO
        device = hs_left_hand.device
        # keypoints_coord_left_hand =  (self.activate(self.left_hand_keypoint(hs_left_hand[:,:,0:21]))*w) #layer_num,bs,21,3
        # keypoints_coord_right_hand =  (self.activate(self.right_hand_keypoint(hs_right_hand[:,:,0:21]))*w)
        # keypoints_coord = torch.cat((keypoints_coord_right_hand,keypoints_coord_left_hand),dim=-2)

        #keypoints_coord_list = []
        # for layer_num in range(hs_left_hand.shape[0]):
        #     keypoints_coord_left_hand_layer =  (self.activate(self.left_hand_keypoint[layer_num](hs_left_hand[i,:,0:21]))*w) #layer_num,bs,21,3
        #     keypoints_coord_right_hand_layer =  (self.activate(self.right_hand_keypoint[layer_num](hs_right_hand[i,:,0:21]))*w)
        #     keypoints_coord_layer = torch.cat((keypoints_coord_right_hand_layer,keypoints_coord_left_hand_layer),dim=-2)
        #     keypoints_coord_list.append(keypoints_coord_layer)
        # keypoints_coord = torch.stack(keypoints_coord_list,dim=0)

        if not cfg.use_diffusion or cfg.guide_condition_feature_w_joints_loss:
            keypoints_coord_left_hand =  (self.activate(self.left_hand_keypoint(hs_left_hand[:,:,0:21]))*cfg.input_img_shape[0]) #layer_num,bs,21,3
            keypoints_coord_right_hand =  (self.activate(self.right_hand_keypoint(hs_right_hand[:,:,0:21]))*cfg.input_img_shape[0])
            keypoints_coord = torch.cat((keypoints_coord_right_hand,keypoints_coord_left_hand),dim=-2)
            #regression_loss = self.detrloss_2d(keypoints_coord, targets['joint_coord'], meta_info['joint_valid'])



        #2d
        # keypoints_coord = torch.cat((keypoints_coord_right_hand,keypoints_coord_left_hand),dim=-2)
        # if (not cfg.use_diffusion or cfg.guide_condition_feature_w_joints_loss) and mode == 'train': 
        #     regression_loss = self.detrloss_2d(keypoints_coord, targets['joint_coord'], meta_info['joint_valid'])
            #regression_loss = self.detrloss_2d(keypoints_coord[0].unsqueeze(0), targets['joint_coord'], meta_info['joint_valid'])

        #3d
        if cfg.use_lift_net:
            # # #使用context embeding
            # #option 1 直接使用detr出来的context embeding
            # embeding_2d_l = torch.cat((hs_left_hand[-1],hs_left_hand_context[-1]),dim=-1) 
            # embeding_2d_r = torch.cat((hs_right_hand[-1],hs_right_hand_context[-1]),dim=-1)

            # #option 2 在多层特征图上找原图上找特征
            # hs_left_hand_context_from_feature =  self.context_feature_ds(get_context_bone_embeding_from_feature_use_initial_pred_sample_point(keypoints_coord_left_hand[-1],features_list_with_vis_info__hidden_dim_256))
            # hs_right_hand_context_from_feature =  self.context_feature_ds(get_context_bone_embeding_from_feature_use_initial_pred_sample_point(keypoints_coord_right_hand[-1],features_list_with_vis_info__hidden_dim_256))
            # embeding_2d_l = torch.cat((hs_left_hand[-1],hs_left_hand_context_from_feature),dim=-1) 
            # embeding_2d_r = torch.cat((hs_right_hand[-1],hs_right_hand_context_from_feature),dim=-1)

            # #option 3 在多层特征图上找原图上找特征，但是使用骨骼信息来找，更加精细化,代码参考dir
            # bone_embeding_l_c = get_context_bone_embeding_from_feature_use_initial_pred(keypoints_coord_left_hand[-1],[features_list_with_vis_info__hidden_dim_256[0]]).permute(0,2,1).detach()
            # bone_embeding_r_c = get_context_bone_embeding_from_feature_use_initial_pred(keypoints_coord_right_hand[-1],[features_list_with_vis_info__hidden_dim_256[0]]).permute(0,2,1).detach() #self.bone_embeding_context_ds
            # bone_embeding_c = self.context_bone_feature_ds_dir(torch.cat([bone_embeding_r_c,bone_embeding_l_c],dim=-1))
            # joints_context_from_bone = self.context_keypoint_feature_ds_dir(get_each_joint_context_bone_context(keypoints_coord[-1],bone_embeding_c,self.Joints_bone_value).permute(0,2,1)).permute(0,2,1)
            # joints_context_from_bone = self.context_keypoint_feature_layernorm(joints_context_from_bone)
            # embeding_2d_l = torch.cat((hs_left_hand[-1],joints_context_from_bone[:,self.single_hand_num_queries:]),dim=-1) 
            # embeding_2d_r = torch.cat((hs_right_hand[-1],joints_context_from_bone[:,:self.single_hand_num_queries]),dim=-1)

            # #option 4 在多层特征图上找原图上找特征(只是用身份特征)，但是使用骨骼信息来找，更加精细化,代码参考dir 
            # bone_embeding_l_c = get_context_bone_embeding_from_feature_use_initial_pred(keypoints_coord_left_hand[-1],pred_segms_dict_softmax_list).permute(0,2,1)
            # bone_embeding_r_c = get_context_bone_embeding_from_feature_use_initial_pred(keypoints_coord_right_hand[-1],pred_segms_dict_softmax_list).permute(0,2,1)#self.bone_embeding_context_ds
            # bone_embeding_c = self.context_bone_feature_ds_dir(torch.cat([bone_embeding_r_c,bone_embeding_l_c],dim=-1))
            # joints_context_from_bone = self.context_keypoint_feature_ds_dir(get_each_joint_context_bone_context(keypoints_coord[-1],bone_embeding_c,self.Joints_bone_value).permute(0,2,1)).permute(0,2,1) #.detach()

            # joints_context_from_bone = self.context_keypoint_feature_layernorm(joints_context_from_bone)

            # embeding_2d_l = torch.cat((hs_left_hand[-1],joints_context_from_bone[:,self.single_hand_num_queries:]),dim=-1) 
            # embeding_2d_r = torch.cat((hs_right_hand[-1],joints_context_from_bone[:,:self.single_hand_num_queries]),dim=-1)

            # ##option 5 adaption method feature map -> pos 
            # all_joints_pos_embeding = torch.cat((hs_right_hand[-1],hs_left_hand[-1]),dim=1)
            #adaption_pos_embeding = self.crossmodal_Transformer(all_joints_pos_embeding,features_list_with_vis_info__hidden_dim_256,pos_feature_list)

            # ##option 6 2d joints + image feature adaption 多平行线
            # mask= get_context_bone_mask_from_feature_use_initial_pred_two_hand_mutiscale(keypoints_coord[-1])
            # keypoint_mask = keypoints_mask_from_bone_info(mask)
            # all_joints_pos_embeding = torch.cat((hs_right_hand[-1],hs_left_hand[-1]),dim=1)
            # adaption_pos_embeding = self.crossmodal_Transformer(all_joints_pos_embeding,features_list_with_vis_info__hidden_dim_256,pos_feature_list,keypoint_mask)

            # # # # 不使用context embeding
            # embeding_2d_l = hs_left_hand[-1]
            # embeding_2d_r = hs_right_hand[-1]

            # ##option 5、6
            # embeding_2d = adaption_pos_embeding
            ##other option
            # embeding_2d = torch.cat((embeding_2d_r,embeding_2d_l),dim=-2)  #注意在图卷积中关节点定义顺序为先右后左

            
            embeding_2d_l = hs_left_hand[-2:-1]
            embeding_2d_r = hs_right_hand[-2:-1]
            embeding_2d = torch.cat((embeding_2d_r,embeding_2d_l),dim=-2)  #注意在图卷积中关节点定义顺序为先右后左
            # embeding_2d = torch.cat([embeding_2d_r,embeding_2d_l],dim=-2)  #注意在图卷积中关节点定义顺序为先右后左

            
            ##option 8 attention guided gcn
            ##sub option 1 全图
            #hs_hand_j_feature =  get_context_bone_embeding_from_feature_use_initial_pred_sample_point_multi_scale(keypoints_coord,[features_list_with_vis_info__hidden_dim_256[0]])
            #hs_hand_j_feature_inter_w_whole_map =  self.vis_j_inter_w_whole_map(hs_hand_j_feature,[features_list_with_vis_info__hidden_dim_256[0]])   #max feature map-> min feature map

            # ##sub option 2 deformer detr
            # ref = 2*keypoints_coord[-1]/256 -1
            # hs_hand_j_feature_inter_w_whole_map = self.point_feature_deformer_fusion_module(embeding_2d[0],ref,[features_list_with_vis_info__hidden_dim_256[0]])

            # ###sub option 3 根据部分分割图作为mask与全图做attention
            # no_bg_segms = ((pred_segms_dict_softmax_list_all[0].max(1))[1] == 0)
            # hs_hand_j_feature =  get_context_bone_embeding_from_feature_use_initial_pred_sample_point_multi_scale(keypoints_coord,[features_list_with_vis_info__hidden_dim_256[0]])
            # hs_hand_j_feature_inter_w_whole_map =  self.vis_j_inter_w_whole_map(hs_hand_j_feature,[features_list_with_vis_info__hidden_dim_256[0]],no_bg_segms)   #max feature map-> min feature map

            # # ###sub option 4 第一阶段汇聚的内容信息之间的关系直接作为gcn的邻接关系
            # hs_hand_j_feature_inter_w_whole_map = torch.cat([hs_right_hand_context,hs_left_hand_context],dim=-2)[0:1]

            # hs_hand_j_feature_inter_selfsa =  self.j_feature_inter_selfsa(hs_hand_j_feature_inter_w_whole_map)


                                               
            #keypoints_coord_3d = self.lift_net(embeding_2d,hs_hand_j_feature_inter_selfsa)
            keypoints_coord_3d = self.lift_net(embeding_2d[0])

            keypoints_coord_3d = keypoints_coord_3d.squeeze().unsqueeze(0)*cfg.input_img_shape[0]
            if   mode == 'train':
                if (not cfg.pred_2d_depth) :
                    regression_loss_3d = self.detrloss(keypoints_coord_3d, targets['joint_coord'], meta_info['joint_valid'])
                else:
                    regression_loss_3d = self.detrloss(keypoints_coord_3d.unsqueeze(-1), targets['joint_coord'], meta_info['joint_valid'])



        #############################vis heatmap#####################################################
        vis = False
        if vis:

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
                image_np =  ((inputs['img'][b]*255).permute(1,2,0)).cpu().numpy().astype(np.uint8)
                joints_vaild = meta_info['joint_valid'][b]

                pred_2d_sample = torch.cat((keypoints_coord[:-1,b,:,:2],keypoints_coord_3d[:,b,:,:2]),dim=0)
                gt_2d_sample = targets['joint_coord'][b,:,:2]
                for j in range(42):
                    if not joints_vaild[j]:
                        continue

                    image_np_list = []
                    pred_2d_list = []
                    gt_2d_list=[]
                    att_mask1_c_list = []
                    att_mask1_p_list = []
                    for idx,scale in enumerate(all_scale_p):
                        # if scale <64:
                        #     continue
                        att_mask1 = all_scale_c[scale][b][j].reshape(1,scale,scale).cpu().numpy()
                        att_mask2 = all_scale_p[scale][b][j].reshape(1,scale,scale).cpu().numpy()
                        att_mask3 = all_scale_all[scale][b][j].reshape(1,scale,scale).cpu().numpy()

                # att_mask2 = att_weights[i][21:22].reshape(1,128,128).cpu().numpy()
                # att_mask3 = att_weights[i][22:38].reshape(16,128,128).cpu().numpy()
                # att_mask4 = att_weights[i][38:].reshape(21,128,128).cpu().numpy()
                
                #img = imgs[i].permute(1,2,0).cpu().numpy()
                        # save_img_path1 = os.path.join("./vis", "c_{}_{}att_mask_joint.png".format(scale,j))
                        # save_img_path2 = os.path.join("./vis", "p_{}_{}att_mask_joint.png".format(scale,j))
                        # save_img_path3 = os.path.join("./vis", "all_{}_{}_att_mask_joint.png".format(scale,j))
                        # xxd
                        save_img_path1 = os.path.join("./vis/appearance", "c_{}_{}_{}att_mask_joint.png".format(b,scale,j))
                        save_img_path2 = os.path.join("./vis/position", "p_{}_{}_{}att_mask_joint.png".format(b,scale,j))
                        # save_img_path3 = os.path.join("./vis/all", "all_{}_{}_{}_att_mask_joint.png".format(b,scale,j))
                        # xxd

                        ##用来画2d 关节点的gt和pred
                        pred_2d = pred_2d_sample[idx,j].cpu().numpy().copy()
                        gt_2d = gt_2d_sample[j].cpu().numpy().copy()

                        # visulize_attention_ratio(image_np, att_mask1, save_img_path1, ratio=0.5)
                        # visulize_attention_ratio(image_np, att_mask2, save_img_path2, ratio=0.5)
                        # visulize_attention_ratio(image_np, att_mask3, save_img_path3, ratio=0.5)
                        # xxd
                        # if scale == 64:
                        visulize_attention_ratio(image_np.copy(), pred_2d, gt_2d, att_mask1, save_img_path1, ratio=0.5)
                        visulize_attention_ratio(image_np.copy(), pred_2d, gt_2d, att_mask2, save_img_path2, ratio=0.5)
                        # visulize_attention_ratio(image_np.copy(), pred_2d, gt_2d, att_mask3, save_img_path3, ratio=0.5)

                        image_np_list.append(image_np.copy())
                        if idx != 2:
                            pred_2d_list.append(pred_2d)
                            gt_2d_list.append(gt_2d)
                            att_mask1_c_list.append(att_mask1)
                        if idx !=3:
                            att_mask1_p_list.append(att_mask2)                        
                        # xxd
                    # save_img_compare = os.path.join("./vis/compare", "compare_{}_{}_att_mask_joint.png".format(b,j))
                    # compare_visulize_attention_ratio(image_np_list,pred_2d_list,gt_2d_list,att_mask1_c_list,att_mask1_p_list,save_img_compare)

                    pass

            
            #dense_loss_sum  = self.dense_loss(pred_dense_color,gt_dense_color_ds_128,meta_info['segm_valid'].bool())
                
            #weak cam params
            #右手：3，2（平移）+ s 左手：3
            # pred_weak_cam_param = self.cam_param_regressor(hs_weak_cam_param[-1].squeeze()) 
            # left_hand_cam_param_gt,left_cam_valid = restore_weak_cam_param(targets['joint_cam'][:,21:],targets['joint_coord'][:,21:,:2],meta_info['joint_valid'][:,21:].bool())
            # right_hand_cam_param_gt,right_cam_valid = restore_weak_cam_param(targets['joint_cam'][:,:21],targets['joint_coord'][:,:21,:2],meta_info['joint_valid'][:,:21].bool())
            # left_cam_loss = torch.abs((left_hand_cam_param_gt.squeeze() - pred_weak_cam_param[:,3:])*left_cam_valid.squeeze(0))/(left_cam_valid.sum()+1)
            # right_cam_loss = torch.abs((right_hand_cam_param_gt.squeeze() - pred_weak_cam_param[:,:3])*right_cam_valid.squeeze(0))/(right_cam_valid.sum()+1)
            # cam_loss = left_cam_loss + right_cam_loss


            #使用weak cam反投影
            # right_keypoints_coord_xy_cam = (keypoints_coord_3d[...,:21,:2].squeeze() - pred_weak_cam_param[:,1:3].unsqueeze(-2) ) / (pred_weak_cam_param[:,None,:1] + 1e-7)
            # left_keypoints_coord_xy_cam = (keypoints_coord_3d[...,21:,:2].squeeze() - pred_weak_cam_param[:,4:].unsqueeze(-2) ) / (pred_weak_cam_param[:,None,3:4]  + 1e-7)
            # right_keypoints_coord_xy_cam = (targets['joint_coord'][:,:21,:2] - right_hand_cam_param_gt.squeeze()[:,None,1:]) / (right_hand_cam_param_gt.squeeze()[:,None,0:1] + 1e-7)


            # xy_cam = torch.cat((right_keypoints_coord_xy_cam,left_keypoints_coord_xy_cam),dim=-2)
            #pred root align
            # xy_cam[:,:21] = xy_cam[:,:21] - xy_cam[:,20,None,:]
            # xy_cam[:,21:] = xy_cam[:,21:] - xy_cam[:,41,None,:]

            # xy_cam_loss  = self.detrloss_L1(xy_cam, targets['joint_cam'][...,:2], meta_info['joint_valid']) #相机坐标下，x,y
            # z_cam = ((keypoints_coord_3d[...,-1:]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)).squeeze(0)
            # keypoints_coord_3d_cam = torch.cat((xy_cam,z_cam),dim=-1)  #depth需要反转回来

            # ##option 7 3d joints + image feature adaption 多平行线
            # mask= get_context_bone_mask_from_feature_use_initial_pred_two_hand_mutiscale(keypoints_coord_3d[-1,...,:2])
            # keypoint_mask = keypoints_mask_from_bone_info(mask)
            # adaption_pos_embeding = self.crossmodal_Transformer(refine_embeding,features_list_with_vis_info__hidden_dim_256,pos_feature_list,keypoint_mask)
            # refine_3d = self.hand_keypoint_3d(adaption_pos_embeding).squeeze().unsqueeze(0)*cfg.input_img_shape[0]
            # refine_regression_loss = self.detrloss(refine_3d, targets['joint_coord'], meta_info['joint_valid'])


            # # ##option 8 监督context
            # hs_hand_context_from_feature =  get_context_bone_embeding_from_feature_use_initial_pred_sample_point(targets['joint_coord'][...,:2],features_list_with_vis_info__hidden_dim_256)
            # context_loss = 0
            # for i in range(4):
            #     context_loss_layer = self.context_loss(tgt_content_list[i].unsqueeze(0),self.context_norm_list[i](hs_hand_context_from_feature[i:i+1]), meta_info['joint_valid'])
            #     context_loss+=context_loss_layer

        # keypoint = self.obj_keypoints(hs_object).sigmoid()
        # keypoint = keypoint.permute(1,0,2,3)
        # preds_obj['keypoint'] = keypoint
        if mode == 'train':
            if cfg.use_diffusion:
                if mode == 'train':
                    t = torch.randint(0, self.num_steps, [bs]).to(device)
                else:
                    # for validation
                    t = (torch.ones(bs) * meta_info["set_t"]).long().to(device)
                    
                current_alpha = (
                    self.alpha_torch[t].to(hs_left_hand.device).unsqueeze(-1)
                )  # (B,1,1,1)       
                
                #joint_coord 2.5D 
                data_2p5D = targets['joint_coord'].transpose(1,2).unsqueeze(-1)/w # (B,3,J,L) L=1 [0,1]
                noise = torch.randn_like(data_2p5D).to(device)  # (B,3,J,L)
                
                noisy_data = (current_alpha**0.5) * data_2p5D + (
                1.0 - current_alpha) ** 0.5 * noise  # (B,3,J,L)
                
                data_2d = torch.cat((hs_left_hand[-1],hs_right_hand[-1]),dim=-2) #(B,J,C) 
                cam = None
                total_input = self.apply_conditioning(
                noisy_data,
                data_2d,
                cam ,
                data_2p5D,
                p_uncond=self.p_uncond,
                    )
                predicted = self.diffmodel(total_input, t)  # (B,3,J,L)
                residual = noise - predicted
                diff_loss = (residual**2).mean()



            loss = {}
            if not cfg.use_diffusion or cfg.guide_condition_feature_w_joints_loss : 
                loss['Reg_loss'] = regression_loss
                loss['A2Jloss'] =  regression_loss
                total_loss = loss['A2Jloss']

            if cfg.use_diffusion:
                loss['Diffloss'] = diff_loss
                if cfg.guide_condition_feature_w_joints_loss:
                    total_loss= diff_loss + total_loss

            if cfg.use_lift_net:
                loss['Reg_loss_3d'] = regression_loss_3d
                total_loss = total_loss + loss['Reg_loss_3d'] * cfg.RegLossFactor 

                # loss['refine_regression_loss'] = refine_regression_loss
                # total_loss = total_loss + refine_regression_loss*3

                # loss['context_loss'] = context_loss*20
                # total_loss = total_loss + context_loss*20
                # loss['dense_loss_sum']=dense_loss_sum*100
                # total_loss = total_loss + dense_loss_sum*100

                # ##
                # loss['cam_loss'] = cam_loss
                # total_loss = total_loss + cam_loss

                # loss['xy_cam_loss'] = xy_cam_loss *3
                # total_loss = total_loss + xy_cam_loss*3

            if cfg.use_seg_loss:
                pred_segm_dict = segms_dict
                #gt_segms_dict_label = targets['img_segm_256_label']
                gt_img_segm_128 = targets['img_segm_128']
                segm_valid = meta_info['segm_valid'].bool()
                
                pred_segm_dict_valid = pred_segm_dict[segm_valid]
                gt_segm_dict_valid = gt_img_segm_128[segm_valid]
                
                if gt_segm_dict_valid.shape[0] == 0  :
                # if True: 
                    loss_dice_loss = loss_soft_ce_loss = loss_focal_loss = torch.zeros_like(pred_segm_dict,requires_grad=True).mean()

                else:
                    loss_dice_loss = self.loss1(pred_segm_dict_valid,gt_segm_dict_valid)*cfg.dice_loss_weight
                    #loss_soft_ce_loss = loss2(pred_segm_dict_valid,gt_segm_dict_valid)*cfg.soft_ce_loss_weight
                    loss_focal_loss = self.loss3(pred_segm_dict_valid,gt_segm_dict_valid)*cfg.focal_loss_weight

                loss["loss_dice_loss"] = loss_dice_loss
                total_loss += loss_dice_loss

                loss["loss_focal_loss"] = loss_focal_loss
                total_loss += loss_focal_loss

            loss['total_loss'] = total_loss

            return loss

        elif mode == 'test':
            ## use the result of last layer as the final 
            if cfg.use_diffusion:
                assert self.strategy in ["best", "average", "reproj", "worst", "all"], (
                    f"Invalid strategy: {self.strategy}. Possible values are 'best', 'average'"
                    "'worst', 'reproj', or 'all'"
                )
                if self.strategy == "all":
                    m_p3d_h36 = {"average": 0, "best": 0, "worst": 0, "reproj": 0}
                else:
                    m_p3d_h36 = {self.strategy: 0}
                #data_2p5D = targets['joint_coord'].transpose(1,2).unsqueeze(-1) # (B,3,J,L) L=1
                data_2d = torch.cat((hs_left_hand[-1],hs_right_hand[-1]),dim=-2) #(B,J,C) 

                cam = None
                data_3d = None #gt
                starting_pose = None
                samples = self.impute(
                                data_2d,
                                self.n_samples,
                                cam=cam,
                                data_3d=data_3d,
                                starting_pose=starting_pose,
                                n_steps=None,
                            )
                if self.strategy == 'average':
                    pred_pose = torch.mean(samples, axis=1).squeeze().transpose()*w ##[0,1] -> [0,w]
                #TODO
                else:
                    pass

            if cfg.use_lift_net:
                if not cfg.pred_2d_depth:
                    # pred_keypoints = refine_3d[0]
                    pred_keypoints = keypoints_coord_3d[0]
                    # pred_keypoints = keypoints_coord_3d_cam

                else:
                    pred_keypoints = torch.cat([keypoints_coord[0],keypoints_coord_3d[0].unsqueeze(-1)],dim=-1)
            else:
                pred_keypoints = keypoints_coord[-1]
            out = {}
            out['joint_coord'] = pred_keypoints
            if cfg.use_diffusion:
                out['diff_joint_coord'] = pred_pose

            # if 'inv_trans' in meta_info:
            #     out['inv_trans'] = meta_info['inv_trans']
            # if 'joint_coord' in targets:
            #     out['target_joint'] = targets['joint_coord']
            # if 'joint_valid' in meta_info:
            #     out['joint_valid'] = meta_info['joint_valid']
            # if 'hand_type_valid' in meta_info:
            #     out['hand_type_valid'] = meta_info['hand_type_valid']
            return out
    








def get_model(mode,joint_num):    
    args = cfg
    backbone, position_embedding = build_backbone(args)
    global_interactive_localtransformer = bulid_all_global_transformer_two_hand_multi_scale_sa_variant_layer_norm_diff_dim_cp(args,return_content_embeding = True,return_heatmap = True)
    # mano_layer = nn.ModuleDict(
    #         {
    #             'r':smplx.create(args.smplx_path, 'mano', use_pca=args.mano_use_pca, is_rhand=True),
    #             'l':smplx.create(args.smplx_path, 'mano', use_pca=args.mano_use_pca, is_rhand=False)
    #         }
    #     )
    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    # if torch.sum(torch.abs(mano_layer['l'].shapedirs[:,0,:] -  mano_layer['r'].shapedirs[:,0,:])) < 1:
    #         print('Fix shapedirs bug of MANO')
    #         mano_layer['l'].shapedirs[:,0,:] *= -1
    # # for quick T pose67
    # for ht in ('r','l'):
    #         beta2J = (mano_layer[ht].J_regressor[:,:,None,None] * mano_layer[ht].shapedirs.unsqueeze(0)).sum(dim=1)
    #         mano_layer[ht].register_buffer('beta2J', beta2J)
    #         template_J = (mano_layer[ht].J_regressor[:,:,None] * mano_layer[ht].v_template.unsqueeze(0)).sum(dim=1)
    #         mano_layer[ht].register_buffer('template_J', template_J)

    model = DETR(backbone = backbone,
        position_embedding = position_embedding,        
        #hand_segm = segm_net.SegmNet(dense_color=True),   
        hand_segm = segm_net.SegmNet(dense_color=False),
        global_interactive_localtransformer = global_interactive_localtransformer,
        #num_queries=args.num_queries,
        #mano_layer = mano_layer,
        args = args
    )
     ## Statistical Model Size
    print('BackboneNet No. of Params = %d M'%(sum(p.numel() for p in backbone.parameters() if p.requires_grad)/1e6))
    print('Transformer No. of Params = %d M'%(sum(p.numel() for p in global_interactive_localtransformer.parameters() if p.requires_grad)/1e6))
    print('Total No. of Params = %d M' % (sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))
    return model
