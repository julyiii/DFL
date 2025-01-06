import os
import os.path as osp
import sys
import math
import numpy as np
import datetime

def clean_file(path):
    ## Clear the files under the path
    for i in os.listdir(path): 
        content_path = os.path.join(path, i) 
        if os.path.isdir(content_path):
            clean_file(content_path)
        else:
            assert os.path.isfile(content_path) is True
            os.remove(content_path)



class Config:
        
    # ~~~~~~~~~~~~~~~~~~~~~~Dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    dataset = 'InterHand2.6M'  # InterHand2.6M  nyu hands2017
    pose_representation = '2p5D' #2p5D


    # ~~~~~~~~~~~~~~~~~~~~~~ paths~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ## Please set your path
    ## Interhand2.6M dataset path. you should change to your dataset path.
    interhand_anno_dir = '/data_ssd/data/huanyao/datasets/InterHand2.6M_5fps_batch1/annotations'
    interhand_images_path = '/data_ssd/data/huanyao/datasets/InterHand2.6M_5fps_batch1/images'
    ## current file dir. change this path to your A2J-Transformer folder dir.
    cur_dir = '/data_ssd/huanyao/yaohuan/a2-j_based-detr'
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~input, output~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #modified
    input_img_shape = (256,256)
    # input_img_shape = (544,544)
    output_hm_shape = (256, 256, 256) # (depth, height, width)
    output_hm_shape_all = 256  ## For convenient
    sigma = 2.5
    bbox_3d_size = 400 # depth axis
    bbox_3d_size_root = 400 # depth axis 
    output_root_hm_shape = 64 # depth axis 
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~detr config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    RegLossFactor = 3
    Refine_regression_loss_3d_Factor = 3
    Relibility_LossFactor = 0


    # ~~~~~~~~~~~~~~~~~~~~~~~~backbone config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #num_feature_levels = 4
    lr_backbone = 1e-4
    masks = False
    #modified
    #backbone = 'resnet50' 
    backbone = 'fpn' 
    
    #modified
    dilation = True # If true, we replace stride with dilation in the last convolutional block (DC5)
    if dataset == 'InterHand2.6M':
        keypoint_num = 42
    elif dataset == 'nyu':
        keypoint_num = 14
    elif dataset == 'hands2017':
        keypoint_num = 21


    # ~~~~~~~~~~~~~~~~~~~~~~~~transformer config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    position_embedding = 'sine' #'sine' #'convLearned' # learned
    hidden_dim = 256
    dropout = 0.1
    nheads = 8
    dim_feedforward = 1024
    #modified
    enc_layers = 2
    dec_layers = 4
    pre_norm = False
    ##modified
    soft_sa_scale = 10
    sa_pre = True
    soft_sa_method = 'multiply' #multiply|add
    use_pos_embeding_in_ca = True
    use_identify_embeding_in_sa  = False
    pred_2d_depth = False
    use_layer_learnable_embeding = False
    aggregate_context_in_sa = True

 
    # num_feature_levels = 4
    # dec_n_points = 4
    # enc_n_points = 4
    #num_queries = 768  ## query numbers, default is 256*3 = 768 
    # kernel_size = 256
    # two_stage = False  ## Whether to use the two-stage deformable-detr, please select False.
    # use_dab = True  ## Whether to use dab-detr, please select True.
    # num_patterns = 0
    # anchor_refpoints_xy = True  ##  Whether to use the anchor anchor point as the reference point coordinate, True.
    # is_3D = True  # True 
    # fix_anchor = True  ## Whether to fix the position of reference points to prevent update, True.
    # use_lvl_weights = False  ## Whether to assign different weights to the loss of each layer, the improvement is relatively limited.
    # lvl_weights = [0.1, 0.15, 0.15, 0.15, 0.15, 0.3]
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~a2j config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # RegLossFactor = 3

    # ~~~~~~~~~~~~~~~~~~~~~~~~lift layer config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    use_lift_net = True

    lift_channel = 256#+32#256*2 + 32
    lift_layers = 3
    
    refine_channel = 512//2
    use_origin_gcn   = True


    
    # ~~~~~~~~~~~~~~~~~~~~~~~~diffusion~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    use_diffusion = False
    schedule = 'cosine'
    beta_start = 0.0001
    beta_end = 0.5
    num_steps = 50
    p_uncond = 0
    encode_3d = True
    cond_mix_mode = 'concat'
    condition_out_dim = 64
    arch = 'gcn'
    channels = 64
    use_nonlocal = True
    diffusion_embedding_dim = 128
    p_dropout = 0.2703806090969092
    layers = 16
    guide_condition_feature_w_joints_loss = True
    n_samples = 5
    sample_strategy ='average'
    
    #################segms######################################################
    #modified
    use_seg_loss = True
    dice_loss_weight = 1
    focal_loss_weight = 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~training config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    lr_dec_epoch = [24, 35] if dataset == 'InterHand2.6M' else [45,47] 
    end_epoch = 100 if dataset == 'InterHand2.6M' else 50 
    lr = 1e-4
    lr_dec_factor = 5  
    train_batch_size = 16

    continue_train = False  ## Whether to continue training, default is False
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~testing config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # test_batch_size = 2#48 #原先64
    # test_batch_size = 28#48 #原先64
    test_batch_size = 4

    trans_test = 'gt' ## 'gt', 'rootnet' # 'rootnet' is not used
    #modified
    test_start_epoch = 38
    test_per_epoch = 2
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~dataset config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    use_single_hand_dataset = False ## Use single-handed data, default is True
    use_inter_hand_dataset = True ## Using interacting hand data, default is True
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~others~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    num_thread = 8
    gpu_ids = '0'   ## your gpu ids, for example, '0', '1-3'
    num_gpus = 1
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~directory setup~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    data_dir = osp.join(cur_dir, 'data')
    output_dir = osp.join(cur_dir, 'output')
    datalistDir = osp.join(cur_dir, 'datalist_gt_filter') ## this is used to save the dataset datalist, easy to debug.
    vis_2d_dir = osp.join(output_dir, 'vis_2d')
    # vis_2d_dir = '/data_ssd/huanyao/yaohuan/a2-j_based-detr/vis_demo'
    # vis_2d_dir = osp.join(output_dir, 'vis_2d_HIC_46')
    vis_3d_dir = osp.join(output_dir, 'vis_3d')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    model_dir = osp.join(output_dir, 'model_dump')
    tensorboard_dir = osp.join(output_dir, 'tensorboard_log')
    clean_tensorboard_dir = False 
    clean_log_dir = False
    if clean_tensorboard_dir is True:
        clean_file(tensorboard_dir)
    if clean_log_dir is True:
        clean_file(log_dir)


    def set_args(self, gpu_ids,exp_name,continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        output_dir = osp.join(self.cur_dir,'output',exp_name)
        self.vis_2d_dir = osp.join(output_dir, 'vis_2d')
        self.vis_3d_dir = osp.join(output_dir, 'vis_3d')
        self.log_dir = osp.join(output_dir, 'log')
        self.result_dir = osp.join(output_dir, 'result')
        self.model_dir = osp.join(output_dir, 'model_dump')
        self.tensorboard_dir = osp.join(output_dir, 'tensorboard_log')
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
        make_folder(cfg.datalistDir)
        make_folder(cfg.model_dir)
        make_folder(cfg.vis_2d_dir)
        make_folder(cfg.vis_3d_dir)
        make_folder(cfg.log_dir)
        make_folder(cfg.result_dir)
        make_folder(cfg.tensorboard_dir)
        
    def save_args(self, save_folder = None, opt_prefix="opt", verbose=True):
        if  save_folder is None:
            save_folder = self.model_dir
        opts = vars(self)
        cls_opts = vars(self.__class__)
        for key in cls_opts.keys():
            if key  not in opts:
                opts[key] = cls_opts[key]
            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        opt_filename = "{}.txt".format(opt_prefix)
        opt_path = os.path.join(save_folder, opt_filename)
        with open(opt_path, "a") as opt_file:
            opt_file.write("====== Options ======\n")
            for k, v in sorted(opts.items()):
                opt_file.write("{option}: {value}\n".format(option=str(k), value=str(v)))
            opt_file.write("=====================\n")
            opt_file.write("launched {} at {}\n".format(str(sys.argv[0]), str(datetime.datetime.now())))
        if verbose:
            print("Saved options to {}".format(opt_path))


cfg = Config()
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
