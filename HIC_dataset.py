# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import os.path as osp
from pickle import FALSE
import numpy as np
import torch
import cv2
import json
import copy
import math
import random
from tqdm import tqdm
from glob import glob
from pycocotools.coco import COCO
# from config_xxd import cfg
from config import cfg

# from utils.preprocessing_HIC import load_img, get_bbox, process_bbox, augmentation, get_iou, load_ply
# from utils.preprocessing_HIC import load_img, get_bbox, process_bbox, augmentation, load_ply
from utils.preprocessing import load_skeleton, transform_input_to_output_space, trans_point2d,load_img
# from utils.preprocessing import load_skeleton, transform_input_to_output_space, trans_point2d
# from utils.vis_HIC import vis_keypoints, save_obj
# xxd
from utils.hic_utils import cam2pixel,get_bbox,load_ply,mano,process_bbox,augmentation
from utils.transforms import pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints,vis_3d_keypoints_compare
import matplotlib.pyplot as plt

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        assert data_split == 'test', 'only testing is supported for HIC dataset'
        # self.data_path = osp.join('..', 'data', 'HIC', 'data')
        self.data_path = "/data_ssd/data/huanyao/datasets/HIC"
        self.focal = (525.0, 525.0)
        self.princpt = (319.5, 239.5)

        # HIC joint set
        # self.joint_set = {
        #                 'joint_num': 28, 
        #                 'joints_name': ('R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Thumb_2', 'R_Thumb_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Index_3', 'L_Index_2', 'L_Index_1', 'L_Thumb_2', 'L_Thumb_3'),
        #                 'flip_pairs': [ (i,i+14) for i in range(14)]
        #                 }
        # self.joint_set['joint_type'] = {'right': np.arange(0,self.joint_set['joint_num']//2), 'left': np.arange(self.joint_set['joint_num']//2,self.joint_set['joint_num'])}
        self.joint_num = 21
        self.root_joint_idx = {'right': 20, 'left': 41} 

        self.datalist = self.load_data()
        
        # xxd
        # self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.joint_type = {'left': np.arange(self.joint_num,self.joint_num*2),'right': np.arange(0,self.joint_num), }
        self.use_single_hand_dataset = cfg.use_single_hand_dataset
        self.use_inter_hand_dataset = cfg.use_inter_hand_dataset
        self.skeleton_path = '/data_ssd/data/huanyao/datasets/InterHand2.6M_5fps_batch1/annotations'
        self.skeleton = load_skeleton(osp.join(self.skeleton_path, 'skeleton.txt'), self.joint_num*2)
        # self.joint_valid = [1.0] * self.joint_num*2
        # self.joint_valid = np.array(self.joint_valid)
        # self.joint_valid[self.joint_type['right']] *= self.joint_valid[self.root_joint_idx['right']]
        # self.joint_valid[self.joint_type['left']] *= self.joint_valid[self.root_joint_idx['left']]

    def load_data(self):
        # load annotation
        db = COCO(osp.join(self.data_path, 'HIC.json'))

        datalist = []
        # xxd
        n=0

        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.data_path, img['file_name'])
            hand_type = ann['hand_type']
            joint_valid = ann['joint_valid']
            #self.joint_type = {'left': np.arange(0,self.joint_num), 'right': np.arange(self.joint_num,self.joint_num*2)}
            joint_valid = np.ones(42)
            # for i in range(self.joint_num*2):
            #     joint_valid.append(1.0)
            # joint_valid = np.array(joint_valid)
            # joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            # joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            # joint_valid_left = joint_valid
            # xxd
            if hand_type == 'right':
                joint_valid[self.joint_num:]=0
            elif hand_type == 'left':
                joint_valid[:self.joint_num]=0

            # # bbox
            # body_bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
            # body_bbox = process_bbox(body_bbox, img_width, img_height, extend_ratio=1.0)
            # if body_bbox is None:
            #     continue

            # mano mesh
            if ann['right_mano_path'] is not None:
                right_mano_path = osp.join(self.data_path, ann['right_mano_path'])
            else:
                right_mano_path = None
            if ann['left_mano_path'] is not None:
                left_mano_path = osp.join(self.data_path, ann['left_mano_path'])
            else:
                left_mano_path = None

            # xxd joint
            # xxd mano coordinates
            if right_mano_path is not None:
                rhand_mesh = load_ply(right_mano_path)
            else:
                rhand_mesh = np.zeros((mano.vertex_num, 3), dtype=np.float32)
            left_mano_path = left_mano_path
            if left_mano_path is not None:
                lhand_mesh = load_ply(left_mano_path)
            else:
                lhand_mesh = np.zeros((mano.vertex_num, 3), dtype=np.float32)
            mano_mesh_cam = np.concatenate((rhand_mesh, lhand_mesh))
            mesh_gt = mano_mesh_cam * 1000
            # xxd joint_cam
            joint_lh_cam = np.dot(mano.sh_joint_regressor, mesh_gt[mano.vertex_num:,:])
            joint_rh_cam = np.dot(mano.sh_joint_regressor, mesh_gt[:mano.vertex_num,:])
            # joint_cam = joint_lh_cam + joint_rh_cam
            # joint_cam = np.dot(mano.sh_joint_regressor, mesh_gt[:,:])
            joint_cam = np.concatenate((joint_rh_cam,joint_lh_cam), 0)
            # single_hand_valid_reorder = [20,7,6,5,11,10,9,19,18,17,15,14,13,3,2,1,0,4,8,12,16,41,28,27,26,32,31,30,40,39,38,36,35,34,24,23,22,21,25,29,33,37]
            # single_hand_valid_reorder = [20,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16,41,24,23,22,21,28,27,26,25,32,31,30,29,36,35,34,33,40,39,38,37]
            # joint_cam = joint_cam.copy()[single_hand_valid_reorder]
            # single_hand_valid_reorder = [20,7,6,5,11,10,9,19,18,17,15,14,13,3,2,1,0,4,8,12,16]
            # single_hand_valid_reorder = [20,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16]
            single_hand_valid_reorder = [4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0]
            double_hand_valid_reorder =single_hand_valid_reorder + [i+self.joint_num for i in single_hand_valid_reorder]
            joint_cam = joint_cam.copy()[double_hand_valid_reorder]
            # left_hand_valid_reorder = [20,7,6,5,11,10,9,19,18,17,15,14,13,3,2,1,0,4,8,12,16]
            # right_hand_valid_reorder = [i+self.joint_num for i in left_hand_valid_reorder]
            # joint_lh_cam = joint_lh_cam.copy()[single_hand_valid_reorder]
            # joint_rh_cam = joint_rh_cam.copy()[single_hand_valid_reorder]
            # joint_cam = np.concatenate((joint_lh_cam, joint_rh_cam), 0)

            # xxd img+camera_para
            focal = self.focal
            princpt = self.princpt
            joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]
            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}

            # xxd bbox
            for idx, h in enumerate(('right', 'left')):
                if ann[h + '_mano_path']  is None:
                    continue
                # bbox_out = out[h[0] + 'hand_bbox'] # xyxy in cfg.input_body_shape space
                if h == 'right':
                    # bbox_gt = self.get_bbox_from_mesh(mesh_gt[:mano.vertex_num,:]) # xywh in original image space
                    bbox_gt = self.get_bbox_from_mesh(mesh_gt[:mano.vertex_num,:],h) # xywh in original image space
                else:
                    # bbox_gt = self.get_bbox_from_mesh(mesh_gt[mano.vertex_num:,:]) # xywh in original image space
                    # bbox_gt = self.get_bbox_from_mesh(mesh_gt[:mano.vertex_num,:], n) # xywh in original image space
                    bbox_gt = self.get_bbox_from_mesh(mesh_gt[:,:],h) # xywh in original image space
                # bbox_gt[2:] += bbox_gt[:2] # xywh -> xyxy
            
            abs_depth = {'right': joint_cam[self.root_joint_idx['right'],2], 'left': joint_cam[self.root_joint_idx['left'],2]} #根节点的深度值，以此为参考

            # datalist.append({
            #     'aid': aid,
            #     'img_path': img_path,
            #     'img_shape': (img_height, img_width),
            #     'body_bbox': body_bbox,
            #     'hand_type': hand_type,
            #     'right_mano_path': right_mano_path,
            #     'left_mano_path': left_mano_path})
            datalist.append({
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                # 'body_bbox': body_bbox,
                'bbox': bbox_gt,
                'hand_type': hand_type,
                'right_mano_path': right_mano_path,
                'left_mano_path': left_mano_path, 
                'joint': joint,
                'abs_depth': abs_depth,
                'cam_param': cam_param})
            
            # xxd
            # if n%50==0:
            #     print('{}_'.format(n)+hand_type+'_gt_joint_left:\n{}'.format(joint_cam[:self.joint_num]))
            #     print('{}_'.format(n)+hand_type+'_gt_joint_right:\n{}'.format(joint_cam[self.joint_num:]))
            n+=1   
        return datalist
    

    def get_bbox_from_mesh(self, mesh,h):
        x = mesh[:,0] / mesh[:,2] * self.focal[0] + self.princpt[0]
        y = mesh[:,1] / mesh[:,2] * self.focal[1] + self.princpt[1]
        xy = np.stack((x,y),1)
        bbox = get_bbox(xy, np.ones_like(x),extend_ratio = 1.)
        bbox = process_bbox(bbox,extend_ratio=1.25)



        # xxd
        # if n%50==0:
        #     print("\nbbox:{}\n".format(bbox))
        #     print("x_max:{}\n".format(max(x)))
        #     print("x_min:{}\n".format(min(x)))
        #     print("y_max:{}\n".format(max(y)))
        #     print("y_min:{}\n".format(min(y)))
        #     print("\nbbox_gt[2:]:{}\n".format(bbox[2:]))
        #     print("\nbbox_gt[:2]:{}\n".format(bbox[:2]))
        return bbox
    
    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    def __len__(self):
        return len(self.datalist)
        # return 64

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, joint, hand_type = data['img_path'], data['img_shape'], data['bbox'], data['joint'], data['hand_type']

        # xxd a2j
        joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); 
        joint_valid = joint['valid'].copy();
        # xxd joint_valid is float
        # joint_valid = joint['valid']
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None].copy()),1)
        # seq_name = data['seq_name']
        contact_vis_np = np.zeros((32, 2)).astype(np.float32)

        # img
        img = load_img(img_path)

        # img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, body_bbox, self.data_split)
        # img = self.transform(img.astype(np.float32))/255.

        # xxd augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, self.data_split, self.joint_type)
        # img, img2bb_trans, inv_trans, rot, do_flip = augmentation(img, body_bbox, self.data_split)
        rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],dtype=np.float32).reshape(1)
        root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]])*1.0

        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth, root_valid =\
            transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type)
        # joint_coord, joint_valid, rel_root_depth =\
        #     transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, self.root_joint_idx, self.joint_type)

        # Some images are blank, filter for training
        if np.sum(img) < 1e-4 :
            joint_valid *= 0
            root_valid *= 0
            # hand_type_valid *= 0
            contact_vis_np *= 0

        img = self.transform(img.astype(np.float32)) / 255.

        # xxd use zero mask.
        mask = np.zeros((img.shape[1], img.shape[2])).astype(np.bool)
        mask = self.transform(mask.astype(np.uint8))

        # # mano coordinates
        # right_mano_path = data['right_mano_path']
        # if right_mano_path is not None:
        #     rhand_mesh = load_ply(right_mano_path)
        # else:
        #     rhand_mesh = np.zeros((mano.vertex_num, 3), dtype=np.float32)
        # left_mano_path = data['left_mano_path']
        # if left_mano_path is not None:
        #     lhand_mesh = load_ply(left_mano_path)
        # else:
        #     lhand_mesh = np.zeros((mano.vertex_num, 3), dtype=np.float32)
        # mano_mesh_cam = np.concatenate((rhand_mesh, lhand_mesh))
        # xxd
        # mesh_gt = mano_mesh_cam * 1000

        # joint_lh = np.dot(mano.sh_joint_regressor, mesh_gt[mano.vertex_num:,:])
        # joint_rh = np.dot(mano.sh_joint_regressor, mesh_gt[:mano.vertex_num,:])
        # joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}

        # # inputs = {'img': img}
        # inputs = {'img': img, 'mask': mask}
        # targets = {'mano_mesh_cam': mano_mesh_cam}
        # meta_info = {'bb2img_trans': bb2img_trans}
        bbox_hand_left = np.array([0.0,0.0,cfg.input_img_shape[0],cfg.input_img_shape[1]],dtype=np.float32)
        bbox_hand_right = np.array([0.0,0.0,cfg.input_img_shape[0],cfg.input_img_shape[1]],dtype=np.float32)
        # xxd
        inputs = {'img': img, 'mask': mask}
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'inv_trans': inv_trans,'bbox_hand_left':bbox_hand_left,"bbox_hand_right":bbox_hand_right}

        return inputs, targets, meta_info

    # xxd
    def evaluate(self, preds):
        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, inv_trans, joint_valid_used = preds['joint_coord'], preds['inv_trans'], preds['joint_valid']
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)
        
        mpjpe_sh = [[] for _ in range(self.joint_num)]
        mpjpe_ih = [[] for _ in range(self.joint_num*2)]
        mpjpe_sh_2d = [[] for _ in range(self.joint_num)]
        mpjpe_sh_3d = [[] for _ in range(self.joint_num)]
        mpjpe_ih_2d = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih_3d = [[] for _ in range(self.joint_num*2)]
        tot_err = []
        mpjpe_dict = {}


        mrrpe = []
        acc_hand_cls = 0; hand_cls_cnt = 0;
        for n in tqdm(range(sample_num),ncols=150):
            vis = False
            mpjpe_per_data_list = []
            mpjpe_per_data = 0

            data = gts[n]
            # bbox, cam_param, joint, gt_hand_type, hand_type_valid = data['bbox'], data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
            bbox, cam_param, joint, gt_hand_type = data['bbox'], data['cam_param'], data['joint'], data['hand_type']
            hand_type = data['hand_type']

            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord'] 
            gt_joint_coord_root = {"right":gt_joint_coord[self.root_joint_idx['right']],"left":gt_joint_coord[self.root_joint_idx['left']]}
            gt_joint_img = joint['img_coord']
            
            ## use original joint_valid param.
            joint_valid = joint['valid']
            # joint_valid = joint_valid_used[n]

            # restore xy coordinates to original image space

            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
            pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
            for j in range(self.joint_num*2):
                pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])

            # restore depth to original camera space
            pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

            # add root joint depth
            pred_joint_coord_img[self.joint_type['right'],2] += data['abs_depth']['right']
            pred_joint_coord_img[self.joint_type['left'],2] += data['abs_depth']['left']

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

            # root joint alignment
            for h in ('right', 'left'):
                pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h],None,:]
                gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h],None,:]
            
            # mpjpe
            ## xyz mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]: ## 在这里，限制了只加载valid的坐标值
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                        mpjpe_per_data_list.append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                        # continue
                    else:
                        mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                        mpjpe_per_data_list.append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                    print(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                
                # else:
                #      if gt_hand_type == 'right':
                #         mpjpe_sh[:self.joint_num].append(0)
                #      elif gt_hand_type == 'left':
                #          mpjpe_sh[self.joint_num:].append(0)

            # xxd
            # if n%50==0:
            #     print("\nmpjpe_xyz_{}".format(n)+"_{}".format(gt_hand_type)+":{}\n".format(mpjpe_sh if gt_hand_type == 'right' or gt_hand_type == 'left' else mpjpe_ih))
            ## xy mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,:2] - gt_joint_coord[j,:2])**2)))
                        # continue
                    else:
                        mpjpe_ih_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,:2] - gt_joint_coord[j,:2])**2)))
            # xxd
            # if n%50==0:
            #     print("\nmpjpe_xy_{}".format(n)+"_{}".format(gt_hand_type)+":{}\n".format(mpjpe_sh_2d if gt_hand_type == 'right' or gt_hand_type == 'left' else mpjpe_ih_2d))

            ## depth mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh_3d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,2] - gt_joint_coord[j,2])**2)))
                        # continue
                    else:
                        mpjpe_ih_3d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,2] - gt_joint_coord[j,2])**2)))
            # xxd
            # if n%50==0:
            #     print("\nmpjpe_depth_{}".format(n)+"_{}".format(gt_hand_type)+":{}\n".format(mpjpe_sh_3d if gt_hand_type == 'right' or gt_hand_type == 'left' else mpjpe_ih_3d))
            #########################################origin################################
            # vis_2d = False
            vis_2d = True  
            #if n%50 == 0 or gt_hand_type=="left":
            if vis_2d:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                vis_kps = pred_joint_coord_img.copy()
                vis_kps_gt = gt_joint_img.copy()
                vis_valid = joint_valid.copy()
                # capture = str(data['capture'])
                # cam = str(data['cam'])
                # frame = str(data['frame'])
                filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
                # vis_keypoints(_img, vis_kps, vis_kps_gt, bbox, vis_valid, self.skeleton, filename)
                vis_keypoints(_img, vis_kps, bbox, vis_valid, self.skeleton, filename)
                # vis_keypoints(_img, vis_kps)
                # vis_keypoints(_img, vis_kps_gt)
                # print('\nimg_'+str(n)+'_'+gt_hand_type+'_kps:\n{}\n'.format(vis_kps))
                # print('\nimg_'+str(n)+'_'+gt_hand_type+'_kps_gt:\n{}\n'.format(vis_kps_gt))
                print('vis 2d over')
            #########################################origin################################

            #########################################xxd################################
            # vis_2d = True   
            # if vis_2d:
            #     img_path = data['img_path']
            #     cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            #     _img = cvimg[:,:,::-1].transpose(2,0,1)
            #     vis_kps = pred_joint_coord_img.copy()
            #     vis_kps_gt = gt_joint_img.copy()
            #     # vis_valid = joint_valid.copy()
            #     vis_valid = np.ones(self.joint_num*2).copy()
            #     # capture = str(data['capture'])
            #     # cam = str(data['cam'])
            #     # frame = str(data['frame'])
            #     # filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
            #     filename = str(n) + '_' + gt_hand_type + '.jpg'
            #     # if n ==60:
            #     #     print("\nvis_kps_60:{}\n".format(vis_kps))
            #     #     print("\nvis_kps_gt_60:{}\n".format(vis_kps_gt))
            #     vis_keypoints(_img, vis_kps, bbox, vis_valid, self.skeleton, filename)
            #     print('vis 2d over')
            #########################################xxd################################
            
            vis_3d = False
            # vis_3d = True
            # if n%50 == 0:
            #     if vis_3d:
            #         filename = 'out_' + str(n) + '_3d.jpg'
            #         vis_3d_cam = pred_joint_coord_cam.copy()
            #         vis_3d_cam_left = pred_joint_coord_cam[self.joint_type['left']].copy()
            #         vis_3d_cam_left[:,2] = pred_joint_coord_cam[self.joint_type['left'],2]
            #         vis_3d_cam_right = pred_joint_coord_cam[self.joint_type['right']].copy()
            #         vis_3d_cam_right[:,2] = pred_joint_coord_cam[self.joint_type['right'],2] 
            #         vis_3d = np.concatenate((vis_3d_cam_left, vis_3d_cam_right), axis= 0)
            #         vis_3d_keypoints(vis_3d, joint_valid, self.skeleton, filename)
            #         print('vis 3d over')

            compare_3d = False
            if compare_3d:
                single_hand_valid_reorder = [20,7,6,5,11,10,9,19,18,17,15,14,13,3,2,1,0,4,8,12,16]  #interhand joints顺序 -> mano 需要在确认一下指尖的validdouble_hand_valid_reorder = single_hand_valid_reorder + [i+self.joint_num  for i in single_hand_valid_reorder]
                double_hand_valid_reorder = single_hand_valid_reorder + [i+self.joint_num  for i in single_hand_valid_reorder]
                filename = 'out_' + str(n) + '_compare3d.jpg'
                #align to 3d gt root
                pred_joint_coord_cam_vis= np.concatenate([pred_joint_coord_cam[:self.joint_num]+gt_joint_coord_root['right'],pred_joint_coord_cam[self.joint_num:]+gt_joint_coord_root['left']],axis=0)
                gt_joint_coord_cam_vis= np.concatenate([gt_joint_coord[:self.joint_num]+gt_joint_coord_root['right'],gt_joint_coord[self.joint_num:]+gt_joint_coord_root['left']],axis=0)

                vis_3d_keypoints_compare([pred_joint_coord_cam_vis,gt_joint_coord_cam_vis], joint_valid, self.skeleton, filename)
                # print('vis 3d over')
    
        
        if hand_cls_cnt > 0: 
            handness_accuracy = acc_hand_cls / hand_cls_cnt
            print('Handedness accuracy: ' + str(handness_accuracy))
        if len(mrrpe) > 0: 
            mrrpe_num = sum(mrrpe)/len(mrrpe)
            print('MRRPE: ' + str(mrrpe_num))
        print()



        if self.use_inter_hand_dataset is True and self.use_single_hand_dataset is True:
            print('..................MPJPE FOR TOTAL HAND..................')
            eval_summary = 'MPJPE for each joint: \n'
            for j in range(self.joint_num*2):
                if j<21:
                    tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
                else:
                    tot_err_j = np.mean(np.stack(mpjpe_ih[j]))

                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
                # eval_summary += (': %.2f, ' %tot_err_j)
                tot_err.append(tot_err_j)
            print(eval_summary)
            tot_err_mean = np.mean(tot_err)
            print('MPJPE for all hand sequences: %.2f' % (tot_err_mean))
            mpjpe_dict['total'] = tot_err_mean
            print()

        if self.use_single_hand_dataset is True:
            print('..................MPJPE FOR SINGLE HAND..................')
            ## xyz
            eval_summary = 'MPJPE for each joint: \n'
            for j in range(self.joint_num):
                mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
                # eval_summary += (': %.2f, ' %mpjpe_sh[j])
            print(eval_summary)
            mpjpe_sh_mean = np.mean(mpjpe_sh)
            print('MPJPE for single hand sequences: %.2f' % (mpjpe_sh_mean))
            mpjpe_dict['single_hand_total'] = mpjpe_sh_mean
            print()

            ## xy
            eval_summary_2d = 'MPJPE for each joint 2d: \n'
            for j in range(self.joint_num):
                mpjpe_sh_2d[j] = np.mean(np.stack(mpjpe_sh_2d[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary_2d += (joint_name + ': %.2f, ' % mpjpe_sh_2d[j])
                # eval_summary_2d += (': %.2f, ' %mpjpe_sh_2d[j])
            print(eval_summary_2d) 
            mpjpe_sh_2d_mean = np.mean(mpjpe_sh_2d)
            print('MPJPE for single hand sequences 2d: %.2f' % (mpjpe_sh_2d_mean))
            mpjpe_dict['single_hand_2d'] = mpjpe_sh_2d_mean
            print()

            ## z
            eval_summary_3d = 'MPJPE for each joint depth: \n'
            for j in range(self.joint_num):
                mpjpe_sh_3d[j] = np.mean(np.stack(mpjpe_sh_3d[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary_3d += (joint_name + ': %.2f, ' % mpjpe_sh_3d[j])
                # eval_summary_3d += (': %.2f, ' %mpjpe_sh_3d[j])
            print(eval_summary_3d) 
            mpjpe_sh_3d_mean = np.mean(mpjpe_sh_3d)
            print('MPJPE for single hand sequences 3d: %.2f' % (mpjpe_sh_3d_mean))
            mpjpe_dict['single_hand_depth'] = mpjpe_sh_3d_mean
            print()


        if self.use_inter_hand_dataset is True:
            print('..................MPJPE FOR INTER HAND..................')
            ## xyz
            eval_summary = 'MPJPE for each joint: \n'
            for j in range(self.joint_num*2):
                mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
                # eval_summary += (': %.2f, ' %mpjpe_ih[j])
            print(eval_summary) 
            mpjpe_ih_mean = np.mean(mpjpe_ih)
            print('MPJPE for interacting hand sequences: %.2f' % (mpjpe_ih_mean))
            mpjpe_dict['inter_hand_total'] = mpjpe_ih_mean
            print()

            ## xy
            eval_summary_2d = 'MPJPE for each joint 2d: \n'
            for j in range(self.joint_num*2):
                mpjpe_ih_2d[j] = np.mean(np.stack(mpjpe_ih_2d[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary_2d += (joint_name + ': %.2f, ' % mpjpe_ih_2d[j])
                # eval_summary_2d += (': %.2f, ' %mpjpe_ih_2d[j])
            print(eval_summary_2d) 
            mpjpe_ih_2d_mean = np.mean(mpjpe_ih_2d)
            print('MPJPE for interacting hand sequences 2d: %.2f' % (mpjpe_ih_2d_mean))
            mpjpe_dict['inter_hand_2d'] = mpjpe_ih_2d_mean
            print()

            ## z
            eval_summary_3d = 'MPJPE for each joint depth: \n'
            for j in range(self.joint_num*2):
                mpjpe_ih_3d[j] = np.mean(np.stack(mpjpe_ih_3d[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary_3d += (joint_name + ': %.2f, ' % mpjpe_ih_3d[j])
                # eval_summary_3d += (': %.2f, ' %mpjpe_ih_3d[j])
            print(eval_summary_3d) 
            mpjpe_ih_3d_mean = np.mean(mpjpe_ih_3d)
            print('MPJPE for interacting hand sequences 3d: %.2f' % (mpjpe_ih_3d_mean))
            mpjpe_dict['inter_hand_depth'] = mpjpe_ih_3d_mean
            print()


        if hand_cls_cnt > 0 and len(mrrpe) > 0:
            return mpjpe_dict, handness_accuracy, mrrpe_num
        else:
            return mpjpe_dict, None, None
    
    # def print_eval_result(self, eval_result):
    #     tot_eval_result = {
    #             'mpvpe_sh': [],
    #             'mpvpe_ih': [],
    #             'rrve': [],
    #             'mrrpe': [],
    #             'bbox_iou': []
    #             }
        
    #     # mpvpe (average all samples)
    #     for mpvpe_sh in eval_result['mpvpe_sh']:
    #         if mpvpe_sh is not None:
    #             tot_eval_result['mpvpe_sh'].append(mpvpe_sh)
    #     for mpvpe_ih in eval_result['mpvpe_ih']:
    #         if mpvpe_ih is not None:
    #             tot_eval_result['mpvpe_ih'].append(mpvpe_ih)
    #     for mpvpe_ih in eval_result['rrve']:
    #         if mpvpe_ih is not None:
    #             tot_eval_result['rrve'].append(mpvpe_ih)
       
    #     # mrrpe (average all samples)
    #     for mrrpe in eval_result['mrrpe']:
    #         if mrrpe is not None:
    #             tot_eval_result['mrrpe'].append(mrrpe)
 
    #     # bbox IoU
    #     for iou in eval_result['bbox_iou']:
    #         if iou is not None:
    #             tot_eval_result['bbox_iou'].append(iou)
        
    #     # print evaluation results
    #     eval_result = tot_eval_result
        
    #     print('bbox IoU: %.2f' % (np.mean(eval_result['bbox_iou']) * 100))
    #     print('MRRPE: %.2f mm' % (np.mean(eval_result['mrrpe'])))
    #     print('MPVPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'] + eval_result['mpvpe_ih'])))
    #     print('MPVPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'])))
    #     print('MPVPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_ih'])))
    #     print('RRVE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['rrve'])))

