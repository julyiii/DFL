import torch
import numpy as np
from torch.nn import functional as F
from plyfile import PlyData, PlyElement
import smplx
import random
import cv2
from config import cfg

import os
import os.path as osp
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from config import cfg
from PIL import Image, ImageDraw
import torch
import logging
from math import ceil
from PIL import Image
def get_keypoint_rgb(skeleton):
    rgb_dict= {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        
    return rgb_dict

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint
def cam2pixel(cam_coord, f, c):
    x=[]
    y=[]
    z=[]
    for i in range(len(cam_coord)):
        if cam_coord[i,2]!=0:
            x.append(cam_coord[i,0] / cam_coord[i,2] * f[0] + c[0])
            y.append(cam_coord[i,1] / cam_coord[i,2] * f[1] + c[1])
            z.append(cam_coord[i,2])
        else:
            x.append(cam_coord[i,0])
            y.append(cam_coord[i,1])
            z.append(cam_coord[i,2])
    x=np.array(x)
    y=np.array(y)
    z=np.array(z)
    return np.stack((x,y,z),1)


def get_bbox(joint_img, joint_valid, extend_ratio=1.2):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    
    return bbox

def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        bbox = None

    return bbox

def process_bbox(bbox, do_sanitize=True, extend_ratio=1.25):
    # if do_sanitize:
    #     bbox = sanitize_bbox(bbox, img_width, img_height)
    #     if bbox is None:
    #         return bbox

   # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = 1
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*extend_ratio
    bbox[3] = h*extend_ratio
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox


def load_ply(file_name):
    plydata = PlyData.read(file_name)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    v = np.stack((x,y,z),1)
    return v


class MANO(object):
    def __init__(self):
        self.layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False, 'create_transl': False}
        self.layer = {'right': smplx.create("/data_ssd/huanyao/yaohuan/new/kypt_transformer/mano/MANO_RIGHT.pkl", 'mano', is_rhand=True, use_pca=False, flat_hand_mean=False, **self.layer_arg), 'left': smplx.create("/data_ssd/huanyao/yaohuan/new/kypt_transformer/mano/MANO_LEFT.pkl", 'mano', is_rhand=False, use_pca=False, flat_hand_mean=False, **self.layer_arg)}
        self.vertex_num = 778
        self.face = {'right': self.layer['right'].faces, 'left': self.layer['left'].faces}
        self.shape_param_dim = 10
        
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(self.layer['left'].shapedirs[:,0,:] - self.layer['right'].shapedirs[:,0,:])) < 1:
            print('Fix shapedirs bug of MANO')
            self.layer['left'].shapedirs[:,0,:] *= -1

        # original MANO joint set
        self.orig_joint_num = 16
        self.orig_joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3')
        self.orig_root_joint_idx = self.orig_joints_name.index('Wrist')
        self.orig_flip_pairs = ()
        self.orig_joint_regressor = self.layer['right'].J_regressor.numpy() # same for the right and left hands

        # changed MANO joint set (single hands)
        self.sh_joint_num = 21 # manually added fingertips
        self.sh_joints_name = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
        self.sh_skeleton = ( (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20) )
        self.sh_root_joint_idx = self.sh_joints_name.index('Wrist')
        self.sh_flip_pairs = ()
        # add fingertips to joint_regressor
        self.sh_joint_regressor = transform_joint_to_other_db(self.orig_joint_regressor, self.orig_joints_name, self.sh_joints_name)
        self.sh_joint_regressor[self.sh_joints_name.index('Thumb_4')] = np.array([1 if i == 745 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.sh_joint_regressor[self.sh_joints_name.index('Index_4')] = np.array([1 if i == 317 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.sh_joint_regressor[self.sh_joints_name.index('Middle_4')] = np.array([1 if i == 445 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.sh_joint_regressor[self.sh_joints_name.index('Ring_4')] = np.array([1 if i == 556 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.sh_joint_regressor[self.sh_joints_name.index('Pinky_4')] = np.array([1 if i == 673 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)


        # changed MANO joint set (two hands)
        self.th_joint_num = 42 # manually added fingertips. two hands
        self.th_joints_name = ('R_Wrist', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', 'L_Wrist', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4')
        self.th_root_joint_idx = {'right': self.th_joints_name.index('R_Wrist'), 'left': self.th_joints_name.index('L_Wrist')}
        self.th_flip_pairs = [(i,i+21) for i in range(21)]
        self.th_joint_type = {'right': np.arange(0,self.th_joint_num//2), 'left': np.arange(self.th_joint_num//2,self.th_joint_num)}

mano = MANO()



def augmentation(img, bbox, joint_coord, joint_valid, hand_type, mode, joint_type):
    img = img.copy(); 
    joint_coord = joint_coord.copy(); 
    hand_type = hand_type.copy();

    original_img_shape = img.shape
    joint_num = len(joint_coord)
    
    if mode == 'train':
        trans, scale, rot, do_flip, color_scale = get_aug_config()

    else:
        trans, scale, rot, do_flip, color_scale = [0,0], 1.0, 0.0, False, np.array([1,1,1])

    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[3] * trans[1]
    img, trans, inv_trans = generate_patch_image(img, bbox, do_flip, scale, rot, cfg.input_img_shape)
    img = np.clip(img * color_scale[None,None,:], 0, 255) 
    
    if do_flip:
        joint_coord[:,0] = original_img_shape[1] - joint_coord[:,0] - 1
        joint_coord[joint_type['right']], joint_coord[joint_type['left']] = joint_coord[joint_type['left']].copy(), joint_coord[joint_type['right']].copy()
        joint_valid[joint_type['right']], joint_valid[joint_type['left']] = joint_valid[joint_type['left']].copy(), joint_valid[joint_type['right']].copy()
        hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
    for i in range(joint_num):
        joint_coord[i,:2] = trans_point2d(joint_coord[i,:2], trans)
 
    return img, joint_coord, joint_valid, hand_type, inv_trans

def get_aug_config():
    trans_factor = 0.15
    scale_factor = 0.25
    rot_factor = 45
    color_factor = 0.2
    
    trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

    return trans, scale, rot, do_flip, color_scale


def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


# #################################origin#####################################
# def vis_keypoints(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None,bbox = None):
    
#     rgb_dict = get_keypoint_rgb(skeleton)
#     _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
#     draw = ImageDraw.Draw(_img)
#     for i in range(len(skeleton)):
#         joint_name = skeleton[i]['name']
#         pid = skeleton[i]['parent_id']
#         parent_joint_name = skeleton[pid]['name']
        
#         kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
#         kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

#         if score[i] > score_thr and score[pid] > score_thr and pid != -1:
#             draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
#         if score[i] > score_thr:
#             draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
#         if score[pid] > score_thr and pid != -1:
#             draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
    
#     if bbox is not None:
#         draw.line([(bbox[0], bbox[1]), (bbox[0], bbox[1] + bbox[3])], fill = (153, 204, 255), width = 1)
#         draw.line([(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1])], fill = (153, 204, 255), width = 1)
#         draw.line([(bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2], bbox[1]+ bbox[3])], fill = (153, 204, 255), width = 1)
#         draw.line([(bbox[0], bbox[1] + bbox[3]), (bbox[0] + bbox[2], bbox[1]+ bbox[3])], fill = (153, 204, 255), width = 1)
#     if save_path is None:
#         _img.save(osp.join(cfg.vis_2d_dir, filename))
#     else:
#         _img.save(osp.join(save_path, filename))
#################################origin#####################################

#################################origin#####################################
#################################xxd#####################################
def vis_keypoints(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None,bbox = None):
    
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
        if score[i] > score_thr:
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])

    
    if bbox is not None:
        draw.line([(bbox[0], bbox[1]), (bbox[0], bbox[1] + bbox[3])], fill = (153, 204, 255), width = 1)
        draw.line([(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1])], fill = (153, 204, 255), width = 1)
        draw.line([(bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2], bbox[1]+ bbox[3])], fill = (153, 204, 255), width = 1)
        draw.line([(bbox[0], bbox[1] + bbox[3]), (bbox[0] + bbox[2], bbox[1]+ bbox[3])], fill = (153, 204, 255), width = 1)
    if save_path is None:
        _img.save(osp.join(cfg.vis_2d_dir, filename))
    else:
        _img.save(osp.join(save_path, filename))
#################################xxd#####################################

def vis_keypoints_single_color(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None,bbox = None):
    color =(255, 10, 215)
    _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=color, width=line_width)
        if score[i] > score_thr:
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=color)
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=color)
    
    if bbox is not None:
        draw.line([(bbox[0], bbox[1]), (bbox[0], bbox[1] + bbox[3])], fill = (153, 204, 255), width = 1)
        draw.line([(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1])], fill = (153, 204, 255), width = 1)
        draw.line([(bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2], bbox[1]+ bbox[3])], fill = (153, 204, 255), width = 1)
        draw.line([(bbox[0], bbox[1] + bbox[3]), (bbox[0] + bbox[2], bbox[1]+ bbox[3])], fill = (153, 204, 255), width = 1)
    if save_path is None:
        _img.save(osp.join(cfg.vis_2d_dir, filename))
    else:
        _img.save(osp.join(save_path, filename))