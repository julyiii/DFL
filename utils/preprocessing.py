# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import cv2
import numpy as np
from config import cfg
import random
import math
import copy
import torch


def load_img(path, order='RGB'):
    
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def load_segms(path):
    label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return label

def load_skeleton(path, joint_num):

    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id
    
    return skeleton

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

def augmentation(img, dense, segm, bbox, joint_coord, joint_valid, hand_type, mode, joint_type,joint_cam):
    img = img.copy(); 
    joint_coord = joint_coord.copy(); 
    hand_type = hand_type.copy();
    joint_cam = joint_cam.copy();

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
    
    #add segm
    if segm is not None:
        box_segm = copy.deepcopy(bbox)
        segm, _, _ = generate_patch_segm(segm, box_segm, do_flip, scale, rot, cfg.input_img_shape)
        segm = segm.astype(np.uint8)
    if dense is not None:
        box_segm = copy.deepcopy(bbox)
        dense, _, _ = generate_patch_segm(dense, box_segm, do_flip, scale, rot, cfg.input_img_shape)
        dense = dense.astype(np.uint8)
    
    if do_flip:
        joint_coord[:,0] = original_img_shape[1] - joint_coord[:,0] - 1
        joint_coord[joint_type['right']], joint_coord[joint_type['left']] = joint_coord[joint_type['left']].copy(), joint_coord[joint_type['right']].copy()
        joint_valid[joint_type['right']], joint_valid[joint_type['left']] = joint_valid[joint_type['left']].copy(), joint_valid[joint_type['right']].copy()
        hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
        ##add 3d点
        joint_cam[:,0] = -joint_cam[:,0]  #直接将3D的x轴转为负数
        joint_cam[joint_type['right']], joint_cam[joint_type['left']] = joint_cam[joint_type['left']].copy(), joint_cam[joint_type['right']].copy()

    for i in range(joint_num):
        joint_coord[i,:2] = trans_point2d(joint_coord[i,:2], trans)

    #fix bugs 在同时翻转2d坐标以及joints valid之后判断是否在图像平面内  only for train 因为只有在训练时才会旋转
    joint_valid = joint_valid * (joint_coord[:,0] >= 0) * (joint_coord[:,0] < cfg.input_img_shape[1]) * (joint_coord[:,1] >= 0) * (joint_coord[:,1] < cfg.input_img_shape[0])
    
    #add rotate 3d joints 第一种数据增强的方式是3d空间的关节点直接像2d空间一样进行旋转，另一种是想dir一样，先2.5D增强，然后反投影
    rot_rad = np.pi * rot / 180
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    global_rotation = np.array([[cs, sn],[-sn, cs]])
    joint_cam[:,:2] = joint_cam[:,:2] @ global_rotation.T
    return img, dense,segm,joint_coord, joint_valid, hand_type, inv_trans, do_flip, joint_cam


def transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, root_joint_idx, joint_type):
    # transform to output heatmap space
    joint_coord = joint_coord.copy(); joint_valid = joint_valid.copy()
    
    joint_coord[:,0] = joint_coord[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    joint_coord[:,1] = joint_coord[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

    joint_coord[joint_type['right'],2] = joint_coord[joint_type['right'],2] - joint_coord[root_joint_idx['right'],2]
    joint_coord[joint_type['left'],2] = joint_coord[joint_type['left'],2] - joint_coord[root_joint_idx['left'],2]
  
    joint_coord[:,2] = (joint_coord[:,2] / (cfg.bbox_3d_size/2) + 1)/2. * cfg.output_hm_shape[0] 

    joint_valid = joint_valid * ((joint_coord[:,2] >= 0) * (joint_coord[:,2] < cfg.output_hm_shape[0])).astype(np.float32)
    joint_valid = joint_valid * ((joint_coord[:,0] >= 0) * (joint_coord[:,0] < cfg.output_hm_shape[1])).astype(np.float32)
    joint_valid = joint_valid * ((joint_coord[:,1] >= 0) * (joint_coord[:,1] < cfg.output_hm_shape[2])).astype(np.float32)

    rel_root_depth = (rel_root_depth / (cfg.bbox_3d_size_root/2) + 1)/2. * cfg.output_root_hm_shape
    root_valid = root_valid * ((rel_root_depth >= 0) * (rel_root_depth < cfg.output_root_hm_shape)).astype(np.float32)    
    
    return joint_coord, joint_valid, rel_root_depth, root_valid


def get_bbox(joint_img, joint_valid):
    x_img = joint_img[:,0][joint_valid==1]; y_img = joint_img[:,1][joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax-xmin, ymax-ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, original_img_shape):

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox

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

def generate_patch_segm(cvimg, bbox, do_flip, scale, rot, out_shape):
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
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_NEAREST)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans 

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

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

def swap_lr_labels_segm_target_channels(segm_target,img_path):
    """
    Flip left and right label (not the width) of a single segmentation image.
    """
    assert isinstance(segm_target, torch.Tensor)
    assert len(segm_target.shape) == 3

    assert segm_target.min() >= 0
    
    #assert segm_target.max() <= 32
    if segm_target.max() > 32:
        print(segm_target.max())
        print("here",segm_target)
        print('img_path',img_path)
    img_segm = segm_target.clone()
    right_idx = ((1 <= img_segm)*(img_segm <= 16)).nonzero(as_tuple=True)
    left_idx = ((17 <= img_segm)*(img_segm <= 32)).nonzero(as_tuple=True)
    img_segm[right_idx[0], right_idx[1], right_idx[2]] += 16
    img_segm[left_idx[0], left_idx[1], left_idx[2]] -= 16
    img_segm_swapped = img_segm.clone()
    img_segm_swapped[1], img_segm_swapped[2] = img_segm_swapped[2].clone(), img_segm_swapped[1].clone()
    return img_segm_swapped
