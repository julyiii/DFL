# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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

import matplotlib.ticker as ticker

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


# def vis_keypoints(img, kps, kps_gt, bbox, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None):
    
#     rgb_dict = get_keypoint_rgb(skeleton)
#     _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
#     draw = ImageDraw.Draw(_img)
#     # for i in range(len(skeleton)):
#     for i in range(len(kps_gt)):
#         joint_name = skeleton[i]['name']
#         pid = skeleton[i]['parent_id']
#         parent_joint_name = skeleton[pid]['name']
        
#         kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
#         kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

#         if score[i] > score_thr and score[pid] > score_thr and pid != -1:
#             # draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
#             draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=(255, 10, 215), width=line_width)
#             draw.line([(kps_gt[i][0], kps_gt[i][1]), (kps_gt[pid][0], kps_gt[pid][1])], fill=(173, 230, 216), width=line_width)

#         if score[i] > score_thr:
#             # draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
#             draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=(255, 10, 215))
#             draw.ellipse((kps_gt[i][0]-circle_rad, kps_gt[i][1]-circle_rad, kps_gt[i][0]+circle_rad, kps_gt[i][1]+circle_rad), fill=(173, 230, 216))

#         if score[pid] > score_thr and pid != -1:
#             # draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
#             draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=(255, 10, 215))
#             draw.ellipse((kps_gt[pid][0]-circle_rad, kps_gt[pid][1]-circle_rad, kps_gt[pid][0]+circle_rad, kps_gt[pid][1]+circle_rad), fill=(173, 230, 216))
    
#     draw.line([(bbox[0], bbox[1]), (bbox[0], bbox[1] + bbox[3])], fill = (153, 204, 255), width = 1)
#     draw.line([(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1])], fill = (153, 204, 255), width = 1)
#     draw.line([(bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2], bbox[1]+ bbox[3])], fill = (153, 204, 255), width = 1)
#     draw.line([(bbox[0], bbox[1] + bbox[3]), (bbox[0] + bbox[2], bbox[1]+ bbox[3])], fill = (153, 204, 255), width = 1)

    
#     if save_path is None:
#         _img.save(osp.join(cfg.vis_2d_dir, filename))
#     else:
#         _img.save(osp.join(save_path, filename))
    
#     plt.close()

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


def vis_3d_keypoints_compare(kps_3d_list, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb(skeleton)
    
    kps_3d = kps_3d_list[0]
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i,0], kps_3d[pid,0]])
        y = np.array([kps_3d[i,1], kps_3d[pid,1]])
        z = np.array([kps_3d[i,2], kps_3d[pid,2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')
    
    kps_3d = kps_3d_list[1]
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i,0], kps_3d[pid,0]])
        y = np.array([kps_3d[i,1], kps_3d[pid,1]])
        z = np.array([kps_3d[i,2], kps_3d[pid,2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c = np.array([0,0,0])/255., linewidth = line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array([0,0,0]).reshape(1,3)/255., marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(np.array([0,0,0])).reshape(1,3)/255., marker='o')
    #plt.show()
    #cv2.waitKey(0)
    fig.savefig(osp.join(cfg.vis_3d_dir, filename), dpi=fig.dpi)
    plt.close()


# def vis_3d_keypoints(kps_3d, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3):

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     rgb_dict = get_keypoint_rgb(skeleton)
    
#     for i in range(len(skeleton)):
#         joint_name = skeleton[i]['name']
#         pid = skeleton[i]['parent_id']
#         parent_joint_name = skeleton[pid]['name']

#         x = np.array([kps_3d[i,0], kps_3d[pid,0]])
#         y = np.array([kps_3d[i,1], kps_3d[pid,1]])
#         z = np.array([kps_3d[i,2], kps_3d[pid,2]])

#         if score[i] > score_thr and score[pid] > score_thr and pid != -1:
#             ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
#             # ax.plot(x, -y, z, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)

#         if score[i] > score_thr:
#             ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
#         if score[pid] > score_thr and pid != -1:
#             ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')

#     ax.set(xlim=[-200, 50], ylim=[-200, 50], zlim=[-100, 100])
#     ax.view_init(5, 20)
    
#     fig.savefig(osp.join(cfg.vis_3d_dir, filename), dpi=fig.dpi)



#     plt.close()

##################################################compare with a2j 2d pose and 3d pose####################################
#######################################origin###################################
# def vis_3d_keypoints(kps_3d, kps_3d_gt, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3):

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     rgb_dict = get_keypoint_rgb(skeleton)
    
#     for i in range(len(skeleton)):
#         joint_name = skeleton[i]['name']
#         pid = skeleton[i]['parent_id']
#         parent_joint_name = skeleton[pid]['name']

#         x = np.array([kps_3d[i,0], kps_3d[pid,0]])
#         y = np.array([kps_3d[i,1], kps_3d[pid,1]])
#         z = np.array([kps_3d[i,2], kps_3d[pid,2]])
#         x1 = np.array([kps_3d_gt[i,0], kps_3d_gt[pid,0]])
#         y1 = np.array([kps_3d_gt[i,1], kps_3d_gt[pid,1]])
#         z1 = np.array([kps_3d_gt[i,2], kps_3d_gt[pid,2]])

#         if score[i] > score_thr and score[pid] > score_thr and pid != -1:
#             ax.plot(x1, z1, -y1, c = "black", linewidth = line_width)
#             ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
#             # ax.plot(x, -y, z, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)

#         if score[i] > score_thr:
#             ax.scatter(kps_3d_gt[i,0], kps_3d_gt[i,2], -kps_3d_gt[i,1], c = "black", marker='^')
#             ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
#         if score[pid] > score_thr and pid != -1:
#             ax.scatter(kps_3d_gt[pid,0], kps_3d_gt[pid,2], -kps_3d_gt[pid,1], c = "black", marker='^')
#             ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')

#     ax.set(xlim=[-200, 50], ylim=[-200, 50], zlim=[-100, 100])
#     ax.view_init(5, 20)
    
#     fig.savefig(osp.join(cfg.vis_3d_dir, filename), dpi=fig.dpi)

#     plt.close()
#######################################origin###################################
#######################################xxd######################################
# def vis_3d_keypoints(kps_3d, kps_3d_gt, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3):
def vis_3d_keypoints(kps_3d, kps_3d_gt, score, skeleton, filename, root, hand_flag, score_thr=0.4, line_width=3, circle_rad=3):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb(skeleton)
    
    rot_rad = np.pi * 45 / 180
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    global_rotation = np.array([[cs, sn],[-sn, cs]])
    dim_0_2 = kps_3d[:,[0,2]]
    kps_3d[:,[0,2]] = dim_0_2@ global_rotation.T

    dim_0_2_gt = kps_3d_gt[:,[0,2]]
    kps_3d_gt[:,[0,2]] = dim_0_2_gt@ global_rotation.T

    kps_3d = np.multiply(2, kps_3d)
    kps_3d_gt = np.multiply(2, kps_3d_gt)

    if hand_flag == 'left':
        kps_3d = kps_3d + root
        kps_3d_gt = kps_3d_gt + root

    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i,0], kps_3d[pid,0]])
        y = np.array([kps_3d[i,1], kps_3d[pid,1]])
        z = np.array([kps_3d[i,2], kps_3d[pid,2]])
        x1 = np.array([kps_3d_gt[i,0], kps_3d_gt[pid,0]])
        y1 = np.array([kps_3d_gt[i,1], kps_3d_gt[pid,1]])
        z1 = np.array([kps_3d_gt[i,2], kps_3d_gt[pid,2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            #origin
            # ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
            # ax.plot(x1, z1, -y1, c = "black", linewidth = line_width)
            ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
            ax.plot(x1, z1, -y1, c = "black", linewidth = line_width)
            # ax.plot(x, -y, z, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)

        if score[i] > score_thr:
            # ax.scatter(kps_3d_gt[i,0], kps_3d_gt[i,2], -kps_3d_gt[i,1], c = "black", marker='^')
            # ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')、

            ax.scatter(kps_3d_gt[i,0], kps_3d_gt[i,2], -kps_3d_gt[i,1], c = "black", marker='^')
            ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
        if score[pid] > score_thr and pid != -1:
            # ax.scatter(kps_3d_gt[pid,0], kps_3d_gt[pid,2], -kps_3d_gt[pid,1], c = "black", marker='^')
            # ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')
            ax.scatter(kps_3d_gt[pid,0], kps_3d_gt[pid,2], -kps_3d_gt[pid,1], c = "black", marker='^')
            ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')
    # xxd
    # ax.grid(False)#默认True，风格线。
    # ax.set_xticks([])#不显示x坐标轴
    # ax.set_yticks([])#不显示y坐标轴
    # ax.set_zticks([])#不显示z坐标轴
    # plt.axis('off')#关闭所有坐标轴

    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))


    # ax.set(xlim=[-200, 100], ylim=[-200, 100], zlim=[-100, 100])
    # ax.set(xlim=[-350, 100], ylim=[-300, 150], zlim=[-150, 150]) # 2_3_2
    ax.set(xlim=[-300, 150], ylim=[-300, 150], zlim=[-100, 200]) # 2_3_3
    # ax.set(xlim=[-200, 250], ylim=[-300, 150], zlim=[-100, 200]) # 2_3_15
    # ax.set(xlim=[-350, 100], ylim=[-300, 150], zlim=[-150, 150]) # 4_4_6
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.zaxis.set_major_formatter(plt.NullFormatter())
    # ax.view_init(5, 20)
    # ax.view_init(5, 180)
    # ax.view_init(5, 270)
    # ax.view_init(5, 315)
    ax.view_init(5, 270)

    
    fig.savefig(osp.join(cfg.vis_3d_dir, filename), dpi=fig.dpi)

    plt.close()
#######################################xxd######################################

# def vis_keypoints(img, kps, kps_gt, bbox, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None):
def vis_keypoints(img, kps, bbox, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None):
    
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
    draw = ImageDraw.Draw(_img)
    # for i in range(len(skeleton)):
    # for i in range(len(kps_gt)):
    for i in range(len(kps)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            # draw.line([(kps_gt[i][0], kps_gt[i][1]), (kps_gt[pid][0], kps_gt[pid][1])], fill="black", width=line_width)
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
            # draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=(255, 10, 215), width=line_width)


        if score[i] > score_thr:
            # draw.ellipse((kps_gt[i][0]-circle_rad, kps_gt[i][1]-circle_rad, kps_gt[i][0]+circle_rad, kps_gt[i][1]+circle_rad), fill="black")
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
            # draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=(255, 10, 215))


        if score[pid] > score_thr and pid != -1:
            # draw.ellipse((kps_gt[pid][0]-circle_rad, kps_gt[pid][1]-circle_rad, kps_gt[pid][0]+circle_rad, kps_gt[pid][1]+circle_rad), fill="black")
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
            # draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=(255, 10, 215))
    
    # draw.line([(bbox[0], bbox[1]), (bbox[0], bbox[1] + bbox[3])], fill = (153, 204, 255), width = 1)
    # draw.line([(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1])], fill = (153, 204, 255), width = 1)
    # draw.line([(bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2], bbox[1]+ bbox[3])], fill = (153, 204, 255), width = 1)
    # draw.line([(bbox[0], bbox[1] + bbox[3]), (bbox[0] + bbox[2], bbox[1]+ bbox[3])], fill = (153, 204, 255), width = 1)

    
    if save_path is None:
        _img.save(osp.join(cfg.vis_2d_dir, filename))
    else:
        _img.save(osp.join(save_path, filename))

    plt.close()


##################################################compare with a2j 2d pose and 3d pose####################################

def visualize_feature_map(img_batch,out_path,type,BI):
    bs,c,h,w = img_batch.shape
    feature_map = torch.squeeze(img_batch)
    feature_map = feature_map.detach().cpu().numpy()

    feature_map_sum = feature_map[0, :, :]
    feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
    for i in range(0, c):
        feature_map_split = feature_map[i,:, :]
        feature_map_split = np.expand_dims(feature_map_split,axis=2)
        if i > 0:
            feature_map_sum +=feature_map_split
        #feature_map_split = BI.transform(feature_map_split)

        # plt.imshow(feature_map_split)
        # plt.savefig(out_path + str(i) + "_{}.jpg".format(type) )
        # plt.xticks()
        # plt.yticks()
        # plt.axis('off')
    feature_map_sum = feature_map_sum/feature_map_sum.max() * 255
    feature_map_sum = BI.transform(feature_map_sum)
    plt.imshow(feature_map_sum)
    plt.savefig(out_path + "sum_{}.jpg".format(type))
    print("save sum_{}.jpg".format(type))


class BilinearInterpolation(object):
    def __init__(self, w_rate: float, h_rate: float, *, align='center'):
        if align not in ['center', 'left']:
            logging.exception(f'{align} is not a valid align parameter')
            align = 'center'
        self.align = align
        self.w_rate = w_rate
        self.h_rate = h_rate

    def set_rate(self,w_rate: float, h_rate: float):
        self.w_rate = w_rate    # w 的缩放率
        self.h_rate = h_rate    # h 的缩放率

    # 由变换后的像素坐标得到原图像的坐标    针对高
    def get_src_h(self, dst_i,source_h,goal_h) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_i = float(dst_i * (source_h/goal_h))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_i = float((dst_i + 0.5) * (source_h/goal_h) - 0.5)
        src_i += 0.001
        src_i = max(0.0, src_i)
        src_i = min(float(source_h - 1), src_i)
        return src_i
    # 由变换后的像素坐标得到原图像的坐标    针对宽
    def get_src_w(self, dst_j,source_w,goal_w) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_j = float(dst_j * (source_w/goal_w))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_j = float((dst_j + 0.5) * (source_w/goal_w) - 0.5)
        src_j += 0.001
        src_j = max(0.0, src_j)
        src_j = min((source_w - 1), src_j)
        return src_j

    def transform(self, img):
        source_h, source_w, source_c = img.shape  # (235, 234, 3)
        goal_h, goal_w = round(
            source_h * self.h_rate), round(source_w * self.w_rate)
        new_img = np.zeros((goal_h, goal_w, source_c), dtype=np.uint8)

        for i in range(new_img.shape[0]):       # h
            src_i = self.get_src_h(i,source_h,goal_h)
            for j in range(new_img.shape[1]):
                src_j = self.get_src_w(j,source_w,goal_w)
                i2 = ceil(src_i)
                i1 = int(src_i)
                j2 = ceil(src_j)
                j1 = int(src_j)
                x2_x = j2 - src_j
                x_x1 = src_j - j1
                y2_y = i2 - src_i
                y_y1 = src_i - i1
                new_img[i, j] = img[i1, j1]*x2_x*y2_y + img[i1, j2] * \
                    x_x1*y2_y + img[i2, j1]*x2_x*y_y1 + img[i2, j2]*x_x1*y_y1
        return new_img

def tensor_to_PIL(tensor):
 image = tensor.cpu().clone()
 image = image.squeeze(0)
 if len(image.shape) == 3:
    a = image.numpy().astype(np.uint8) #e.float()
    image = Image.fromarray(a)
     
 else:
    a = image.numpy().astype(np.uint8) #e.float()
    image = Image.fromarray(a)
 return image



def visulize_attention_ratio(img, pred_2d, gt_2d, attention_mask, save_img_path, ratio=0.5,cmap="jet"):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:放大或缩小图片的比例，可选
    cmap: attention map的style,可选
    """
    #print("load image from: ", save_img_path)
    # # load the image
    # # img = Image.open(img_path, mode='r')
    # # img_h, img_w = img.size[0], img.size[1]

    # img_h,img_w = img.shape[0:2]
    # plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # # scale the image
    # img_h, img_w = int(img_h * ratio), int(img_w * ratio)
    
    # from PIL import Image
    # img = Image.fromarray(img).resize((img_h, img_w))
    # plt.imshow(img, alpha=1)
    # plt.axis('off')
    
    # from PIL import Image
    # img = Image.open('/data0/huanyao/code/a2-j_based-detr/image0599.jpg', mode='r')
    # # img_h img_w = img.size[0], img.size[1]

    # #img = Image.fromarray(img).resize((img_h, img_w))
    # img_h,img_w = (512,512)
    # img_h, img_w = int(img_h * ratio), int(img_w * ratio)

    # plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # # scale the image
    

    plt.imshow(img, alpha=1)
    plt.axis('off')



    # normalize the attention mask
    attention_mask = np.average(attention_mask, axis=0)
    # print(attention_mask.shape)
    # exit()
    if img is not None:
        img_h,img_w = img.shape[0],img.shape[1]  
    else:
        img_h,img_w = attention_mask.shape[-2],attention_mask.shape[-1]  
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, cmap=cmap) # interpolation='nearest',
    #plt.scatter(pred_2d[0], pred_2d[1], marker='x', color='w', s=80)
    plt.scatter(gt_2d[0], gt_2d[1], marker='+', color='yellow', s=80)
    plt.savefig(save_img_path, dpi=100)
    plt.close()

def compare_visulize_attention_ratio(image_np_list,pred_2d_list,gt_2d_list,att_mask1_c_list,att_mask1_p_list,save_img_compare,cmap="jet"):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:放大或缩小图片的比例，可选
    cmap: attention map的style,可选
    """
    #print("load image from: ", save_img_path)
    # # load the image
    # # img = Image.open(img_path, mode='r')
    # # img_h, img_w = img.size[0], img.size[1]

    # img_h,img_w = img.shape[0:2]
    # plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # # scale the image
    # img_h, img_w = int(img_h * ratio), int(img_w * ratio)
    
    # from PIL import Image
    # img = Image.fromarray(img).resize((img_h, img_w))
    # plt.imshow(img, alpha=1)
    # plt.axis('off')
    
    # from PIL import Image
    # img = Image.open('/data0/huanyao/code/a2-j_based-detr/image0599.jpg', mode='r')
    # # img_h img_w = img.size[0], img.size[1]

    # #img = Image.fromarray(img).resize((img_h, img_w))
    # img_h,img_w = (512,512)
    # img_h, img_w = int(img_h * ratio), int(img_w * ratio)

    # plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # # scale the image
    for i in range(3):
        #c
        plt.subplot(2, 3, i+1)
        plt.axis('off')

        img = image_np_list[i].copy()
        img_h,img_w = img.shape[0],img.shape[1]  
        plt.imshow(img, alpha=1)
        attention_mask = np.average(att_mask1_c_list[i], axis=0)
        mask = cv2.resize(attention_mask, (img_h, img_w))
        normed_mask = mask / mask.max()
        normed_mask = (normed_mask * 255).astype('uint8')
        plt.imshow(normed_mask, alpha=0.5, cmap=cmap) # interpolation='nearest',
        plt.scatter(pred_2d_list[i][0], pred_2d_list[i][1], marker='x', color='w', s=80)
        plt.scatter(gt_2d_list[i][0], gt_2d_list[i][1], marker='+', color='yellow', s=80,alpha=0.5)
        #p
        plt.subplot(2, 3, i+4)
        plt.axis('off')

        img = image_np_list[i].copy()
        img_h,img_w = img.shape[0],img.shape[1]  

        plt.imshow(img, alpha=1)
        attention_mask = np.average(att_mask1_p_list[i], axis=0)
        mask = cv2.resize(attention_mask, (img_h, img_w))
        normed_mask = mask / mask.max()
        normed_mask = (normed_mask * 255).astype('uint8')
        plt.imshow(normed_mask, alpha=0.5, cmap=cmap) # interpolation='nearest',
        plt.scatter(pred_2d_list[i][0], pred_2d_list[i][1], marker='x', color='w', s=80)
        plt.scatter(gt_2d_list[i][0], gt_2d_list[i][1], marker='+', color='yellow', s=80,alpha=0.5)

    plt.savefig(save_img_compare, dpi=100)
    plt.close()
# xxd
    #     #c
    # i = 2
    # plt.subplot(2, 1, 1)
    # plt.axis('off')

    # img = image_np_list[i].copy()
    # img_h,img_w = img.shape[0],img.shape[1]  
    # plt.imshow(img, alpha=1)
    # attention_mask = np.average(att_mask1_c_list[i], axis=0)
    # mask = cv2.resize(attention_mask, (img_h, img_w))
    # normed_mask = mask / mask.max()
    # normed_mask = (normed_mask * 255).astype('uint8')
    # plt.imshow(normed_mask, alpha=0.5, cmap=cmap) # interpolation='nearest',
    #     # plt.scatter(pred_2d_list[i][0], pred_2d_list[i][1], marker='x', color='w', s=80)
    # plt.scatter(gt_2d_list[i][0], gt_2d_list[i][1], marker='+', color='yellow', s=80,alpha=0.5)
    #     #p
    # plt.subplot(2, 1, 2)
    # plt.axis('off')

    # img = image_np_list[i].copy()
    # img_h,img_w = img.shape[0],img.shape[1]  

    # plt.imshow(img, alpha=1)
    # attention_mask = np.average(att_mask1_p_list[i], axis=0)
    # mask = cv2.resize(attention_mask, (img_h, img_w))
    # normed_mask = mask / mask.max()
    # normed_mask = (normed_mask * 255).astype('uint8')
    # plt.imshow(normed_mask, alpha=0.5, cmap=cmap) # interpolation='nearest',
    #     # plt.scatter(pred_2d_list[i][0], pred_2d_list[i][1], marker='x', color='w', s=80)
    # plt.scatter(gt_2d_list[i][0], gt_2d_list[i][1], marker='+', color='yellow', s=80,alpha=0.5)

    # plt.savefig(save_img_compare, dpi=100)
    # plt.close()
# xxd



def visulize_attention_ratio_old(img, attention_mask, save_img_path, ratio=0.5,cmap="jet"):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:放大或缩小图片的比例，可选
    cmap: attention map的style,可选
    """
    print("load image from: ", save_img_path)
    # load the image
    from PIL import Image
    img = Image.open('/data0/huanyao/code/a2-j_based-detr/image0599.jpg', mode='r')
    # img_h img_w = img.size[0], img.size[1]

    #img = Image.fromarray(img).resize((img_h, img_w))
    img_h,img_w = (512,512)
    img_h, img_w = int(img_h * ratio), int(img_w * ratio)

    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    

    plt.imshow(img, alpha=1)
    plt.axis('off')
    
    # normalize the attention mask
    attention_mask = np.average(attention_mask, axis=0)
    # print(attention_mask.shape)
    # exit()
    #img_h,img_w = attention_mask.shape[-2],attention_mask.shape[-1]  
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    plt.savefig(save_img_path, dpi=100)
    #plt.close()