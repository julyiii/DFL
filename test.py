# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from utils.vis import vis_keypoints
import torch.backends.cudnn as cudnn
from utils.transforms import flip
import os
import time

from calflops import calculate_flops

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6,7', dest='gpu_ids')
    parser.add_argument('--test_set', type=str, default='test', dest='test_set')
    parser.add_argument('--test_epoch', type=str, default='0', dest='test_epoch')
    parser.add_argument('--exp_name',type=str)

    
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def test():
    args = parse_args()
    cfg.set_args(args.gpu_ids,args.exp_name)
    cudnn.benchmark = True
    #print(os.path.join(cfg.log_dir,"test_log.txt"))

    if cfg.dataset == 'InterHand2.6M':
        assert args.test_set, 'Test set is required. Select one of test/val'
    else:
        args.test_set = 'test'

    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set)
    tester._make_model()
    
    preds = {'joint_coord': [], 'inv_trans': [], 'joint_valid': [] }

    timer = []
    

    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator,ncols=150)):
            
            # forward

            # # # calflops
            input_cal = {}
            input_cal["img"]= inputs['img']
            input_cal["bbox_hand_left"] = meta_info["bbox_hand_left"]
            input_cal["bbox_hand_right"] = meta_info["bbox_hand_right"]

            flops, macs, params = calculate_flops(model=tester.model,
                                                kwargs = input_cal,
                                                # output_precision=4,
                                                # output_as_string=True,
                                                print_results=False)
            
            print("batchseize:%s ours FLOPs:%s   MACs:%s   Params:%s \n" %(cfg.test_batch_size, flops, macs, params))


            # start = time.time()
            # out = tester.model(inputs, targets, meta_info, 'test')
            # out = tester.model(input_cal["img"],input_cal["bbox_hand_left"],input_cal["bbox_hand_right"])
            # end = time.time()

            # joint_coord_out = out['joint_coord'].cpu().numpy()
            # inv_trans = out['inv_trans'].cpu().numpy()
            # joint_vaild = out['joint_valid'].cpu().numpy()

            # preds['joint_coord'].append(joint_coord_out)
            # preds['inv_trans'].append(inv_trans)
            # preds['joint_valid'].append(joint_vaild)

            # timer.append(end-start)
    
    
    # evaluate
    # print('average fps:',1 / np.mean(timer))
    # start1 = time.time()
    # preds = {k: np.concatenate(v) for k,v in preds.items()}
    # end1 = time.time()

    # mpjpe_dict, hand_accuracy, mrrpe = tester._evaluate(preds)
    # print(mpjpe_dict)
    # print('time per batch is',np.mean(timer))
    
    # with open(os.path.join(cfg.log_dir,"test_log.txt"),"a") as f:
    #     f.write(str(args.test_epoch)+"\n"+str(mpjpe_dict)+"\n")
    #     f.write(str(end1-start1)+"\n")


if __name__ == "__main__":
    test()
