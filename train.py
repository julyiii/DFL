# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn

from base import Tester
from tqdm import tqdm
import numpy as np
import time
import os

from torch.utils.tensorboard import SummaryWriter
from utils.misc import fix_seeds

from torch import autograd
'''
command for opening tensorboard is : 
tensorboard --logdir=your/root/file/path/output/tensorboard_log --port 
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--continue', default=False, dest='continue_train', action='store_true')
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

def main():
    fix_seeds(0)
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.exp_name,args.continue_train)
    cudnn.benchmark = True
    cfg.save_args()


    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    
    tbwriter = SummaryWriter(cfg.tensorboard_dir)

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            #with autograd.detect_anomaly():
            # forward
            trainer.optimizer.zero_grad()

            loss = trainer.model(inputs, targets, meta_info, 'train')
            loss = {k:loss[k].mean() for k in loss}
            
            # backward
            loss['total_loss'].backward()
            #torch.nn.utils.clip_grad_norm_(parameters=trainer.model.parameters(), max_norm=10, norm_type=2)

            trainer.optimizer.step()

            trainer.gpu_timer.toc()
            if itr % 25 ==0:
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                    ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
                trainer.logger.info(' '.join(screen))
            if itr % 100 ==0: 
                tbwriter.add_scalar('loss/total_loss', loss['total_loss'], epoch*len(trainer.batch_generator)+itr)
                
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        
        # save model
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)
        if epoch >= cfg.test_start_epoch and epoch % cfg.test_per_epoch==0:
            mpjpe_dict, hand_accuracy, mrrpe = test_per_epoch(epoch)
            # if cfg.use_single_hand_dataset:
            #     tbwriter.add_scalar('mpjpe/single_hand_total', mpjpe_dict['single_hand_total'], epoch)
            #     tbwriter.add_scalar('mpjpe/single_hand_2d', mpjpe_dict['single_hand_2d'], epoch)
            #     tbwriter.add_scalar('mpjpe/single_hand_depth', mpjpe_dict['single_hand_depth'], epoch)
            # if cfg.use_inter_hand_dataset:
            #     tbwriter.add_scalar('mpjpe/inter_hand_total', mpjpe_dict['inter_hand_total'], epoch)
            #     tbwriter.add_scalar('mpjpe/inter_hand_2d', mpjpe_dict['inter_hand_2d'], epoch)
            #     tbwriter.add_scalar('mpjpe/inter_hand_depth', mpjpe_dict['inter_hand_depth'], epoch)
            # if cfg.use_single_hand_dataset and cfg.use_inter_hand_dataset:
            #     tbwriter.add_scalar('mpjpe/total', mpjpe_dict['total'], epoch)
            # if hand_accuracy is not None:
            #     tbwriter.add_scalar('hand_accuracy', hand_accuracy, epoch)
            # if mrrpe is not None:
            #     tbwriter.add_scalar('mrrpe', mrrpe, epoch)
    tbwriter.close()
    



def test_per_epoch(test_epoch):

    args = parse_args()
    cfg.set_args(cfg,args.gpu_ids,args.exp_name)
    cudnn.benchmark = True

    args.test_set = 'test'
    args.test_epoch = str(test_epoch)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set)
    tester._make_model()

    preds = {'joint_coord': [], 'inv_trans': [], 'joint_valid': [] }

    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
            
            # forward
            out = tester.model(inputs, targets, meta_info, 'test')

            joint_coord_out = out['joint_coord'].cpu().numpy()
            inv_trans = out['inv_trans'].cpu().numpy()
            joint_vaild = out['joint_valid'].cpu().numpy()


            preds['joint_coord'].append(joint_coord_out)
            preds['inv_trans'].append(inv_trans)
            preds['joint_valid'].append(joint_vaild)

            
    # evaluate
    preds = {k: np.concatenate(v) for k,v in preds.items()}
    mpjpe_dict, hand_accuracy, mrrpe = tester._evaluate(preds)   
    
    with open(os.path.join(cfg.log_dir,"test_log.txt"),"a") as f:
        f.write(str(test_epoch)+"\n"+str(mpjpe_dict)+'\n')
        
    return mpjpe_dict, hand_accuracy, mrrpe



if __name__ == "__main__":
    main()
