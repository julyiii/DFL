import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from config import cfg
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d, load_segms, swap_lr_labels_segm_target_channels
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import pickle
import torch.nn.functional as F

from eval import EvalUtil
import glob
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode):
        self.mode = mode # train, test, val
        self.img_path = cfg.interhand_images_path
        self.annot_path = cfg.interhand_anno_dir
        self.datalist_dir = cfg.datalistDir

        # output_json_file = "/data_ssd/huanyao/yaohuan/a2-j_based-detr/baseline_ours_mpjpe3d_campare_idx_3_2.json"
        # output_json_file = "./a2j_ours_mpjpe3d_campare_idx_3_3.json"
        #output_json_file = "./a2j_ours_mpjpe3d_campare_idx_2_3.json"
        output_json_file = "./save_for_pose_path_idx_4_15.json"

        # output_json_file = "./a2j_ours_mpjpe3d_campare_idx_1_3.json"
        # output_json_file = "./a2j_ours_mpjpe3d_campare_idx_4_4.json"
        with open(output_json_file) as f:
            filter_idx = json.load(f)
        if self.mode == 'val':
            self.rootnet_output_path = '../rootnet_output/rootnet_interhand2.6m_output_val.json'
        else:
            self.rootnet_output_path = '/data_ssd/data/huanyao/datasets/InterHand2.6M_5fps_batch1/rootnet_output/rootnet_interhand2.6m_output_all_test.json'
        self.transform = transform
        self.joint_num = 21 # single hand
        self.root_joint_idx = {'right': 20, 'left': 41} 
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num*2)
        self.use_single_hand_dataset = cfg.use_single_hand_dataset
        self.use_inter_hand_dataset = cfg.use_inter_hand_dataset
        self.vis = False
        
        ## use the total Interhand2.6M dataset
        datalist_file_path_sh = osp.join(self.datalist_dir , mode + '_datalist_sh_all.pkl')
        datalist_file_path_ih = osp.join(self.datalist_dir , mode + '_datalist_ih_all.pkl')
        
        # generate_new_datalist : whether to get datalist from existing file
        generate_new_datalist = True
        if osp.exists(datalist_file_path_sh) and osp.exists(datalist_file_path_ih):
            if (osp.getsize(datalist_file_path_sh) + osp.getsize(datalist_file_path_ih)) != 0:
                generate_new_datalist = False
        # generate_new_datalist = True
                
        # with open("./MANO_valid_filename_"+mode+".json") as f:
        #         MANO_valids = json.load(f)
        
        ## if the datalist is empty or doesn't exist, generate the pkl file and save the datalist
        if generate_new_datalist is True:
            self.datalist = []
            self.datalist_sh = []
            self.datalist_ih = []
            self.sequence_names = []
            
            # load annotation
            print("Load annotation from  " + osp.join(self.annot_path, self.mode))
            db = COCO(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data.json'))
            with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
                cameras = json.load(f)
            with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
                joints = json.load(f)
            with open("./MANO_valid_filename_"+mode+".json") as f:
                MANO_valids = json.load(f)
            with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot.json')) as f:
                mano_params = json.load(f)


            # rootnet is not used
            if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                print("Get bbox and root depth from " + self.rootnet_output_path)
                rootnet_result = {}
                with open(self.rootnet_output_path) as f:
                    annot = json.load(f)
                for i in range(len(annot)):
                    rootnet_result[str(annot[i]['annot_id'])] = annot[i]
            else:
                print("Get bbox and root depth from groundtruth annotation")

            pose_path = []
            save_for_pose_path = []
            # save_for_pose_path = {}
            # path_idx = 0
            # get images and annotations
            for aid in tqdm(list(db.anns.keys())[::]):
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                hand_type = ann['hand_type']
                capture_id = img['capture']
                subject = img['subject']
                seq_name = img['seq_name']
                cam = img['camera']
                frame_idx = img['frame_idx'] 
                hand_type_valid = ann['hand_type_valid']

                img_path = osp.join(self.img_path, self.mode, img['file_name'])

                

                campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
                focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
                joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
                joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)

                # # #filter like intag hand
                if hand_type != 'interacting' or (hand_type_valid != 1):
                    continue
                loaded_img = load_img(img_path)

                if loaded_img.mean() < 10:
                    continue
                    
                if str(frame_idx) not in mano_params[str(capture_id)]:continue
                try:
                    if mano_params[str(capture_id)][str(frame_idx)]['left'] is None or mano_params[str(capture_id)][str(frame_idx)]['right'] is None:
                        continue
                except:
                    pass
                
                # #add for choose pose
                # capture_pose =  img['file_name'].split('/')[0:2]
                # capture_pose_str = capture_pose[0] + capture_pose[1]
                # if capture_pose_str not in pose_path:
                #     pose_path.append(capture_pose_str)

                #     image_path = osp.join(self.img_path, mode,capture_pose[0],capture_pose[1],"*","*.jpg")
                    
                #     all_imgae = glob.glob(image_path) 
                #     choose_n = random.randint(0,len(all_imgae)-1)
                #     ###
                #     # path_idx +=1
                #     save_for_pose_path.append(all_imgae[choose_n])
                #     # save_for_pose_path[path_idx]=all_imgae[choose_n]
                    
                # else:
                #     continue

                joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]
                joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(self.joint_num*2)

                ## Filter the data that does not meet the training requirements.
                ## All preprocessing refers to the baseline of Interhand2.6M(ECCV2020).
                # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
                joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
                joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
                # hand_type = ann['hand_type']
                hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)
                
                ##获得segm vaild
                if str(frame_idx) not in  MANO_valids[str(capture_id)]:
                    segm_valid = False
                else:
                    MANO_valid = MANO_valids[str(capture_id)][str(frame_idx)]
                    if hand_type == 'right' and MANO_valid[0]==1:
                        segm_valid = True
                    elif hand_type == 'left' and MANO_valid[1]==1:
                        segm_valid = True
                    elif hand_type == 'interacting' and MANO_valid[0]*MANO_valid[1] == 1:
                        segm_valid = True
                    else:
                        segm_valid = False


                # rootnet is not used
                if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                    bbox = np.array(rootnet_result[str(aid)]['bbox'],dtype=np.float32)
                    abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
                else:
                    img_width, img_height = img['width'], img['height']
                    bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
                    bbox = process_bbox(bbox, (img_height, img_width))
                    abs_depth = {'right': joint_cam[self.root_joint_idx['right'],2], 'left': joint_cam[self.root_joint_idx['left'],2]} #根节点的深度值，以此为参考
                
                cam_param = {'focal': focal, 'princpt': princpt}
                joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
                data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 
                        'bbox': bbox, 'joint': joint, 'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 
                        'abs_depth': abs_depth, 'file_name': img['file_name'], 'capture': capture_id, 'cam': cam,
                        'frame': frame_idx, 'subject': subject, 'imgid': image_id,'segm_valid':segm_valid
                }
                
                if hand_type == 'right' or hand_type == 'left':
                    if self.use_single_hand_dataset is True:
                        self.datalist_sh.append(data)
                elif hand_type == 'interacting':
                    if self.use_inter_hand_dataset is True:
                        self.datalist_ih.append(data)
                if seq_name not in self.sequence_names:
                    self.sequence_names.append(seq_name)

            # output_json_file = '/data_ssd/huanyao/yaohuan/a2-j_based-detr/save_for_pose_path_idx_4_15.json'
            # with open(output_json_file, "w") as f:
            #         json.dump(save_for_pose_path, f)
            # print("Dumped save_for_pose_path_idx to %s" % output_json_file)       
            # Save the generated datalist to pkl file, easy to debug
            with open(datalist_file_path_sh, 'wb') as fs:
                pickle.dump(self.datalist_sh, fs)
            with open(datalist_file_path_ih, 'wb') as fi:
                pickle.dump(self.datalist_ih, fi)



            ###save filter image
            

        # Directly load the datalist saved in the previous file
        else:
            if self.use_single_hand_dataset is True:
                with open (datalist_file_path_sh, 'rb') as fsl:
                    self.datalist_sh = pickle.load(fsl)
            else:
                self.datalist_sh = []
            if self.use_inter_hand_dataset is True:
                with open (datalist_file_path_ih, 'rb') as fil:
                    self.datalist_ih = pickle.load(fil)
            else:
                self.datalist_ih = []
        if mode == 'train':
            self.datalist = (self.datalist_sh + self.datalist_ih)
        else:
            #通过idx过滤
            # fiter_datalist_ih= []
            # for idx in filter_idx:
            #     joint_select = self.datalist_ih[int(idx)]
            #     if joint_select['joint']['valid'].sum() != 42:
            #         continue
            #     fiter_datalist_ih.append(joint_select)
            #     print(idx)
            # self.datalist_ih = fiter_datalist_ih
            
            # 通过文件名过滤
            fiter_datalist_ih= []
            idx_in_filter_data = 0
            for idx in range(len(self.datalist_ih)):
                if self.datalist_ih[idx]['img_path'] in filter_idx:

                # if joint_select['joint']['valid'].sum() != 42:
                #     continue
                    fiter_datalist_ih.append(self.datalist_ih[idx])

                    print(idx_in_filter_data,self.datalist_ih[idx]['img_path'])

                    idx_in_filter_data+=1

            fiter_datalist_ih_1 = []
            # idx_map_to_img = [2,8,10,15,20,34,41,43,47,57,63,66,74,92,94,95,97]
            idx_map_to_img = [1,2,3,5,6,11,14,29,31,37,41,43,45]
            for idx_1 in idx_map_to_img:
                fiter_datalist_ih_1.append(fiter_datalist_ih[idx_1])
            self.datalist_ih = fiter_datalist_ih_1

            # self.datalist_ih = fiter_datalist_ih
            self.datalist = (self.datalist_sh + self.datalist_ih)



            # self.datalist = (self.datalist_sh + self.datalist_ih)

            #self.datalist = self.datalist_ih[:]



    
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))


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
        # return len(self.datalist)
        # return 10000
        return 10


    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy();
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None].copy()),1)
        seq_name = data['seq_name']
        contact_vis_np = np.zeros((32, 2)).astype(np.float32)

        # image load
        img = load_img(img_path)
        
        #segm load
        segm_valid = data['segm_valid']


        if (segm_valid is True) and (self.mode) == 'train':
            segm_path = img_path.replace('images','segms').replace('jpg','png')
            segm = np.array(load_segms(segm_path),dtype=np.uint8)
        else:
            # #保留双手中只有单手有mano标注的分割图
            # if hand_type[0] * hand_type[1] == 1:
            #     segm_path = img_path.replace('images','segms').replace('jpg','png')
            #     segm = np.array(load_segms(segm_path),dtype=np.uint8)
            # else:
            #     segm = np.zeros((512,334),dtype=np.uint8)
            segm = np.zeros((512,334),dtype=np.uint8)


        segm = (
            np.stack((segm, segm, segm), axis=2)
            if segm is not None
            else None
        )

        # # ##dense pic load
        # if (segm_valid is True) and (self.mode) == 'train':
        #     dense_path = img_path.replace('images','dense_pose').replace('jpg','png')
        #     dense = np.array(load_img(dense_path),dtype=np.uint8)
        # else:
        #     dense =  np.zeros((512,334,3),dtype=np.uint8)
        dense =  np.zeros((512,334,3),dtype=np.uint8)




        # augmentation
        img, dense,segm, joint_coord, joint_valid, hand_type, inv_trans, do_flip, joint_cam_aug = augmentation(img,dense,segm,bbox, joint_coord, joint_valid, hand_type, self.mode, self.joint_type,joint_cam)
        rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],dtype=np.float32).reshape(1)
        root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]])*1.0


        vis = False
        if vis:
            #2d
            import os
            ##interhand关节点定义顺序与two hand这篇论文定义不同，因此改变关节点顺序
            # single_hand_valid_reorder = [20,7,6,5,11,10,9,19,18,17,15,14,13,3,2,1,0,4,8,12,16]  #interhand joints顺序 -> mano 需要在确认一下指尖的valid
            # double_hand_valid_reorder = single_hand_valid_reorder + [i+self.joint_num  for i in single_hand_valid_reorder]
            # pred_joint_coord_img = (normlize_joint_img_corrd*256)[double_hand_valid_reorder]
            # vis_kps = pred_joint_coord_img.copy()
            # vis_valid = joint_valid.copy()[double_hand_valid_reorder]
            # # vis_valid[:] = 0
            # # vis_valid[4] = 1 
            # filename = 'out_' + str(idx) + '.jpg'
            # vis_keypoints(img.transpose(2,0,1), vis_kps, vis_valid, self.skeleton, filename)
            cam_param = data['cam_param']
            root_align = joint_cam_aug - joint_cam_aug[41]
            root_align_wo_rot =  joint_cam - joint_cam[41]
            pred_joint_coord_cam = joint_coord[:,0:2]
            dir_path = os.path.dirname("/data_ssd/yaohuan/A2J-Transformer/project_3d.jpg")
            if not os.path.exists(dir_path):
                # 如果不存在则创建文件夹
                os.makedirs(dir_path)
            single_hand_valid_reorder = [20,7,6,5,11,10,9,19,18,17,15,14,13,3,2,1,0,4,8,12,16]  #interhand joints顺序 -> mano 需要在确认一下指尖的validdouble_hand_valid_reorder = single_hand_valid_reorder + [i+self.joint_num  for i in single_hand_valid_reorder]
            double_hand_valid_reorder = single_hand_valid_reorder + [i+self.joint_num  for i in single_hand_valid_reorder]
            #pred_joint_coord_img = pred_joint_coord_cam[double_hand_valid_reorder]
            pred_joint_coord_img = pred_joint_coord_cam

            vis_kps = pred_joint_coord_img.copy()
            vis_valid = joint_valid.copy()[double_hand_valid_reorder]
            # vis_valid[:] = 0
            # vis_valid[4] = 1            
            # filename = 'out_' + str(idx) + '.jpg'
            #vis_keypoints(img.transpose(2,0,1), vis_kps, vis_valid, self.skeleton, filename)
            vis_keypoints(img.transpose(2,0,1), vis_kps, vis_valid, self.skeleton, filename=os.path.basename(img_path), save_path=dir_path)
            #3d 
            vis_3d_keypoints(root_align, joint_valid, self.skeleton, filename = "rot3d.jpg", score_thr=0.4, line_width=3, circle_rad=3)
            vis_3d_keypoints(root_align_wo_rot, joint_valid, self.skeleton, filename = "rot3dwo_rot.jpg", score_thr=0.4, line_width=3, circle_rad=3)


        
        
        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth, root_valid =\
            transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type)

        # Some images are blank, filter for training
        if np.sum(img) < 1e-4 :
            joint_valid *= 0
            root_valid *= 0
            hand_type_valid *= 0
            contact_vis_np *= 0
            segm_valid *= 0

        img = self.transform(img.astype(np.float32)) / 255.

        dense = self.transform(dense.astype(np.float32)) / 255.

        # use zero mask.
        mask = np.zeros((img.shape[1], img.shape[2])).astype(np.bool)
        mask = self.transform(mask.astype(np.uint8))
        bbox_hand_left = np.array([0.0,0.0,cfg.input_img_shape[0],cfg.input_img_shape[1]],dtype=np.float32)
        bbox_hand_right = np.array([0.0,0.0,cfg.input_img_shape[0],cfg.input_img_shape[1]],dtype=np.float32)
        
        ##part segms
        if segm is  not None:
            segm = torch.FloatTensor(segm)
            segm = segm.permute(2, 0, 1)
            
            if do_flip:
                segm = swap_lr_labels_segm_target_channels(segm,img_path)
                segm = segm.clone().long()
                
        # downsample to target resolution
        img_segm_128 = (
            F.interpolate(segm[None, :, :, :].float(), 128, mode="nearest")
            .long()
            .squeeze()
        )[0]
        
        if img_segm_128.max() > 32:
            segm_valid = False

        # #add for 3d joints
        # #防止右手不存在时，其根节点值很奇怪，导致左手学的相互对于右手根节点的值很奇怪
        # if  joint_valid[self.root_joint_idx['right']] == 0:
        #     joint_cam_aug[self.root_joint_idx['right']] = joint_cam_aug[self.root_joint_idx['left']]
            
            
        # #root align
        # joint_cam_aug[self.joint_type['right'],:] = joint_cam_aug[self.joint_type['right'],:] - joint_cam_aug[self.root_joint_idx['right'],:]
        # joint_cam_aug[self.joint_type['left'],:] = joint_cam_aug[self.joint_type['left'],:] - joint_cam_aug[self.root_joint_idx['left'],:]

        
        seq_name = data['seq_name']
        inputs = {'img': img, 'mask': mask}
        targets = {'joint_coord': joint_coord, 'joint_cam':joint_cam_aug, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type,'img_segm_128':img_segm_128,'segm_valid':segm_valid,"dense":dense}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'hand_type_valid': hand_type_valid, 'segm_valid':segm_valid,
                     'inv_trans': inv_trans, 'capture': int(data['capture']), 'seq_name':seq_name,'cam': int(data['cam']), 'frame': int(data['frame']),
                     'bbox_hand_right':bbox_hand_right,'bbox_hand_left':bbox_hand_left}
        return inputs, targets, meta_info


    def evaluate(self, preds):
        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, inv_trans, joint_valid_used = preds['joint_coord'], preds['inv_trans'], preds['joint_valid']
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)
        
        mpjpe_sh = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih = [[] for _ in range(self.joint_num*2)]
        mpjpe_sh_2d = [[] for _ in range(self.joint_num*2)]
        mpjpe_sh_3d = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih_2d = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih_3d = [[] for _ in range(self.joint_num*2)]
        tot_err = []
        mpjpe_dict = {}

        mpjpe_dict_save_filter = {}

        # xxd
        a2j_ours_img_idx = {}
        # xxd
        
        mrrpe = []
        acc_hand_cls = 0; hand_cls_cnt = 0;

        util = EvalUtil(num_kp=self.joint_num*2)

        for n in tqdm(range(sample_num),ncols=150):
            vis = False
            mpjpe_per_data_list = []
            mpjpe_per_data = 0

            data = gts[n]
            bbox, cam_param, joint, gt_hand_type, hand_type_valid = data['bbox'], data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
            hand_type = data['hand_type']

            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord'].copy() 
            gt_joint_img = joint['img_coord'].copy() 
            
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
            
            mpjpe_sample = []
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
                        mpjpe_sample.append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))

            #mpjpe_dict_save_filter[n] = float(np.mean(mpjpe_sample))

            ## xy mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,:2] - gt_joint_coord[j,:2])**2)))
                        # continue
                    else:
                        mpjpe_ih_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,:2] - gt_joint_coord[j,:2])**2)))
            ## depth mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh_3d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,2] - gt_joint_coord[j,2])**2)))
                        # continue
                    else:
                        mpjpe_ih_3d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,2] - gt_joint_coord[j,2])**2)))

            util.feed(gt_joint_coord,joint_valid,pred_joint_coord_cam)


            vis_2d = False    
            if vis_2d:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                vis_kps = pred_joint_coord_img.copy()
                vis_kps_gt = gt_joint_img.copy()
                vis_valid = joint_valid.copy()
                capture = str(data['capture'])
                cam = str(data['cam'])
                frame = str(data['frame'])
                filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
                vis_keypoints(_img, vis_kps, vis_kps_gt, bbox, vis_valid, self.skeleton, filename)
                print('vis 2d over')

            
            vis_3d = False
            if vis_3d:
                filename = 'out_' + str(n) + '_3d.jpg'
                vis_3d_cam = pred_joint_coord_cam.copy()
                vis_3d_cam_left = pred_joint_coord_cam[self.joint_type['left']].copy()
                vis_3d_cam_left[:,2] = pred_joint_coord_cam[self.joint_type['left'],2]
                vis_3d_cam_right = pred_joint_coord_cam[self.joint_type['right']].copy()
                vis_3d_cam_right[:,2] = pred_joint_coord_cam[self.joint_type['right'],2] 
                vis_3d = np.concatenate((vis_3d_cam_left, vis_3d_cam_right), axis= 0)
                vis_3d_keypoints(vis_3d, joint_valid, self.skeleton, filename)
                print('vis 3d over')
            ###############################origin##########################################
            # vis_2d = True   
            # if vis_2d:
            #     img_path = data['img_path']
            #     cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            #     _img = cvimg[:,:,::-1].transpose(2,0,1)
            #     vis_kps = pred_joint_coord_img.copy()
            #     vis_kps_gt = gt_joint_img.copy()
            #     # vis_valid = joint_valid.copy()
            #     vis_valid = np.ones(self.joint_num*2).copy()
            #     capture = str(data['capture'])
            #     cam = str(data['cam'])
            #     frame = str(data['frame'])
            #     # filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
            #     filename = str(n) + '_' + gt_hand_type + '.jpg'
            #     # if n ==60:
            #     #     print("\nvis_kps_60:{}\n".format(vis_kps))
            #     #     print("\nvis_kps_gt_60:{}\n".format(vis_kps_gt))
            #     vis_keypoints(_img, vis_kps, vis_kps_gt, bbox, vis_valid, self.skeleton, filename)
            #     print('vis 2d over')

            
            # vis_3d = True
            # if vis_3d:
            #     # filename = 'out_' + str(n) + '_3d.jpg'
            #     filename = str(n) + '_3d.jpg'
            #     vis_3d_cam = pred_joint_coord_cam.copy()
            #     vis_3d_cam_left = pred_joint_coord_cam[self.joint_type['left']].copy()
            #     vis_3d_cam_left[:,2] = pred_joint_coord_cam[self.joint_type['left'],2]
            #     vis_3d_cam_right = pred_joint_coord_cam[self.joint_type['right']].copy()
            #     vis_3d_cam_right[:,2] = pred_joint_coord_cam[self.joint_type['right'],2]
            #     vis_3d_gt =  gt_joint_coord.copy()
            #     vis_valid_3d = np.ones(self.joint_num*2).copy()

            #     vis_3d = np.concatenate((vis_3d_cam_left, vis_3d_cam_right), axis= 0)
            #     vis_3d_keypoints(vis_3d, vis_3d_gt, vis_valid_3d, self.skeleton, filename)
            #     print('vis 3d over')
            ###############################origin##########################################

            ####################################xxd##################################### 
            vis_2d = False   
            if vis_2d:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                vis_kps = pred_joint_coord_img.copy()
                vis_kps_gt = gt_joint_img.copy()
                # vis_valid = joint_valid.copy()
                vis_valid = np.ones(self.joint_num*2).copy()
                capture = str(data['capture'])
                cam = str(data['cam'])
                frame = str(data['frame'])
                # filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
                filename = str(n) + '_' + gt_hand_type + '.jpg'
                # if n ==60:
                #     print("\nvis_kps_60:{}\n".format(vis_kps))
                #     print("\nvis_kps_gt_60:{}\n".format(vis_kps_gt))
                vis_keypoints(_img, vis_kps, vis_kps_gt, bbox, vis_valid, self.skeleton, filename)
                print('vis 2d over')

            
            vis_3d = False
            if vis_3d:
                # filename = 'out_' + str(n) + '_3d.jpg'
                # filename = str(n) + '_3d.jpg'
                filename_right = str(n) + '_3d_right.jpg'
                filename_left = str(n) + '_3d_left.jpg'
                vis_3d_cam = pred_joint_coord_cam.copy()
                vis_3d_cam_left = pred_joint_coord_cam[self.joint_type['left']].copy()
                vis_3d_cam_left[:,2] = pred_joint_coord_cam[self.joint_type['left'],2]
                vis_3d_cam_right = pred_joint_coord_cam[self.joint_type['right']].copy()
                vis_3d_cam_right[:,2] = pred_joint_coord_cam[self.joint_type['right'],2]
                vis_3d_gt =  gt_joint_coord.copy()
                # vis_valid_3d = np.ones(self.joint_num*2).copy()
                vis_valid_3d_right = np.concatenate((np.ones(self.joint_num),np.zeros(self.joint_num)))
                vis_valid_3d_left = np.concatenate((np.zeros(self.joint_num),np.ones(self.joint_num)))
                root_left2right = np.array([80,0,0])
                vis_3d_cam = np.concatenate((vis_3d_cam_right,vis_3d_cam_left), axis= 0)
                vis_3d_keypoints(vis_3d_cam.copy(), vis_3d_gt.copy(), vis_valid_3d_right, self.skeleton, filename_right,root_left2right,'right')
                vis_3d_keypoints(vis_3d_cam.copy(), vis_3d_gt.copy(), vis_valid_3d_left, self.skeleton, filename_left,root_left2right,'left')
                print('vis 3d over')
            
            # img_path = data['img_path']

            # a2j_ours_img_idx[n] = img_path
            # output_json_file = "/data_ssd/huanyao/yaohuan/a2-j_based-detr/baseline_ours_img_idx_3_2.json"
            # with open(output_json_file, "w") as f:
            #     json.dump(a2j_ours_img_idx, f)
            # print("Dumped baseline_ours_img_idx to %s" % output_json_file)
            ####################################xxd#####################################            
        
        if hand_cls_cnt > 0: 
            handness_accuracy = acc_hand_cls / hand_cls_cnt
            print('Handedness accuracy: ' + str(handness_accuracy))
        if len(mrrpe) > 0: 
            mrrpe_num = sum(mrrpe)/len(mrrpe)
            print('MRRPE: ' + str(mrrpe_num))
        print()

        # import json
        # with open("our_model_fiter_reults.json", "w") as f:
        #     f.write(json.dumps(mpjpe_dict_save_filter))



        if self.use_inter_hand_dataset is True and self.use_single_hand_dataset is True:
            print('..................MPJPE FOR TOTAL HAND..................')
            eval_summary = 'MPJPE for each joint: \n'
            for j in range(self.joint_num*2):
                tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
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
            for j in range(self.joint_num*2):
                mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
            print(eval_summary)
            mpjpe_sh_mean = np.mean(mpjpe_sh)
            print('MPJPE for single hand sequences: %.2f' % (mpjpe_sh_mean))
            mpjpe_dict['single_hand_total'] = mpjpe_sh_mean
            print()

            ## xy
            eval_summary_2d = 'MPJPE for each joint 2d: \n'
            for j in range(self.joint_num*2):
                mpjpe_sh_2d[j] = np.mean(np.stack(mpjpe_sh_2d[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary_2d += (joint_name + ': %.2f, ' % mpjpe_sh_2d[j])
            print(eval_summary_2d) 
            mpjpe_sh_2d_mean = np.mean(mpjpe_sh_2d)
            print('MPJPE for single hand sequences 2d: %.2f' % (mpjpe_sh_2d_mean))
            mpjpe_dict['single_hand_2d'] = mpjpe_sh_2d_mean
            print()

            ## z
            eval_summary_3d = 'MPJPE for each joint depth: \n'
            for j in range(self.joint_num*2):
                mpjpe_sh_3d[j] = np.mean(np.stack(mpjpe_sh_3d[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary_3d += (joint_name + ': %.2f, ' % mpjpe_sh_3d[j])
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
            print(eval_summary_3d) 
            mpjpe_ih_3d_mean = np.mean(mpjpe_ih_3d)
            print('MPJPE for interacting hand sequences 3d: %.2f' % (mpjpe_ih_3d_mean))
            mpjpe_dict['inter_hand_depth'] = mpjpe_ih_3d_mean
            print()

        # util.print_data_euclidean_dist()
        # mean, median, auc, pck_curve_all, threshs = util.get_measures(0, 50, 11)
        # util.plot_pck(threshs,pck_curve_all,auc)

        if hand_cls_cnt > 0 and len(mrrpe) > 0:
            return mpjpe_dict, handness_accuracy, mrrpe_num
        else:
            return mpjpe_dict, None, None
        
    def evaluate_3d(self, preds):
        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, inv_trans, joint_valid_used = preds['joint_coord'], preds['inv_trans'], preds['joint_valid']
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)
        
        mpjpe_sh = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih = [[] for _ in range(self.joint_num*2)]
        mpjpe_sh_2d = [[] for _ in range(self.joint_num*2)]
        mpjpe_sh_3d = [[] for _ in range(self.joint_num*2)]
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
            bbox, cam_param, joint, gt_hand_type, hand_type_valid = data['bbox'], data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
            hand_type = data['hand_type']

            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord'].copy()
            gt_joint_img = joint['img_coord'].copy()
            
            ## use original joint_valid param.
            joint_valid = joint['valid']
            # joint_valid = joint_valid_used[n]

            # # restore xy coordinates to original image space
            # pred_joint_coord_img = preds_joint_coord[n].copy()
            # pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
            # pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
            # for j in range(self.joint_num*2):
            #     pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])

            # # restore depth to original camera space
            # pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

            # # add root joint depth
            # pred_joint_coord_img[self.joint_type['right'],2] += data['abs_depth']['right']
            # pred_joint_coord_img[self.joint_type['left'],2] += data['abs_depth']['left']

            # # back project to camera coordinate system
            # pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)



            #here rewrite 仅仅改变了这里,将预测出来的3d点替代原来2.5D经过运算得到的结果
            pred_joint_coord_cam = preds_joint_coord[n].copy()
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
            

            # ## xy mpjpe
            # for j in range(self.joint_num*2):
            #     if joint_valid[j]:
            #         if gt_hand_type == 'right' or gt_hand_type == 'left':
            #             mpjpe_sh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,:2] - gt_joint_coord[j,:2])**2)))
            #             # continue
            #         else:
            #             mpjpe_ih_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,:2] - gt_joint_coord[j,:2])**2)))
            # ## depth mpjpe
            # for j in range(self.joint_num*2):
            #     if joint_valid[j]:
            #         if gt_hand_type == 'right' or gt_hand_type == 'left':
            #             mpjpe_sh_3d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,2] - gt_joint_coord[j,2])**2)))
            #             # continue
            #         else:
            #             mpjpe_ih_3d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,2] - gt_joint_coord[j,2])**2)))

            vis_2d = False    
            if vis_2d:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                vis_kps = pred_joint_coord_img.copy()
                vis_kps_gt = gt_joint_img.copy()
                vis_valid = joint_valid.copy()
                capture = str(data['capture'])
                cam = str(data['cam'])
                frame = str(data['frame'])
                filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
                vis_keypoints(_img, vis_kps, vis_kps_gt, bbox, vis_valid, self.skeleton, filename)
                print('vis 2d over')

            
            vis_3d = False
            if vis_3d:
                filename = 'out_' + str(n) + '_3d.jpg'
                vis_3d_cam = pred_joint_coord_cam.copy()
                vis_3d_cam_left = pred_joint_coord_cam[self.joint_type['left']].copy()
                vis_3d_cam_left[:,2] = pred_joint_coord_cam[self.joint_type['left'],2]
                vis_3d_cam_right = pred_joint_coord_cam[self.joint_type['right']].copy()
                vis_3d_cam_right[:,2] = pred_joint_coord_cam[self.joint_type['right'],2] 
                vis_3d = np.concatenate((vis_3d_cam_left, vis_3d_cam_right), axis= 0)
                vis_3d_keypoints(vis_3d, joint_valid, self.skeleton, filename)
                print('vis 3d over')
                
        
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
                tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
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
            for j in range(self.joint_num*2):
                mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
            print(eval_summary)
            mpjpe_sh_mean = np.mean(mpjpe_sh)
            print('MPJPE for single hand sequences: %.2f' % (mpjpe_sh_mean))
            mpjpe_dict['single_hand_total'] = mpjpe_sh_mean
            print()

            # ## xy
            # eval_summary_2d = 'MPJPE for each joint 2d: \n'
            # for j in range(self.joint_num*2):
            #     mpjpe_sh_2d[j] = np.mean(np.stack(mpjpe_sh_2d[j]))
            #     joint_name = self.skeleton[j]['name']
            #     eval_summary_2d += (joint_name + ': %.2f, ' % mpjpe_sh_2d[j])
            # print(eval_summary_2d) 
            # mpjpe_sh_2d_mean = np.mean(mpjpe_sh_2d)
            # print('MPJPE for single hand sequences 2d: %.2f' % (mpjpe_sh_2d_mean))
            # mpjpe_dict['single_hand_2d'] = mpjpe_sh_2d_mean
            # print()

            # ## z
            # eval_summary_3d = 'MPJPE for each joint depth: \n'
            # for j in range(self.joint_num*2):
            #     mpjpe_sh_3d[j] = np.mean(np.stack(mpjpe_sh_3d[j]))
            #     joint_name = self.skeleton[j]['name']
            #     eval_summary_3d += (joint_name + ': %.2f, ' % mpjpe_sh_3d[j])
            # print(eval_summary_3d) 
            # mpjpe_sh_3d_mean = np.mean(mpjpe_sh_3d)
            # print('MPJPE for single hand sequences 3d: %.2f' % (mpjpe_sh_3d_mean))
            # mpjpe_dict['single_hand_depth'] = mpjpe_sh_3d_mean
            # print()


        if self.use_inter_hand_dataset is True:
            print('..................MPJPE FOR INTER HAND..................')
            ## xyz
            eval_summary = 'MPJPE for each joint: \n'
            for j in range(self.joint_num*2):
                mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
            print(eval_summary) 
            mpjpe_ih_mean = np.mean(mpjpe_ih)
            print('MPJPE for interacting hand sequences: %.2f' % (mpjpe_ih_mean))
            mpjpe_dict['inter_hand_total'] = mpjpe_ih_mean
            print()

            # ## xy
            # eval_summary_2d = 'MPJPE for each joint 2d: \n'
            # for j in range(self.joint_num*2):
            #     mpjpe_ih_2d[j] = np.mean(np.stack(mpjpe_ih_2d[j]))
            #     joint_name = self.skeleton[j]['name']
            #     eval_summary_2d += (joint_name + ': %.2f, ' % mpjpe_ih_2d[j])
            # print(eval_summary_2d) 
            # mpjpe_ih_2d_mean = np.mean(mpjpe_ih_2d)
            # print('MPJPE for interacting hand sequences 2d: %.2f' % (mpjpe_ih_2d_mean))
            # mpjpe_dict['inter_hand_2d'] = mpjpe_ih_2d_mean
            # print()

            # ## z
            # eval_summary_3d = 'MPJPE for each joint depth: \n'
            # for j in range(self.joint_num*2):
            #     mpjpe_ih_3d[j] = np.mean(np.stack(mpjpe_ih_3d[j]))
            #     joint_name = self.skeleton[j]['name']
            #     eval_summary_3d += (joint_name + ': %.2f, ' % mpjpe_ih_3d[j])
            # print(eval_summary_3d) 
            # mpjpe_ih_3d_mean = np.mean(mpjpe_ih_3d)
            # print('MPJPE for interacting hand sequences 3d: %.2f' % (mpjpe_ih_3d_mean))
            # mpjpe_dict['inter_hand_depth'] = mpjpe_ih_3d_mean
            # print()

        if hand_cls_cnt > 0 and len(mrrpe) > 0:
            return mpjpe_dict, handness_accuracy, mrrpe_num
        else:
            return mpjpe_dict, None, None
        