from __future__ import division
import sys
import cv2
import os
import numpy as np
import argparse
from depth_evaluation_utils import *
import matplotlib.pyplot as plt
import scipy.misc
from utils import *
def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='jet'):
    # convert to disparity
  #  depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth


parser = argparse.ArgumentParser()
parser.add_argument("--kitti_dir", type=str, help='Path to the KITTI dataset directory')
parser.add_argument("--pred_file", type=str, help="Path to the prediction file")
parser.add_argument("--test_file_list", type=str, default='./data/kitti/test_files_eigen.txt', 
    help="Path to the list of test files")
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=80, help="Threshold for maximum depth")
parser.add_argument('--crop', type=str, default='grag', help="crop_type")

args = parser.parse_args()

def main():
    pred_depths = np.load(args.pred_file)
    print( 'pred depth shape:',pred_depths.shape)
    test_files = read_text_lines(args.test_file_list)
    if os.path.isdir(args.kitti_dir):
        gt_files, gt_calib, im_sizes, im_files, cams = \
            read_file_data(test_files, args.kitti_dir)

        num_test = len(pred_depths)

       
        gt_depths = []
    
        for t_id in range(num_test):
            camera_id = cams[t_id]  # 2 is left, 3 is right

            depth, _= generate_depth_map(gt_calib[t_id], 
                                    gt_files[t_id], 
                                    im_sizes[t_id], 
                                    camera_id, 
                                    True, 
                                    True) 
            gt_depths.append(depth.astype(np.float32))

    else:
        gt_depths = np.load(args.kitti_dir)    
        num_test = len(gt_depths)        


    rms     = np.zeros(num_test, np.float32)
    irmse   = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)
    SILog   = np.zeros(num_test, np.float32)

    for i in range(num_test):    
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])
        if gt_depth.shape !=pred_depth.shape:
            pred_depth= cv2.resize(pred_depth, 
                        (gt_depth.shape[1], gt_depth.shape[0]), 
                        interpolation=cv2.INTER_LINEAR)
        mask = np.logical_and(gt_depth > args.min_depth, 
                              gt_depth < args.max_depth)

        gt_height, gt_width = gt_depth.shape
        if args.crop=='eigen':
            crop=np.array([3,218,44,1179])
        elif args.crop=='grag':
            crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                         0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
   
     
        pred_depth[mask] *= scalor

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i],SILog[i],irmse[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10},{:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms','irmse', 'SILog', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f},{:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(),irmse.mean(),SILog.mean(), a1.mean(), a2.mean(), a3.mean()))

    f = open("kitti_eval.txt",'a')
    f.write("%s\n"%args.pred_file)
    f.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10},{:>10}\n".format('abs_rel', 'sq_rel', 'rms', 'log_rms','irmse', 'SILog', 'a1', 'a2', 'a3'))
    f.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f},{:10.4f}\n\n".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(),irmse.mean(),SILog.mean(), a1.mean(), a2.mean(), a3.mean()))
    f.close()

main()
