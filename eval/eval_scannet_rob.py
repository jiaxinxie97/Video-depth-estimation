from __future__ import division
import sys
import cv2
import os
import numpy as np
import argparse
from depth_evaluation_utils import *
from glob import glob
parser = argparse.ArgumentParser()
parser.add_argument("--kitti_dir", type=str, help='Path to the KITTI dataset directory')
parser.add_argument("--pred_file", type=str, help="Path to the prediction file")
parser.add_argument("--test_file_list", type=str, default='./data/ScanNet/ScanNet_files_ROB.txt', 
    help="Path to the list of test files")
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=5, help="Threshold for maximum depth")
args = parser.parse_args()

def main():
    
    pred_depths = np.load(args.pred_file)
    print len(pred_depths)
    test_frames=open('/mnt/external2/DORN_scannet_ROB_v2/DORN_scannet_excludeinf.txt','r')
    test_frames=test_frames.readlines()

    gt_depths = []
    pred_depths_resized = []
    test_files=[]    
    for test_frame in test_frames:
         test_files.append('/mnt/external3/scanet_test/'+test_frame[:-1].split(' ')[0]+'/depth/'+test_frame[:-1].split(' ')[1]+'.png')
        # if os.path.exists('/data/jiaxin/eigen_multi-inputs/'+test_dir+'/'+os.path.basename(test_image)[:-4]+'.npy'):
        #   test_files.append(test_image)
    num_test=len(test_files)
    print num_test 
    for t_id in range(num_test):
        #camera_id = cams[t_id]  # 2 is left, 3 is right
#        test_dir,test_name=os.path.split(test_files[t_id])
    #    print test_files[t_id]
        depth=cv2.imread(test_files[t_id],-1)
        depth=depth/1000.
#        print (depth.shape)
        pred_depth=pred_depths[t_id]
#        print (pred_depth.shape)
#        print(pred_depth.min(),pred_depth.max())
#        pred_depths_resized.append(pred_depth)
        pred_depths_resized.append(
            cv2.resize(pred_depth, 
                       (depth.shape[1], depth.shape[0]), 
                       interpolation=cv2.INTER_LINEAR))
       # depth = generate_depth_map(gt_calib[t_id], 
        #                           gt_files[t_id], 
        #                           im_sizes[t_id], 
        #                           camera_id, 
        #                           False, 
        #                           True)
        gt_depths.append(depth.astype(np.float32))
    pred_depths = pred_depths_resized

    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)
    irmse    = np.zeros(num_test, np.float32)
    SILog   = np.zeros(num_test, np.float32)
    mask_number=0
    for i in range(num_test):    
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])
 #       print (pred_depth.shape)
        mask = np.logical_and(gt_depth > args.min_depth, 
                              gt_depth < args.max_depth)
        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        gt_height, gt_width = gt_depth.shape
#	print(gt_height,gt_width)
      #  crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
     #                    0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
 #       print(crop)
      #  crop_mask = np.zeros(mask.shape)
      #  crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    #    mask = np.logical_and(mask, crop_mask)
       # mask_number=mask_number+mask.sum()
        # Scale matching
#        scalor=1
        scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
        pred_depth[mask] *= scalor
        #if np.abs(scalor-1)>0.2:
        #print "max and min", pred_depth[mask].max(), pred_depth[mask].min()
        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i],SILog[i],irmse[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])
      

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10},{:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'irmse','SILog','a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f},{:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(),irmse.mean(),SILog.mean(), a1.mean(), a2.mean(), a3.mean()))
    print "mask_mean:",mask_number/num_test
main()
