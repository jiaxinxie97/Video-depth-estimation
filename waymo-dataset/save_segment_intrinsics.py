import os
import tensorflow as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import scipy.misc
import imageio
tf.enable_eager_execution()
from glob import glob
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def get_range_image(laser_name, return_index):
  """Returns range image given a laser name and its return index."""
  return range_images[laser_name][return_index]

def save_depth_image(projected_points,img):
   img_height,img_width,_=img.shape
   depth=np.zeros((img_height,img_width),dtype=np.uint16)
   for point in projected_points:
    #print(point[0],point[1],point[2])
    depth[int(point[1]),int(point[0])]=np.array(point[2]*256.0,dtype=np.uint16)
   return depth


#FILENAME = './waymo/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord'
def decode_tf(FILENAME):
  basename=os.path.basename(FILENAME)[:-9]
  if not os.path.exists('./waymo_decode_val/'+basename+'/intrinsic/'):
      os.makedirs('./waymo_decode_val/'+basename+'/intrinsic/')

  dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
  count=0
#print(len(dataset))
  frame = open_dataset.Frame()
  for data in dataset:
    frame.ParseFromString(bytearray(data.numpy()))
    #(range_images, camera_projections,
      #range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
#    plt.figure(figsize=(25, 20))
    #print (frame.context.camera_calibrations[0].intrinsic)
    #print (frame.context.camera_calibrations[0].width)
    #print (frame.context.camera_calibrations[0].height)
  #  if frame.context.stats.time_of_day=='Day':
    cam_file=open('./waymo_decode_val/'+basename+'/intrinsic/'+str(count).zfill(5)+'_cam.txt','w')
    intrinsics=np.array([frame.context.camera_calibrations[0].intrinsic])
    width=frame.context.camera_calibrations[0].width
    height=frame.context.camera_calibrations[0].height
    cam_file.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d'%(intrinsics[0,0],intrinsics[0,1],intrinsics[0,2],intrinsics[0,3],intrinsics[0,4]\
            ,intrinsics[0,5],intrinsics[0,6],intrinsics[0,7],intrinsics[0,8],width,height))
    cam_file.close()
    '''
    if count==0 and frame.context.stats.time_of_day=='Day':
      day_file.write('%s\n'%(basename))
      day_file.close()
    for index, image in enumerate(frame.images):
      if index==0:
        imageio.imwrite('./waymo_decode/'+basename+'/image/'+str(count).zfill(5)+'.jpg',tf.image.decode_jpeg(image.image))
        break
    frame.lasers.sort(key=lambda laser: laser.name)
        #show_range_image(get_range_image(open_dataset.LaserName.TOP, 0), 1)
        #show_range_image(get_range_image(open_dataset.LaserName.TOP, 1), 4)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
           frame,
           range_images,
           camera_projections,
           range_image_top_pose)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
           frame,
           range_images,
           camera_projections,
           range_image_top_pose,
           ri_index=1)
        # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
        # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)
    images = sorted(frame.images, key=lambda i:i.name)
    cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
    cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

        # The distance between lidar points and vehicle frame origin.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
    mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)
    cp_points_all_tensor = tf.cast(tf.gather_nd(
    cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))
    projected_points_all_from_raw_data = tf.concat(
           [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()
    saved_depth=save_depth_image(projected_points_all_from_raw_data,tf.image.decode_jpeg(image.image))
    imageio.imwrite('./waymo_decode/'+basename+'/depth/'+str(count).zfill(5)+'.png',saved_depth)
    '''
        #print(projected_points_all_from_raw_data.shape)
    count+=1
    
tf_files=sorted(glob('/disk1/jiaxin/waymo/val/*.tfrecord'))
#tf_files=tf_files[0:1]
for tf_file in tf_files:
    print (tf_file)
 #   if 'tfrecord' in tf_file:
    decode_tf(tf_file)
  #show_camera_image(image, frame.camera_labels, [3, 3, index+1])

