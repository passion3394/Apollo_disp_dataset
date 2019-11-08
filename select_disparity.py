import os
import sys
import cv2
import glob
import shutil
import random
import pickle
import numpy as np

#the evaluation ids of the movable objects
movable_ids = {33,34,35,36,37,38,39,40,161,162,163,164,165,166,167,168}

'''
this script will do these things:
(1) according to all the depth of camera 5, find the corresponding scene image_2 and image_3, from the image_2 labeling file to check if the labeling contains some movable objects, if the area of the movable objects is more than thres pixels(thres is the threshold), then skip the pic.
(2) after (1), we will got the image_2/image_3/depth pics, then we will convert the depth image to disparity image, the convert rule will be:
    (a)  disparity value less than 20*256 on the down half will be zero
    (b)  disparity value less than 6*256 on the up half will be zero
(3) split training and val 
'''
# generate the train and val occording to the flag
def gen_train_val(one_list,flag):
    if flag == 'training':
        training_2 = training_image2_dir + one_list[0].split('/')[-1].replace('_Camera_5','')
        training_3 = training_image3_dir + one_list[1].split('/')[-1].replace('_Camera_6','')
        training_disp = training_disp_dir + one_list[2].split('/')[-1].replace('_Camera_5','')
        print('gen:'+one_list[0])
        convert_disp(one_list,training_2,training_3,training_disp)
        
    else:
        val_2 = val_image2_dir + one_list[0].split('/')[-1].replace('_Camera_5','')
        val_3 = val_image3_dir + one_list[1].split('/')[-1].replace('_Camera_6','')
        val_disp = val_disp_dir + one_list[2].split('/')[-1].replace('_Camera_5','')
        print('gen:'+one_list[0])
        convert_disp(one_list,val_2,val_3,val_disp)
        
# write the disparity map to dest_path
# according to the label_path, assign disparity of 0 to the area of dest_path
# and 
# (a)  disparity value less than 20*256 on the down half will be zero
# (b)  disparity value less than 6*256 on the up half will be zero
def convert_disp(one_list,image_2_path,image_3_path,dest_path):
    #prepare the paths of the four pics
    left_img_path = one_list[0]
    right_img_path = one_list[1]
    left_depth_path = one_list[2]
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    left_depth = cv2.imread(left_depth_path,-1)

    #handle the pose of the two RGB images
    left_pose_path = left_img_path.replace('ColorImage','Pose')
    left_pose_path = left_pose_path.replace(left_pose_path.split('/')[-1],'pose.txt')
    print(left_pose_path)
    right_pose_path = right_img_path.replace('ColorImage','Pose')
    right_pose_path = right_pose_path.replace(right_pose_path.split('/')[-1],'pose.txt')
    print(right_pose_path)
    left_pose = open(left_pose_path,'r')
    right_pose = open(right_pose_path,'r')

    #according to the pose.txt in the coresponding record dir to get the pose of the img
    left_poses =left_pose.readlines()
    right_poses = right_pose.readlines()
    left_img_pose = []
    right_img_pose = []
    for pose in left_poses:
        pose_split = pose.split(' ')
        if pose_split[-1].strip() == left_img_path.split('/')[-1]:
            for xi in range(len(pose_split)-1):
                left_img_pose.append(float(pose_split[xi]))
    
    for pose in right_poses:
        pose_split = pose.split(' ')
        if pose_split[-1].strip() == right_img_path.split('/')[-1]:
            for xi in range(len(pose_split)-1):
                right_img_pose.append(float(pose_split[xi]))

    if len(left_img_pose) == 0 or len(right_img_pose) ==0:
        return 
        
    #convert the list to 4*4 arrary
    print(left_img_pose)
    print(right_img_pose)
    left_img_pose = np.mat(left_img_pose)
    right_img_pose = np.mat(right_img_pose)
    left_img_pose = left_img_pose.reshape(4,4)
    right_img_pose = right_img_pose.reshape(4,4)
    print(left_img_pose)
    print(right_img_pose)

    #compute the extrinsic para of the two cameras
    trans_right_to_left =  right_img_pose.I * left_img_pose
    #print(trans_right_to_left)
    #print(trans_right_to_left[:3,:3])
    print(trans_right_to_left[:3,3])

    #print(trans_left_to_right)

    #do the rectification work
    distCoeff = np.zeros(4)

    image_size = (left_img.shape[1],left_img.shape[0])
    print(image_size)

    camera5_mat = [ [2304.54786556982, 0.0, 1686.23787612802],
                    [0.0, 2305.875668062, 1354.98486439791],
                    [0.0, 0.0, 1.0] ]
    camera6_mat = [ [2300.39065314361, 0.0, 1713.21615190657],
                    [0.0, 2301.31478860597, 1342.91100799715],
                    [0.0, 0.0, 1.0]]

    camera5_mat = np.array(camera5_mat)
    camera6_mat = np.array(camera6_mat)

    # compare the two image
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=camera5_mat,
        distCoeffs1=distCoeff,
        cameraMatrix2=camera6_mat,
        distCoeffs2=distCoeff,
        imageSize=image_size,
        R=trans_right_to_left[:3,:3],
        T=trans_right_to_left[:3,3],flags=0,alpha=0)
        #flags=cv2.CALIB_ZERO_DISPARITY,
        #alpha=1)

    '''
    print(R1)
    print(R2)
    print(P1)
    print(P2)
    print(Q)
    print(roi1)
    print(roi2)
    '''

    # for warping image 5
    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrix=camera5_mat,
        distCoeffs=distCoeff,
        R=R1,
        newCameraMatrix=P1,
        size=image_size,
        m1type=cv2.CV_32FC1)

    # for warping image 6
    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrix=camera6_mat,
        distCoeffs=distCoeff,
        R=R2,
        newCameraMatrix=P2,
        size=image_size,
        m1type=cv2.CV_32FC1)

    left_image_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    cv2.imwrite(image_2_path,left_image_rect)
    right_image_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
    cv2.imwrite(image_3_path,right_image_rect)
    left_depth_rect = cv2.remap(left_depth, map1x, map1y, cv2.INTER_LINEAR)

    #after rectify, the focus and principal point of the cameras have changed
    f = Q[2][3]
    B = 1 / Q[3][2]
    cx_minus_cxp = Q[3][3] / (-Q[3][2])
    disp_img = cx_minus_cxp + f*B*200 / left_depth_rect
    disp_img = np.uint16(disp_img) * 50  #multi the disp value by 50
    print(disp_img.max())
    '''
    disp_img_m = cv2.imread('left_img_disp.png')
    bld = cv2.addWeighted(left_image_rect,0.3,disp_img_m,0.7,0)
    cv2.imwrite('bld.png',bld)
    '''

    #according to the label_path, assign disparity of 0 to the area of dest_path
    img = cv2.imread(one_list[3],-1)
    for i in movable_ids:
        mask = img == i
        rows,cols = np.where(mask)
        disp_img[rows,cols] = 0
    
    #(a)(b)
    '''
    g_mask = disp_img < 100 * 10
    rows,cols = np.where(g_mask)
    disp_img[rows,cols] = 0
    '''
    cv2.imwrite(dest_path,disp_img) 

def judge_label_valid(label_path):
    #first judge the image number of bytes to skip the invalid labeling
    label_file_size = os.path.getsize(label_path)
    if label_file_size < 100000:
        return False
    
    #secondly statistic the pixels of each movable object
    thres = 5000  # threshold1: if the pixels of each movable object is more than thres, skip it
    img = cv2.imread(label_path,-1)
    try:
        for i in movable_ids:
            mask = img == i
            rows,cols = np.where(mask)
            if len(rows) > thres:
                return False
    except:
        return False
    return True

# apollo_depth dataset dir name
apollo_depth = './apollo_depth/Depth/'
# apollo scene dataset dir name
apollo_scene = './apollo_scene'
# get all the scene filename
apollo_scene_filelist = glob.glob(apollo_scene+'/**/*_Camera_5.jpg', recursive=True)

# judge if the new dataset exists or not 
if not os.path.exists('apollo_depth_v1'):
    os.makedirs('./apollo_depth_v1')
if not os.path.exists('apollo_depth_v1/training'):
    os.makedirs('./apollo_depth_v1/training')
if not os.path.exists('apollo_depth_v1/val'):
    os.makedirs('./apollo_depth_v1/val')

# declair each image_2/image_3/disp_occ_0 of training or val
training_image2_dir = './apollo_depth_v1/training/image_2/'
training_image3_dir = './apollo_depth_v1/training/image_3/'
training_disp_dir = './apollo_depth_v1/training/disp_occ_0/'
val_image2_dir = './apollo_depth_v1/val/image_2/'
val_image3_dir = './apollo_depth_v1/val/image_3/'
val_disp_dir = './apollo_depth_v1/val/disp_occ_0/'

# judge if the six folders exist
if not os.path.exists(training_image2_dir):
    os.makedirs(training_image2_dir)
if not os.path.exists(training_image3_dir):
    os.makedirs(training_image3_dir)
if not os.path.exists(training_disp_dir):
    os.makedirs(training_disp_dir)
if not os.path.exists(val_image2_dir):
    os.makedirs(val_image2_dir)
if not os.path.exists(val_image3_dir):
    os.makedirs(val_image3_dir)
if not os.path.exists(val_disp_dir):
    os.makedirs(val_disp_dir)

all_list = [] # the total list of the image_2/image_3/disp_occ_0

write_pickle = False

if write_pickle:
    #enumerate all the scene filename
    for filename in apollo_scene_filelist:
        one_line = []
        label_path = filename.replace('ColorImage','Label')
        label_path = label_path.replace('.jpg','_bin.png')
        if not os.path.exists(label_path):
            print('label image of ' + label_path + ' do not exist')
            continue
        #print(label_path)
        #occording to the camera 5 to generate the camera 6 image filename and disp_ooc_0 name
        right_name = filename.replace('Camera_5','Camera_6')
        right_name = right_name.replace('Camera 5','Camera 6')
        disp_ind = filename.find('Record')
        disp_name = apollo_depth + filename[disp_ind:-4] + '.png'
        #print(filename)
        #print(right_name)
        #print(disp_name)
        if not os.path.exists(right_name):
            print('the pair of '+filename+' and '+ right_name + ' do not exists')
            continue
        if not os.path.exists(disp_name):
            print('the combination of '+filename+' and '+ right_name + ' and '+disp_name + ' do not exists')
            continue
        #judge the label file 
        if not judge_label_valid(label_path):
            continue
        #judge the bytes number of the depth image
        #if the total bytes number is less than 2100000,then skip the depth image
        depth_file_size = os.path.getsize(disp_name)
        if depth_file_size < 2100000:
            continue
        one_line.append(filename)
        one_line.append(right_name)
        one_line.append(disp_name)
        one_line.append(label_path)
        all_list.append(one_line)

    with open('all_list.pickle','wb') as f:
        pickle.dump(all_list,f)

else:
    with open('all_list.pickle','rb') as f:
        all_list = pickle.load(f)
        print(all_list)

total_list_num = len(all_list)
print(total_list_num)
random.shuffle(all_list)

# split the training and val dataset
training_rate = 0.8
for i in range(len(all_list)):
    if i < int(len(all_list)*training_rate):
        gen_train_val(all_list[i],'training') 
    else:
        gen_train_val(all_list[i],'val')
