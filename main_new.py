import dataloader
import itertools
import time
import glob
from dataloader import CustomDataset, get_transform
from efficientdet.dataset import get_exact_file_name_from_path, ChallengeDataset, Resizer, Normalizer, collater#, Augmenter, 
from torch.utils.data import DataLoader
from torchvision import transforms
import model
import evaluation
import torch
import torch.distributed as dist
import math
import argparse
import os
import sys
import xml.etree.ElementTree as elemTree
import cv2
import numpy as np
from IPython.core.display import Image

import train_new
from train_new import boolean_string, load_weight_from_file
from efficientdet.utils import BBoxTransform, ClipBoxes
from backbone import EfficientDetBackbone
from utils.utils import invert_affine, preprocess_video, postprocess


'''
!!!!!!!!!!!!!!!!!!!!! 필독!!!!!!!!!!!!!!!!!!!!!!!!!!!
** 컨테이너 내 기본 제공 폴더
- /datasets : read only 폴더 (각 태스크를 위한 데이터셋 제공)
- /tf/notebooks :  read/write 폴더 (참가자가 Wirte 용도로 사용할 폴더)
1. 참가자는 /datasets 폴더에 주어진 데이터셋을 적절한 폴더(/tf/notebooks) 내에 복사/압축해제 등을 진행한 뒤 사용해야합니다.
   예시> Jpyter Notebook 환경에서 압축 해제 예시 : !bash -c "unzip /datasets/objstrgzip/18_NLP_comments.zip -d /tf/notebooks/
   예시> Terminal(Vs Code) 환경에서 압축 해제 예시 : bash -c "unzip /datasets/objstrgzip/18_NLP_comments.zip -d /tf/notebooks/
   
2. 참가자는 각 문제별로 데이터를 로드하기 위해 적절한 path를 코드에 입력해야합니다. (main.py 참조)
3. 참가자는 모델의 결과 파일(Ex> prediction.txt)을 write가 가능한 폴더에 저장되도록 적절 한 path를 입력해야합니다. (main.py 참조)
4. 세션/컨테이너 등 재시작시 위에 명시된 폴더(datasets, notebooks) 외에는 삭제될 수 있으니 
   참가자는 적절한 폴더에 Dataset, Source code, 결과 파일 등을 저장한 뒤 활용해야합니다.
   
!!!!!!!!!!!!!!!!!!!!! 필독!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

model_dir = 'saved_model'

def print_indented(n_sp, *args):
    print('  ' * n_sp, *args)

def make_folder(path) :
    try :
        os.mkdir(os.path.join(path))
    except :
        pass

def save_model(model_name, model, optimizer, scheduler):
    make_folder(model_dir)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    torch.save(state, os.path.join(model_dir, model_name + '.pth'))
    print('model saved')
    
def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_dir, model_name + '.pth'))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')
    

def get_list_of_file_path_under_1st_with_2nd_extension(direc, ext = ''):
    li_path_total = []
    is_extension_given = is_this_empty_string(ext)
    for dirpath, dirnames, filenames in os.walk(os.path.expanduser(direc)):
        n_file_1 = len(filenames)
        if n_file_1:
            if is_extension_given:
                li_path = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(ext.lower())]
            else:
                li_path = [os.path.join(dirpath, f) for f in filenames]
            n_file_2 = len(li_path)
            if n_file_2:
                li_path_total += li_path
    return sorted(li_path_total)

def get_list_of_image_path_under_this_directory(dir_img):
    dir_img = os.path.expanduser(dir_img)
    li_fn_img = get_list_of_file_path_under_1st_with_2nd_extension(dir_img)
    if is_this_empty_string(ext):
        li_fn_img = [fn for fn in li_fn_img if is_image_file(fn)]
    return sorted(li_fn_img)

def display(preds, imgs, obj_list):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        
        return imgs[i]


def get_side_length_of_division(wid, nn, overlap_ratio, n_sp):
    print_indented(n_sp, 'get_side_length_of_division START')
    print_indented(n_sp + 1, 'wid : ' + str(wid) + ', nn : ' + str(nn) + ', overlap_ratio : ' + str(overlap_ratio))
    side_len = int(round(wid / (1.0 + nn * (1.0 - overlap_ratio))))
    print_indented(n_sp + 1, 'side_len : ' + str(side_len))
    #exit(0);
    print_indented(n_sp, 'get_side_length_of_division END')
    return side_len


def get_interval(wid, ss, nn, n_sp):
    print_indented(n_sp, 'get_interval START')
    print_indented(n_sp + 1, 'wid :', wid, ', ss :', ss, ', nn :', nn)
    interval = int(0)
    if nn > 0:
        overlap_ratio = get_overlap_ratio(wid, ss, nn)
        interval = int(round((1.0 - overlap_ratio) * ss))
        #interval = int(round((1.0 - overlap_ratio) * ss))
    print_indented(n_sp + 1, 'interval : ' + str(interval))
    #exit(0)
    print_indented(n_sp, 'get_interval END')
    return interval

def get_offset_list(wid, ss, nn, n_sp):
    print_indented(n_sp, 'get_offset_list START')
    print_indented(n_sp + 1, 'wid : ' + str(wid) + ', ss : ' + str(ss) + ', nn : ' + str(nn))
    interval = get_interval(wid, ss, nn, n_sp + 1)
    overlap = ss - interval 
    #t0 = get_overlap(wid, ss, nn, n_sp + 1);    print_indented(n_sp + 1, 't0 :', t0)
    print_indented(n_sp + 1, 'overlap :', overlap, ', interval :', interval);   #exit()
    li = [0]
    while li[-1] + ss < wid:
        li.append(li[-1] + interval)
        dif = wid - (li[-1] + ss)
        print_indented(n_sp + 2, 'dif : ' + str(dif) + ' / overlap : ' + str(overlap))
        if dif < overlap:
            li[-1] += dif
    print_indented(n_sp + 1, 'li :', li)
    #exit(0)
    print_indented(n_sp, 'get_offset_list END')
    return li


def get_overlap_ratio(wid, ss, nn):
    return 1.0 - (wid - ss) / (nn * ss)

#'''
def get_overlap(wid, ss, nn, n_sp):
    print_indented(n_sp, 'get_overlap START')
    print_indented(n_sp + 1, 'wid : ' + str(wid) + ', ss : ' + str(ss) + ', nn : ' + str(nn)) #  exit()
    overlap = int(-1)
    if nn > 0:
        overlap_ratio = get_overlap_ratio(wid, ss, nn)
        print_indented(n_sp + 1, 'overlap_ratio : ' + str(overlap_ratio)) #  exit()
        #overlap_f = overlap_ratio * ss; print_indented(n_sp + 1, 'overlap_f :', overlap_f);
        overlap = int(round(overlap_ratio * ss))
    print_indented(n_sp + 1, 'overlap : ' + str(overlap)) #  exit()
    print_indented(n_sp, 'get_overlap END')
    return overlap
#'''

def get_num_and_side_length_of_division(wid, min_side, max_side, overlap_ratio, n_sp):
    print_indented(n_sp, 'get_num_and_side_length_of_division START')
    print_indented(n_sp + 1, 'wid : ' + str(wid) + ', min_side : ' + str(min_side) + ', max_side : ' + str(max_side) + ', overlap_ratio : ' + str(overlap_ratio))
    n_div = 0    #print
    w_cur = wid;    w_pre = -1;
    if w_cur > max_side:
        while w_cur >= min_side:
            w_pre = w_cur
            n_div += 1
            w_cur = get_side_length_of_division(wid, n_div, overlap_ratio, n_sp + 2)
            print_indented(n_sp + 2, 'n_div : ' + str(n_div) + ', w_cur : ' + str(w_cur))
        n_div -= 1
    print_indented(n_sp + 1, 'n_div : ' + str(n_div) + ', w_pre : ' + str(w_pre))
    #exit(0)
    print_indented(n_sp, 'get_num_and_side_length_of_division END')
    return n_div, w_pre

def compute_offsets_4_mosaicking(min_side, max_side_ratio, min_overlap_ratio, wid, hei, n_sp):
    print_indented(n_sp, 'compute_offsets_4_mosaicking START')
    print_indented(n_sp + 1, 'min_side :', min_side, ', min_overlap_ratio :', min_overlap_ratio, ', wid :', wid, ', hei :', hei)
    #   compute min_side_enough
    max_side = min_side * max_side_ratio
    print_indented(n_sp + 1, 'max_side : ' + str(max_side))
    #   if both w and h is smaller than max_side
    if wid <= max_side and hei <= max_side:
        print_indented(n_sp + 1, "wid <= max_side and hei <= max_side")
        li_offset_x = [0];    li_offset_y = [0];    len_side = -1; 
    #   else if h is smaller than max_side
    elif hei <= max_side:
        print_indented(n_sp + 1, "hei <= max_side")
        li_offset_y = [0];  
        n_x, len_side_x = get_num_and_side_length_of_division(wid, min_side, max_side, min_overlap_ratio, n_sp + 2)
        #len_side_x = get_side_length_of_division(wid, n_x, min_overlap_ratio, n_sp + 1)
        len_side = len_side_x
        li_offset_x = get_offset_list(wid, len_side, n_x, n_sp + 2)
    #   else if h is smaller than min_side_enough
    elif wid <= max_side:
        print_indented(n_sp + 1, "wid <= max_side")
        li_offset_x = [0]
        n_y, len_side_y = get_num_of_division(hei, min_side, max_side, min_overlap_ratio, n_sp + 2)
        #len_side_y = get_side_length_of_division(hei, n_y, min_overlap_ratio, n_sp + 1)
        len_side = len_side_y
        li_offset_y = get_offset_list(hei, len_side, n_x, n_sp + 2)
    #   else #  both w and h is larger than min_side_enough
    else:
        print_indented(n_sp + 1, "wid > max_side and hei > max_side")
        n_x, len_side_x = get_num_and_side_length_of_division(wid, min_side, max_side, min_overlap_ratio, n_sp + 2)
        #len_side_x = get_side_length_of_division(wid, n_x, min_overlap_ratio, n_sp + 1)
        print_indented(n_sp + 2, 'n_x b4 : ' + str(n_x) + ', len_side_x : ' + str(len_side_x))
        n_y, len_side_y = get_num_of_division(hei, min_side, max_side, min_overlap_ratio, n_sp + 2)
        #len_side_y = get_side_length_of_division(hei, n_y, min_overlap_ratio, n_sp + 1)
        print_indented(n_sp + 2, 'n_y b4 : ' + str(n_y) + ', len_side_y : ' + str(len_side_y)) #exit()
        len_side = math.floor(max([len_side_x, len_side_y]))
        n_x = get_num_of_division(wid, len_side, min_overlap_ratio, n_sp + 2)
        n_y = get_num_of_division(hei, len_side, min_overlap_ratio, n_sp + 2)
        li_offset_x = get_offset_list(wid, len_side, n_x, n_sp + 2)
        li_offset_y = get_offset_list(hei, len_side, n_y, n_sp + 2)
    li_offset = np.floor(list(itertools.product(li_offset_x, li_offset_y)))
    print_indented(n_sp + 1, 'li_offset :', li_offset);
    print_indented(n_sp + 1, 'len_side :', len_side);  #exit(0);
    print_indented(n_sp, 'compute_offsets_4_mosaicking END')
    return li_offset, len_side


def resize_and_pad_bottom_or_right(image, new_shape, bg_color):

    height, width, _ = image.shape
    if height > width:
        scale = new_shape / height
        resized_height = new_shape
        resized_width = int(width * scale)
    else:
        scale = new_shape / width
        resized_height = int(height * scale)
        resized_width = new_shape

    image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((new_shape, new_shape, 3)) * bg_color
    new_image[0 : resized_height, 0 : resized_width] = image
    return new_image
  

def letterbox(img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
    #else:  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh



class LoadImages:  # for inference
    def __init__(self, path, include_original, is_letterbox, max_side_ratio, n_sp, min_divide_side = 608, min_overlap_ratio = 0.2, img_size = 416, no_disp = False):
    #def __init__(self, path, img_size = 416):
        self.include_original = include_original
        self.no_disp = no_disp
        self.li_offset_xy = [];
        self.w_ori = -1;    self.h_ori = -1
        self.min_overlap_ratio = min_overlap_ratio; self.li_offset_xy = []; self.len_side = -1
        self.min_divide_side = max([min_divide_side, img_size])
        self.height = img_size
        self.is_letterbox = is_letterbox
        self.max_side_ratio = max_side_ratio
        self.n_sp = n_sp
        img_formats = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
        vid_formats = ['.mov', '.avi', '.mp4']
        
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob('%s/*.*' % path))
        elif os.path.isfile(path):
            files = [path]
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)
        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]
        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            #ret_val, img0 = self.cap.read()
            ret_val, im0_bgr_hwc = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    #ret_val, img0 = self.cap.read()
                    ret_val, im0_bgr_hwc = self.cap.read()
            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')
        else:
            # Read image
            self.count += 1
            #img0 = cv2.imread(path)  # BGR
            im0_bgr_hwc = cv2.imread(path)  # BGR
            #assert img0 is not None, 'File Not Found ' + path
            #assert im0_bgr_hwc is not None, 'File Not Found ' + path
            if im0_bgr_hwc is None:
                return None, None, None
            if not self.no_disp:
                print('image %g/%g %s: ' % (self.count, self.nF, path), end='')
                
        h_ori, w_ori, _ = im0_bgr_hwc.shape
        if h_ori != self.h_ori or w_ori != self.w_ori:
            self.h_ori = h_ori; self.w_ori = w_ori;
            #print('self.min_divide_side : ', self.min_divide_side); exit()
            self.li_offset_xy, self.len_side = compute_offsets_4_mosaicking(self.min_divide_side, self.max_side_ratio, self.min_overlap_ratio, self.w_ori, self.h_ori, self.n_sp + 1)
        li_im_rgb_chw = []
        for offset_xy in self.li_offset_xy:
            x_from, y_from = offset_xy
            x_to, y_to = min([x_from + self.len_side, self.w_ori]), min([y_from + self.len_side, self.h_ori]);
            im_bgr_hwc, *_ = letterbox(im0_bgr_hwc[int(y_from) : int(y_to), int(x_from) : int(x_to)], new_shape      = self.height)
            im_rgb_chw = im_bgr_hwc[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and then HWC to CHW
            im_rgb_chw = np.ascontiguousarray(im_rgb_chw, dtype=np.float32)  # uint8 to float32
            im_rgb_chw /= 255.0  # 0 - 255 to 0.0 - 1.0
            li_im_rgb_chw.append(im_rgb_chw)
        if(self.include_original and len(li_im_rgb_chw) > 1):
            if self.is_letterbox:
                #print('is_letterbox TRUE'); exit(0);
                im_bgr_hwc, *_ = letterbox(im0_bgr_hwc, new_shape = self.height, mode = 'square')
            else:
                #print('is_letterbox FALSE'); exit(0);
                im_bgr_hwc = resize_and_pad_bottom_or_right(im0_bgr_hwc, new_shape = self.height, bg_color = 127)
            im_rgb_chw = im_bgr_hwc[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and then HWC to CHW
            im_rgb_chw = np.ascontiguousarray(im_rgb_chw, dtype=np.float32)  # uint8 to float32
            im_rgb_chw /= 255.0  # 0 - 255 to 0.0 - 1.0
            li_im_rgb_chw.append(im_rgb_chw)
        return li_im_rgb_chw, path, im0_bgr_hwc
        
    def __len__(self):
        return self.nF  # number of files

def test(model, dir_img, input_size, threshold, iou_threshold, use_cuda, device, prediction_dir, is_mosaic, n_sp):
    print_indented(n_sp, 'test STRAT')
    class_name = {1 : 'bus', 2 : 'car', 3 :'carrier', 4 : 'cat', 5 : 'dog', 
                  6 : 'motorcycle', 7 : 'movable_signage', 8 : 'person', 9 : 'scooter', 10 : 'stroller', 
                  11 : 'truck', 12 : 'wheelchair', 13 : 'barricade', 14 : 'bench', 15 : 'chair',
                  16 : 'fire_hydrant', 17 : 'kiosk', 18 : 'parking_meter', 19 : 'pole', 20 : 'potted_plant', 
                  21 : 'power_controller', 22 : 'stop', 23 : 'table', 24 : 'traffic_light_controller', 
                  25 : 'traffic_sign', 26 : 'tree_trunk', 27 : 'bollard', 28 : 'bicycle'}
    
    obj_list = list(class_name.values())
    print_indented(n_sp + 1, 'obj_list :', obj_list);  #exit(0);
    model.to(device)
    model.eval()

    #print('prediction_dir : ', prediction_dir); exit(0);
    make_folder(prediction_dir)
   
    path_prediction = prediction_dir + '/predictions_score_' + str(threshold) + '_iou_' + str(iou_threshold) + '.xml'
    #print('path_prediction : ', path_prediction);   exit(0);
    
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()


    # 이미지 전체 반복
    pred_xml = elemTree.Element('predictions')
    pred_xml.text = '\n  '
    #batch_size = data_loader_test.batch_size
    #n_img = len(li_path_img)
    if is_mosaic:
        data_loader_test = LoadImages(dir_img, include_original = True, is_letterbox = False, max_side_ratio = 1.5, min_divide_side = 512, min_overlap_ratio = 0.2, img_size = 512, no_disp = True, n_sp = n_sp + 1) 
    else:
        data_loader_test = get_list_of_image_path_under_this_directory(dir_img)
    n_img = len(data_loader_test)
    for idx, data in enumerate(data_loader_test) :
    #for idx, path_img in enumerate(li_path_img) :
        
        #if 10 == idx:
        #    path_img = '/tf/notebooks/datasets/07_object_detection/val/ZED3_052263_L_P012028.png'
        if 0 == idx % 100:
            #print('idx : ', idx, ' / ', n_img, '\tpath_img : ', path_img)
            print_indented(n_sp + 1, 'idx :', idx, ' /', n_img)
            start_time = time.time()
            if idx > 0:
                fps = 100.0 / (time.time() - start_time)
                print_indented(n_sp + 1, 'fps :', fps)
                start_time = time.time()
        if not is_mosaic:
            path_img = data
            im_bgr = cv2.imread(path_img)
            if im_bgr is None:
                continue
            img_name = get_exact_file_name_from_path(path_img)
            # frame preprocessing
            ori_imgs, framed_imgs, framed_metas = preprocess_video(im_bgr, max_size=input_size)

            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            #x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
            x = x.to(torch.float32).permute(0, 3, 1, 2)
        else:     
            x = data
            if x[0] is None:
                continue
            print_indented(n_sp + 1, 'len(x) :', len(x));  #  exit(0)
            print_indented(n_sp + 1, 'type(x[0]) :', type(x[0]));  #  exit(0)
            print_indented(n_sp + 1, 'type(x[1]) :', type(x[1]));  #  exit(0)
            print_indented(n_sp + 1, 'type(x[2]) :', type(x[2]));  #  exit(0)

            print_indented(n_sp + 1, 'x[1] :', x[1]);  #  exit(0)
            print_indented(n_sp + 1, 'len(x[0]) :', len(x[0]));  #  exit(0)
            print_indented(n_sp + 1, 'type(x[0][0]) :', type(x[0][0]));  #  exit(0)
            print_indented(n_sp + 1, 'x[0][0].shape :', x[0][0].shape );  #  exit(0)
            print_indented(n_sp + 1, 'x[2].shape :', x[2].shape);  #  exit(0)

            print_indented(n_sp + 1, 'type(x) :', type(x));    #exit(0)
            x = torch.from_numpy(np.stack(x[0])).to(device)
        # model predict
        with torch.no_grad():
            #t0 = model(x)
            #print_indented(n_sp + 1, 'type(t0) :', type(t0));   #exit(0)
            #print_indented(n_sp + 1, 'len(t0) :', len(t0));   exit(0)
            features, regression, classification, anchors = model(x)
            #exit(0);
        
        if is_mosaic:
            out = postprocess_mosaic(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

        else:
            out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

            # result
            out = invert_affine(framed_metas, out)
            boxes, labels, scores = out[0]['rois'], out[0]['class_ids'], out[0]['scores']
            '''
            img_show = display(out, ori_imgs, obj_list)
            # show frame by frame
            #cv2.imwrite(img_name + '_detected.jpg', img_show);
        
            #if idx == n_img:
            #    cv2.imwrite('detected.jpg', img_show);
                cv2.imwrite('detected.jpg', img_show);
            #exit(0);
            '''
        texts = []
        # 이미지 한장에 대하여
        #for n in range(count) :
        xml_image = elemTree.SubElement(pred_xml, 'image')
            
        xml_image.attrib['name'] = img_name
        xml_image.text = '\n    '

        for index in range(len(boxes)) :
            box, label, score = boxes[index], int(labels[index] + 1), scores[index]
            #box, label, score = boxes[index], int(labels[index]), scores[index]
            # class, score, x1, y1, x2, y2
            xml_predict = elemTree.SubElement(xml_image, 'predict')
            xml_predict.tail = '\n    '
            xml_predict.attrib['class_name'] = class_name[label] 
            xml_predict.attrib['score'] = str(float(score))
            xml_predict.attrib['x1'] = str(int(round(box[0])))
            xml_predict.attrib['y1'] = str(int(round(box[1])))
            xml_predict.attrib['x2'] = str(int(round(box[2])))
            xml_predict.attrib['y2'] = str(int(round(box[3])))
            '''
            print('\n');    
            print('class_name[label] :', class_name[label]);
            print('float(score) :', float(score));
            print('int(round(box[0]))', int(round(box[0])));
            print('int(round(box[1]))', int(round(box[1])));
            print('int(round(box[2]))', int(round(box[2])));
            print('int(round(box[3]))', int(round(box[3])));
            '''
            if index == len(boxes) - 1 :
                xml_predict.tail = '\n  '
        #exit(0);
        xml_image.tail = '\n  '
        #if idx == len(data_loader_test) - 1 and n == (count - 1):
        #if 12 == idx:# and n == (count - 1):
        #    xml_image.tail = '\n'
        #    break;
        if idx == n_img - 1:# and n == (count - 1):
            xml_image.tail = '\n'
    pred_xml = elemTree.ElementTree(pred_xml)
    #pred_xml.write(prediction_dir + '/predictions.xml')
    pred_xml.write(path_prediction)
    print_indented(n_sp, 'test END')

def train_default(model, data_loader_train, data_loader_val, device, num_epochs, prediction_dir, print_iter, optimizer=None, lr_scheduler=None) :
    # 모델을 GPU나 CPU로 옮깁니다
    model.to(device)

    # 옵티마이저(Optimizer)를 만듭니다
    if optimizer == None :
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001,
                                    momentum=0.9, weight_decay=0.0005)
        
    # 학습률 스케쥴러를 만듭니다
    if lr_scheduler == None :
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)

    for epoch in range(num_epochs):
        
        print(epoch)
        
        model.train()
        count = 0
        
        for images, targets in data_loader_train:
            count += len(images)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            world_size = 1

            if dist.is_available() and dist.is_initialized() :
                world_size = dist.get_world_size()

            if world_size >= 2:
                with torch.no_grad():
                    names = []
                    values = []
                    for k in sorted(loss_dict.keys()):
                        names.append(k)
                        values.append(loss_dict[k])
                    values = torch.stack(values, dim=0)
                    dist.all_reduce(values)
                    if average:
                        values /= world_size
                    loss_dict = {k: v for k, v in zip(names, values)}

            losses_reduced = sum(loss for loss in loss_dict.values())

            loss_value = losses_reduced.item()
            if count % print_iter < data_loader_train.batch_size :
                print('epoch {} [{}/{}] loss_classifier : {} loss_box_reg : {} loss_objectness : {} loss_rpn_box_reg : {}'.format(
                    epoch, count, len(data_loader_train.dataset), loss_dict['loss_classifier'], loss_dict['loss_box_reg'], loss_dict['loss_objectness'], loss_dict['loss_rpn_box_reg']
                ))

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(targets)
                print(loss_dict)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        validation(model, data_loader_val, device, prediction_dir)
                
        save_model('{}'.format(epoch), model, optimizer, lr_scheduler)

        lr_scheduler.step()

def validation(model, data_loader_val, device, prediction_dir) :
    test(model, data_loader_val, device, prediction_dir)
    GT_PATH = os.path.join(DATASET_PATH, 'val')
    DR_PATH = os.path.join(prediction_dir,'predictions.xml')
    res = evaluation.evaluation_metrics(GT_PATH, DR_PATH)
    print('validation : ', res)
    return res

try:
    from nipa import nipa_data
    #DATASET_PATH = nipa_data.get_data_root('deepfake')
    DATASET_PATH = nipa_data.get_data_root('object_detection')
except:
    DATASET_PATH = os.path.join('/tf/notebooks/datasets/07_object_detection')


def get_list_of_file_path_under_1st_with_2nd_extension(direc, ext = ''):
    li_path_total = []
    is_extension_given = is_this_empty_string(ext)
    for dirpath, dirnames, filenames in os.walk(os.path.expanduser(direc)):
        n_file_1 = len(filenames)
        if n_file_1:
            if is_extension_given:
                li_path = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(ext.lower())]
            else:
                li_path = [os.path.join(dirpath, f) for f in filenames]
            n_file_2 = len(li_path)
            if n_file_2:
                li_path_total += li_path
    return sorted(li_path_total)

def is_this_empty_string(strin):
    return (strin in (None, '')) or (not strin.stip())

def is_image_file(fn): 
    ext = (".bmp", ".ppm", ".png", ".gif", ".jpg", ".jpeg", ".tif", ".pgm")
    return fn.endswith(ext)

def get_list_of_image_path_under_this_directory(dir_img, ext = ''):
    dir_img = os.path.expanduser(dir_img)
    li_fn_img = get_list_of_file_path_under_1st_with_2nd_extension(dir_img, ext)
    if is_this_empty_string(ext):
        li_fn_img = [fn for fn in li_fn_img if is_image_file(fn)]
    return sorted(li_fn_img)

def get_max_and_min_size_of_image_under_this_directory(direc):
    li_path = get_list_of_image_path_under_this_directory(direc)
    n_img = len(li_path)
    w_max, h_max, w_min, h_min = -1, -1, 10000000000000, 100000000000000
    for iI, path in enumerate(li_path):
        if 0 == iI % 1000:
            print('iI : ', iI, ' / ', n_img)   
            print('w_max : ', w_max, ', h_max : ', h_max, ',\t w_min : ', w_min, ', h_min : ', h_min);
        im = cv2.imread(path)
        h, w = im.shape[:2]
        if w < w_min:
            w_min = w
            print('w_min new : ', w_min)
        if h < h_min:
            h_min = h
            print('h_min new : ', h_min)
        if w > w_max:
            w_max = w
            print('w_max new : ', w_max)
        if h > h_max:
            h_max = h
            print('h_max new : ', h_max)
    return w_max, h_max, w_min, h_min


def main():
    '''
    w_max, h_max, w_min, h_min = get_max_and_min_size_of_image_under_this_directory(os.path.join(DATASET_PATH, 'train'))
    print('w_max : ', w_max, ', h_max : ', h_max, ',\t w_min : ', w_min, ', h_min : ', h_min)
    exit(0)
    '''
    n_sp = 0
    #print('DATASET_PATH : ', DATASET_PATH); exit();
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=29)
    #args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    #args.add_argument("--num_epochs", type=int, default=30)
    #args.add_argument("--model_name", type=str, default="1")
    #args.add_argument("--batch", type=int, default=2)
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--prediction_dir", type=str, default="prediction")
    args.add_argument("--print_iter", type=int, default=10)


    args.add_argument('-p', '--project', type=str, default='ai_challenge_2020', help='project file that contains parameters')
    args.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    #args.add_argument('-c', '--compound_coef', type=int, default=2, help='coefficients of efficientdet')
    #parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    args.add_argument('-n', '--num_workers', type=int, default=0, help='num_workers of dataloader')
    #parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    args.add_argument('--batch_size', type=int, default=38, help='The number of images per batch among all devices')
    #args.add_argument('--batch_size', type=int, default=10, help='The number of images per batch among all devices')
    args.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    #parser.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--lr', type=float, default=1e-2)
    args.add_argument('--threshold', type=float, default=0.2)
    args.add_argument('--iou_threshold', type=float, default=0.2)
    args.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    args.add_argument('--num_epochs', type=int, default=500)
    #parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    #parser.add_argument('--val_interval_step', type=int, default=5, help='Number of epoches between valing phases')
    args.add_argument('--dir_pred', type=str, default='prediction', help='directory of predeicton result xml file.')
    #args.add_argument('--n_cls', type=int, default=29, help='Number of classes')
    args.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    #parser.add_argument('--save_interval', type=int, default=1, help='Number of steps between saving')
    args.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    args.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    args.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    args.add_argument('--log_path', type=str, default='logs/')
    #parser.add_argument('-w', '--load_weights', type=str, default=None,
    #args.add_argument('-w', '--load_weights', type=str, default='last',
    args.add_argument('-w', '--model_name', type=str, default='last',
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    args.add_argument('--saved_path', type=str, default=model_dir)
    #parser.add_argument('--debug', type=boolean_string, default=False, help='whether visualize the predicted boxes of training, '
    args.add_argument('--debug', type=boolean_string, default=True, help='whether visualize the predicted boxes of training. The output images will be in test/')
    args.add_argument('--is_mosaic', action='store_true', help='whether split the image into small overlaped sub-images to detect small objects.')
    args.add_argument('--xywh', type=boolean_string, default=False, help='whether bounding box representation is left-top-width-height or not.')
    '''
    for ii in range(100):
        print('random() : ', random.random())
    exit(0);
    '''   
    #t0 = os.path.basename(sys_argv[0]); t1 = os.path.basename(__file__);
    #print('t0 : ', t0); print('t1 : ', t1);
    #exit(0);
    '''
    if os.path.basename(sys_argv[0]) == os.path.basename(__file__):
        #print("called from this py file")
        args = parser.parse_args()
    else:
        #print("called from another py file")
        args = parser.parse_args([])
    '''     
    '''
    if sys_argv is None or 1 == len(sys_argv):
        #print('len(sys_argv) is 1')
        args = parser.parse_args()
        #print('args : ', args); exit()
    else:
        #print('sys_argv : ', sys_argv);   exit()
        args = parser.parse_args(sys_argv)
    '''



    config = args.parse_args()

    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    model_name = config.model_name
    batch = config.batch_size
    mode = config.mode
    is_mosaic = config.is_mosaic
    #print('is_mosaic :', is_mosaic);    exit(0)
    prediction_dir = config.prediction_dir + "_" + mode
    #print('prediction_dir :', prediction_dir);  exit(0);
    print_iter = config.print_iter
    #print('AAA main')
    # 도움 함수를 이용해 모델을 가져옵니다
    #new_model = model.get_model_instance_segmentation(num_classes)
    #opt = train_new.get_args(sys.argv)
    #print('opt : ', opt);   exit(0);
    params = train_new.Params(f'projects/{config.project}.yml')
    print('params.obj_list : ', params.obj_list);   #exit(0);
    #'''
    # 학습을 GPU로 진행하되 GPU가 가용하지 않으면 CPU로 합니다
    device = torch.device('cuda') if cuda else torch.device('cpu')

    #is_xywh = False;

        
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    
    if mode == 'train':
        model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=config.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

   
    
        #'''    
        #check parameter of model
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in model.parameters())
        print("num of parameter :", total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :", trainable_params)
        print("------------------------------------------------------------")
        #exit(0);



        dataset_train = CustomDataset(DATASET_PATH, get_transform(True), config.xywh, 'train')
        dataset_val = CustomDataset(DATASET_PATH, get_transform(False), config.xywh, 'val')
        train_new.train(opt, dataset_train, dataset_val)
        '''        
        # 데이터셋과 정의된 변환들을 사용합니다
        data_loader_train = dataloader.data_loader(DATASET_PATH, batch, phase='train')
        data_loader_val = dataloader.data_loader(DATASET_PATH, 1, phase='val')

        # 옵티마이저(Optimizer)를 만듭니다
        params = [p for p in new_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=base_lr,
                                    momentum=0.9, weight_decay=0.0005)
            
        # 학습률 스케쥴러를 만듭니다
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)
        
        train(new_model, data_loader_train, data_loader_val, device, num_epochs, prediction_dir, print_iter, optimizer=optimizer, lr_scheduler=lr_scheduler)
        '''        
    
    elif 'test' == mode or 'val' == mode:
        #print('model_name : '); print(model_name); exit(0);

        model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=config.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

        #'''    
        #check parameter of model
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in model.parameters())
        print("num of parameter :", total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :", trainable_params)
        print("------------------------------------------------------------")
        #exit(0);




        #model, last_step, weights_path = load_weight_from_file(model, config.model_name, config.saved_path)
        
        #model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
        model.load_state_dict(torch.load(os.path.join(config.saved_path,  'efficientdet-d0_5_45000.pth')))
        model.requires_grad_(False)
        model.eval()
   
    

        if config.cuda:
            model = model.cuda()
        #testset_given = CustomDataset(DATASET_PATH, get_transform(False), config.xywh, 'test')

        #test_params = {'batch_size': config.batch_size,
        #          'shuffle': False,
        #          'drop_last': False,
        #          'collate_fn': collater,
        #          'num_workers': config.num_workers}

        #input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        #test_set = ChallengeDataset(input_side = input_sizes[config.compound_coef], data_given = testset_given, root_dir=os.path.join(config.data_path, params.project_name), set=params.test_set,
        #                  transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
        #                                                 Resizer(input_sizes[config.compound_coef])]))


        #test_generator = DataLoader(test_set, **test_params)
        #data_loader_test = dataloader.data_loader(DATASET_PATH, 1, config.xywh, phase='test')        
        #data_loader_test = dataloader.data_loader(DATASET_PATH, config.batch_size, config.xywh, phase='test')        
        test(model, os.path.join(DATASET_PATH, mode), input_sizes[config.compound_coef], config.threshold, config.iou_threshold, config.cuda, device, prediction_dir, is_mosaic, n_sp + 1)
    print("That's it!")

if __name__ == '__main__' :
    main()
