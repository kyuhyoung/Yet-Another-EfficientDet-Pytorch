# Author: Zylo117

import os

import cv2
import numpy as np
import torch
from glob import glob
from torch import nn
from torchvision.ops import nms
from torchvision.ops.boxes import batched_nms
from typing import Union
import uuid

from utils.sync_batchnorm import SynchronizedBatchNorm2d

from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
import math
import random
#import webcolors

def print_indented(n_sp, *args):
    if n_sp >= 0:
        print('  ' * n_sp, *args)



def invert_affine(metas: Union[float, list, tuple], preds):
    #print('type(preds) :', type(preds));    exit()
    print('len(preds) :', len(preds))   # exit()
    for i in range(len(preds)):
        if len(preds[i]['rois_ltrb']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois_ltrb'][:, [0, 2]] = preds[i]['rois_ltrb'][:, [0, 2]] / metas
                preds[i]['rois_ltrb'][:, [1, 3]] = preds[i]['rois_ltrb'][:, [1, 3]] / metas
            else:
                #print('metas[i] :', metas[i]);  exit()
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois_ltrb'][:, [0, 2]] = (preds[i]['rois_ltrb'][:, [0, 2]] - padding_w) / (new_w / old_w)
                preds[i]['rois_ltrb'][:, [1, 3]] = (preds[i]['rois_ltrb'][:, [1, 3]] - padding_h) / (new_h / old_h)
    return preds


def letterboxing_opencv(image, wh_tgt, letterbox_type, n_sp, only_return_image = True, means = None, interpolation = None):
    '''resize image with unchanged aspect ratio using padding'''
    print_indented(n_sp, 'letterboxing_opencv START');
    #iw, ih = image.shape[0:2][::-1]
    is_color = len(image.shape) > 2
    h_src, w_src = image.shape[:2]
    w_tgt, h_tgt = wh_tgt
    scale = min(w_tgt / w_src, h_tgt / h_src)
    if abs(scale - 1.0) > 1e-5:  
        w_new = int(w_src * scale); h_new = int(h_src * scale)
        if interpolation:
            image = cv2.resize(image, (w_new, h_new), interpolation = interpolation)
        else:     
            image = cv2.resize(image, (w_new, h_new), interpolation = cv2.INTER_CUBIC)
    else:
        w_new = w_src;  h_new = h_src;       
    if 'top_left' == letterbox_type:
        x_offset = 0;   y_offset = 0
        x_padding = w_tgt - w_new;  y_padding = h_tgt - h_new;
    elif 'center' == letterbox_type:
        x_offset = (w_tgt - w_new) // 2;    y_offset = (h_tgt - h_new) // 2
        x_padding = x_offset;               y_padding = y_offset;
    else:
        raise NameError('Invalid letterbox_type')        
    if is_color:
        chn = image.shape[2]
        new_image = np.zeros((h_tgt, w_tgt, chn), np.float)
    else:
        new_image = np.zeros((h_tgt, w_tgt), np.float)
        
    if means:
        #new_image.fill(128)
        new_image[...] = means
    #new_image[dy:dy+nh, dx:dx+nw,:] = image
    new_image[y_offset : y_offset + h_new, x_offset : x_offset + w_new, :] = image
    if only_return_image:
        return new_image
    else:
        return new_image, w_new, h_new, w_src, h_src, x_offset, y_offset 

def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w
    x_offset = 0;   y_offset = 0;
    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    #return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h
    return canvas, new_w, new_h, old_w, old_h, x_offset, y_offset


def preprocess(*image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
        means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


#def preprocess_video(*frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
def preprocess_video(ori_imgs, letterbox_type, n_sp, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    
    #for iI, ori_img in enumerate(ori_imgs):
    #    print('iI :', iI, ' / ', len(ori_imgs), ', ori_img.shape :', ori_img.shape);
    #exit()
    ##  for non-mosaicking mode, original 1920 x 1080 image is just passed from frame capture of cv2.
    ##   iI : 0 / 1 , ori_img.shape : (1080, 1920, 3) 
    ##
    ##  for mosaicking mode, original 1920 x 1080 image is divided into 8 tiles thru dataloader. Each tile is 600 x 600 and the last (9th) image is the whole 1920 x 1080 image.  
    ##   iI : 0 / 9 , ori_img.shape : (600, 600, 3) 
    ##   iI : 1 / 9 , ori_img.shape : (600, 600, 3) 
    ##   iI : 2 / 9 , ori_img.shape : (600, 600, 3) 
    ##   iI : 3 / 9 , ori_img.shape : (600, 600, 3) 
    ##   iI : 4 / 9 , ori_img.shape : (600, 600, 3) 
    ##   iI : 5 / 9 , ori_img.shape : (600, 600, 3) 
    ##   iI : 6 / 9 , ori_img.shape : (600, 600, 3) 
    ##   iI : 7 / 9 , ori_img.shape : (600, 600, 3) 
    ##   iI : 8 / 9 , ori_img.shape : (1080, 1920, 3) 
    print_indented(n_sp, 'preprocess_video START')
    #print('type(ori_imgs[0]) :', type(ori_imgs[0]));  exit()
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    #imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size, means=None) for img in normalized_imgs]
    imgs_meta = [letterboxing_opencv(img[..., ::-1], (max_size, max_size), letterbox_type, n_sp + 1, only_return_image = False) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]
    print_indented(n_sp, 'preprocess_video END')

    return ori_imgs, framed_imgs, framed_metas
'''
def xyxy_2_ltwh(xyxy):
    ltwh = xyxy
    ltwh[..., 2] = xyxy[..., 2] - xyxy[..., 0]
    ltwh[..., 3] = xyxy[..., 3] - xyxy[..., 1]
    return ltwh
'''

def xyxy_2_ltwh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        if 2 < xyxy.ndim: 
            shape_ori = xyxy.shape
            n_box = np.prod(shape_ori[:-1])
            xyxy = np.reshape(xyxy, (n_box, shape_ori[-1]))  
            t0 = np.hstack((xyxy[:, 0 : 2], xyxy[:, 2 : 4] - xyxy[:, 0 : 2] + 1))
            return np.reshape(t0, shape_ori)
        else:        
            return np.hstack((xyxy[..., 0 : 2], xyxy[..., 2 : 4] - xyxy[..., 0 : 2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def scale_bbox(ltrb, gain, wh_src, wh_tgt, letterbox_type, bbox_type, n_sp):
    print_indented(n_sp, 'scale_bbox START')
    if 'center' == letterbox_type:
        ltrb[:, 0] -= (wh_src[0] - wh_tgt[0] * gain) / 2  # x padding
        ltrb[:, 1] -= (wh_src[1] - wh_tgt[1] * gain) / 2  # y padding
        if 'ltrb' == bbox_type:
            ltrb[:, 2] -= (wh_src[0] - wh_tgt[0] * gain) / 2  # x padding
            ltrb[:, 3] -= (wh_src[1] - wh_tgt[1] * gain) / 2  # y padding
    elif not('top_left' == letterbox_type):
        raise NameError('invalid letterbox type')
    ltrb[:, :4] /= gain
    ltrb[:, [0, 2]] = ltrb[:, [0, 2]].clamp(min = 0, max = wh_tgt[0] - 1)
    ltrb[:, [1, 3]] = ltrb[:, [1, 3]].clamp(min = 0, max = wh_tgt[1] - 1)
    print_indented(n_sp, 'scale_bbox END')
    return ltrb

def letterbox_bbox_2_ori_bbox(bbox_letterbox, wh_letterbox, wh_ori, letterbox_type, bbox_type, xy_offset, n_sp):
    print_indented(n_sp, 'letterbox_bbox_2_ori_bbox START')
    wh_src = wh_letterbox;  wh_tgt = wh_ori;
    gain = min(wh_src[0] / wh_tgt[0], wh_src[1] / wh_tgt[1])
    bbox_ori = scale_bbox(bbox_letterbox, gain, wh_src, wh_tgt, letterbox_type, bbox_type, n_sp + 1)
    #bbox_ori[:, :2] += xy_offset[:] 
    bbox_ori[:, 0] += xy_offset[0]; bbox_ori[:, 1] += xy_offset[1] 
    if 'ltrb' == bbox_type:
        #bbox_ori[:, 2:4] += xy_offset[:]; 
        bbox_ori[:, 2] += xy_offset[0]; bbox_ori[:, 3] += xy_offset[1]; 
    print_indented(n_sp, 'letterbox_bbox_2_ori_bbox END')
    return bbox_ori

def ori_bbox_2_letterbox_bbox(bbox_ori, wh_ori, wh_letterbox, letterbox_type, n_sp):
    print_indented(n_sp, 'ori_bbox_2_letterbox_bbox START')
    wh_src = wh_ori;    wh_tgt = wh_letterbox;
    gain = max(wh_src[0] / wh_tgt[0], wh_src[1] / wh_tgt[1])
    bbox_letterbox = scale_bbox_ltrb(bbox_ori, gain, wh_src, wh_tgt, letterbox_type, bbox_type, n_sp + 1)
    print_indented(n_sp, 'ori_bbox_2_letterbox_bbox END')
    return bbox_letterbox

'''
def ori_bbox_2_letterbox_bbox_ltrb(ltrb, wh_src, wh_tgt, letterbox_type) : #wh_img1, coords_xyxy, wh_img0):
    # Rescale coords1 (xyxy) from img1_shape to img0_shape
    #gain = max(wh_img1) / max(wh_img0)  # gain  = old / new
    if 'top_left' == letterbox_type:
        gain = max(wh_tgt[0] / wh_src[0], wh_tgt[1] / wh_src[1])
        ltrb[:, :4] *= gain
    elif 'center' == letterbox_type: 
        gain = max(wh_src) / max(wh_tgt)  # gain  = old / new
        #print('wh_img0 : ', wh_img0);     print('wh_img1 : ', wh_img1); #exit()
    #coords_xyxy[:, [0, 2]] -= (wh_img1[0] - wh_img0[0] * gain) / 2  # x padding
        ltrb[:, [0, 2]] -= (wh_src[0] - wh_tgt[0] * gain) / 2  # x padding
        ltrb[:, [1, 3]] -= (wh_src[1] - wh_tgt[1] * gain) / 2  # y padding
        ltrb[:, :4] /= gain
        #ltrb[:, :4] = ltrb[:, :4].clamp(min = 0)
    else:
        raise NameError('invalid letterbox type')
    ltrb[:, [0, 2]] = ltrb[:, [0, 2]].clamp(min = 0, max = wh_tgt[0] - 1)
    ltrb[:, [1, 3]] = ltrb[:, [1, 3]].clamp(min = 0, max = wh_tgt[1] - 1)
    return ltrb

def ori_bbox_2_letterbox_bbox_ltwh(ltwh, wh_src, wh_tgt, letterbox_type):# wh_img1, coords_ltwh, wh_img0):
    # Rescale coords1 (xyxy) from img1_shape to img0_shape
    if 'top_left' == letterbox_type:
        gain = max(wh_tgt[0] / wh_src[0], wh_tgt[1] / wh_src[1])
        ltwh[:, :4] *= gain
    elif 'center' == letterbox_type: 
        gain = max(wh_src) / max(wh_tgt)  # gain  = old / new
        ltwh[:, : 2] -= (wh_src - wh_tgt * gain) / 2  # x padding
        #coords_ltwh[:, : 2] -= (wh_img1[0] - wh_img0[0] * gain) / 2  # x padding
        #coords[:, :2] -= (img1_shape[1] - img0_shape[1] * gain) / 2  # x padding
        #coords[:, 2:4] -= (img1_shape[0] - img0_shape[0] * gain) / 2  # y padding
        ltwh[:, :4] /= gain
        ltwh[:, :4] = ltwh[:, :4].clamp(min = 0)
    else:
        raise NameError('invalid letterbox type')
    ltwh[:, [0, 2]] = ltwh[:, [0, 2]].clamp(min = 0, max = wh_tgt[0] - 1)
    ltwh[:, [1, 3]] = ltwh[:, [1, 3]].clamp(min = 0, max = wh_tgt[1] - 1)

    return ltwh

#def letterbox_2_li_ltwh_ori(torch_tensor_det_xyxy_letterbox, wh_ori, wh_letterbox, wh_frame):
#def letterboxing(bbox_ori, wh_ori, wh_letterbox, type_bbox_ori, type_bbox_letterbox, n_sp):
#    return letterbox_bbox_2_ori_bbox(bbox_ori, wh_ori, wh_letterbox, type_bbox_ori, type_bbox_letterbox, (0, 0), n_sp) 

def letterbox_bbox_2_ori_bbox(bbox_letterbox, wh_letterbox, wh_ori, type_bbox_letterbox, type_bbox_ori, xy_offset_ori, letterbox_type, n_sp):
    #li_ltwh_ori = []
    #li_bbox_ori = []
    #ltrbbbox_ori = bbox_letterbox.clone()
    print_indented(n_sp, 'letterbox_bbox_2_ori_bbox START')
    if 'ltwh' == type_bbox_letterbox:
        ltrb_letterbox = ltwh_2_ltrb(bbox_letterbox)
    elif 'xywh' == type_bbox_letterbox:
        ltrb_letterbox = xywh_2_ltrb(bbox_letterbox)
    else:
        ltrb_letterbox = bbox_letterbox.copy()
    ltrb_ori = scale_bbox_ltrb(ltrb_letterbox, wh_letterbox, wh_ori, letterbox_type)
    for ii in range(2):
        ltrb_ori[:, ii + 0] += xy_offset_ori[ii];  
        ltrb_ori[:, ii + 2] += xy_offset_ori[ii];  
    #ltrb_ori[:, [0, 2]] += xy_offset_ori[0];  
    #ltrb_ori[:, [1, 3]] += xy_offset_ori[1];

    if 'ltwh' == type_bbox_ori:
        bbox_ori = ltrb_2_ltwh(ltrb_ori)
    elif 'xywh' == type_bbox_ori:
        bbox_ori = ltrb_2_xywh(ltrb_ori)
    else:
        bbox_ori = ltrb_ori.clone()
    print_indented(n_sp, 'letterbox_bbox_2_ori_bbox END')
    return bbox_ori
'''

def merge_divided_detections(det_batch, idx_bbox_from, li_offset_xy, is_whole_included, wh_letterbox, wh_tile, wh_ori, letterbox_type, bbox_type, n_sp):
    '''
    bbox_type : one of 'ltrb' / 'ltwh' / 'xywh'
         for image_i, pred in enumerate(li_det):
                 pred[:, 0] += li_offset_xy[image_i][0]; 
                         pred[:, 1] += li_offset_xy[image_i][1];
                             '''
                                 #im_bgr_resized = cv2.resize(im_bgr_hwc_ori_np, None, fx = ratio_resize, fy = ratio_resize)
                                     
    print_indented(n_sp, 'merge_divided_detections START')
    print_indented(n_sp + 1, 'is_whole_included :', is_whole_included);  #exit()
    print_indented(n_sp + 1, 'li_offset_xy.shape :', li_offset_xy.shape);  #exit()
    print_indented(n_sp + 1, 'type(li_offset_xy) :', type(li_offset_xy));  #exit()
    print_indented(n_sp + 1, 'det_batch.shape :', det_batch.shape);  #exit()
    print_indented(n_sp + 1, 'bbox_type :', bbox_type);  #exit()
    #print_indented(n_sp + 1, 'type(im_bgr_hwc_ori_np) :', type(im_bgr_hwc_ori_np)); #exit()
    if isinstance(det_batch, torch.Tensor):
        li_offset_xy = torch.from_numpy(li_offset_xy).to(det_batch)
    for image_i, offset_xy in enumerate(li_offset_xy):
        print_indented(n_sp + 2, 'image_i :', image_i, ' / ', len(li_offset_xy))
        #print('offset_xy.shape :', offset_xy.shape);     #exit()
        print_indented(n_sp + 3, 'offset_xy :', offset_xy);     #exit()
        #offset_xy[0] = 100; offset_xy[1] = 1000;
        #print('offset_xy after :', offset_xy);     #exit()
        print_indented(n_sp + 3, 'det_batch[image_i, 1, :4] b4 :', det_batch[image_i, 1, :4])
        det_batch[image_i, :, idx_bbox_from : idx_bbox_from + 4] = letterbox_bbox_2_ori_bbox(det_batch[image_i, :, idx_bbox_from : idx_bbox_from + 4], wh_letterbox, wh_tile, letterbox_type, bbox_type, offset_xy, n_sp + 3)
        #det_batch[image_i, :, idx_bbox_from : idx_bbox_from + 2] += offset_xy[:]
        #if 'ltrb' == bbox_type:
        #    det_batch[image_i, :, idx_bbox_from + 2 : idx_bbox_from + 4] += offset_xy[:]
        print_indented(n_sp + 3, 'det_batch[image_i, 1, :4] after :', det_batch[image_i, 1, :4])
        print_indented(n_sp + 3, 'offset_xy :', offset_xy)
        #if offset_xy[0] and offset_xy[1]:
        #    exit()
        #print('AAA');   exit()
        #for xy in range(2):     
        #    det_batch[image_i, :, xy] += offset_xy[xy]
        '''
            if is_whole_included:
                ltrb = xywh2xyxy(li_det[image_i, :, :4])
                for iR in range(len(ltrb)):
                    if li_det[image_i, iR, 4] < 0.6: continue
                    cv2.rectangle(im_bgr_resized, (ltrb[iR, 0], ltrb[iR, 1]), (ltrb[iR, 2], ltrb[iR, 3]), (0, 0, 255))  
       '''         
    #exit()
    if is_whole_included:
        bbox_letterbox = det_batch[-1, :, idx_bbox_from : idx_bbox_from + 4]
        #wh_ori = (im_bgr_hwc_ori_np.shape[1], im_bgr_hwc_ori_np.shape[0])
        bbox_ori = letterbox_bbox_2_ori_bbox(bbox_letterbox, wh_letterbox, wh_ori, letterbox_type, bbox_type, (0, 0), n_sp + 2)
        print_indented(n_sp + 3, 'det_batch[-1, 1, :4] b4 :', det_batch[-1, 1, :4])
        det_batch[-1, :, idx_bbox_from : idx_bbox_from + 4] = bbox_ori
        print_indented(n_sp + 3, 'det_batch[-1, 1, :4] after :', det_batch[-1, 1, :4])
        print_indented(n_sp + 3, 'det_batch[len(li_offset_xy) - 1, 1, :4] after :', det_batch[len(li_offset_xy) -1, 1, :4])
    #exit()     
    n_batch, n_det, n_attribute = tuple(det_batch.size())
    n_det_total = n_batch * n_det;
    #print('n_batch : ', n_batch);   print('n_det : ', n_det);   print('n_attribute : ', n_attribute);   print('n_det_total : ', n_det_total);  #exit();
    #det = np.vstack(li_det)
    det_merged = det_batch.view(-1, n_attribute);
    print('det_merged.size() : ', det_merged.size()); #exit()
    det_merged.unsqueeze_(0)
    return det_merged
                                                                                                                                                                                                                                                                                                                                                                                                                
def xywh_2_ltrb(xywh):          
    ltrb = xywh.new(xywh.shape)
    ltrb[..., 0] = xywh[..., 0] - xywh[..., 2] / 2.0
    ltrb[..., 1] = xywh[..., 1] - xywh[..., 3] / 2.0
    ltrb[..., 2] = xywh[..., 0] + xywh[..., 2] / 2.0
    ltrb[..., 3] = xywh[..., 1] + xywh[..., 3] / 2.0
    return ltrb


def bbox_iou(box1, box2, bbox_type, n_sp):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    print_indented(n_sp, 'bbox_iou START')
    #print_indented(n_sp + 1, 'box1.shape :', box1.shape);   #exit();
    #print_indented(n_sp + 1, 'box2.shape :', box2.shape);   #exit();
    ##   box1.shape : torch.Size([7])
    ##   box2.shape : torch.Size([100, 7])
    box2 = box2.t()
    # Get the coordinates of bounding boxes
    if 'ltrb' == bbox_type:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    elif 'xywh' == bbox_type:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    elif 'ltwh' == bbox_type:
        b1_x1, b1_y1 = box1[0], box1[1]
        b1_x2 = b1_x1 + box1[2];    b1_y2 = b1_y1 + box1[3]
        b2_x1, b2_y1 = box2[0], box2[1]
        b2_x2 = b2_x1 + box2[2];    b2_y2 = b2_y1 + box2[3]
    else:
        raise NameError('invalid bbox_type')   
    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    #inter_area = (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)).clamp(0) * (min(b1_y2, b2_y2) - max(b1_y1, b2_y1)).clamp(0)
    #print_indented(n_sp + 1, 'inter_area.shape :', inter_area.shape);   #exit();
    ##   inter_area.shape : torch.Size([100])
    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area
    print_indented(n_sp, 'bbox_iou END')
    return inter_area / union_area  # iou


def bbox_ios(box_self, box_other, bbox_type, n_sp):
    #print('type(box1) : ', type(box1));     print('type(box2) : ', type(box2)); #exit()
    #print('box1.shape : ', box1.shape);     print('box2.shape : ', box2.shape); #exit()
    # Returns the intersecion over self_area of box_self wrt box_other. box_self is 4, box_other is nx4
    print_indented(n_sp, 'bbox_ios START')
    print_indented(n_sp + 1, 'box_self.shape :', box_self.shape);   #exit();
    print_indented(n_sp + 1, 'box_other.shape :', box_other.shape);   #exit();
    print_indented(n_sp + 1, 'bbox_type :', bbox_type);   #exit();

    box_other = box_other.t()
    # Get the coordinates of bounding boxes
    if 'ltrb' == bbox_type:
        # x1, y1, x2, y2 = box1
        self_x1, self_y1, self_x2, self_y2 = box_self[0], box_self[1], box_self[2], box_self[3]
        other_x1, other_y1, other_x2, other_y2 = box_other[0], box_other[1], box_other[2], box_other[3]
    elif 'xywh' == bbox_type:
        # x, y, w, h = box1
        self_x1, self_x2 = box_self[0] - box_self[2] / 2, box_self[0] + box_self[2] / 2
        self_y1, self_y2 = box_self[1] - box_self[3] / 2, box_self[1] + box_self[3] / 2
        other_x1, other_x2 = box_other[0] - box_other[2] / 2, box_other[0] + box_other[2] / 2
        other_y1, other_y2 = box_other[1] - box_other[3] / 2, box_other[1] + box_other[3] / 2
    
    elif 'ltwh' == bbox_type:
        self_x1, self_y1 = box_self[0], box_self[1]
        self_x2 = self_x1 + box_self[2];    self_y2 = self_y1 + box_self[3]
        other_x1, other_y1 = box_other[0], box_other[1]
        other_x2 = other_x1 + box_other[2]; other_y2 = other_y1 + box_other[3]
    else:
        raise NameError('invalid bbox_type')   
   
    # Intersection area
    #inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    inter_area = (min(self_x2, other_x2) - torch.max(self_x1, other_x1)).clamp(0) * (min(self_y2, other_y2) - max(self_y1, other_y1)).clamp(0)
    print_indented(n_sp + 1, 'inter_area.shape :', inter_area.shape);   #exit();
    # Self Area
    self_area = (self_x2 - self_x1) * (self_y2 - self_y1) + 1e-16
    print_indented(n_sp, 'bbox_ios END')
    return inter_area / self_area  # ios



def compensate_division(li_ltrb_c_cc, li_group, li_str_class, ios_threshold, n_sp): # li_ltrb_c_cc : list of torch 2D       tensor
    print_indented(n_sp, 'compensate_division START')
    li_ltrb_c_cc_filtered = []
    #for ii in range(len(li_ltrb_c_cc)):
    #    print_indented(n_sp + 1, 'ii :', ii, ', li_ltrb_c_cc[ii].shape : ', li_ltrb_c_cc[ii].shape);
    li_id = [int(lrtb_c_cc[0, -1]) for lrtb_c_cc in li_ltrb_c_cc]
    li_id_unique = list(set(li_id))
    li_id_very_unique = []
    li_li_str_unique = []
    for id_unique in li_id_unique:
        str_cls = li_str_class[id_unique]
        is_new = True
        for li_str_unique in li_li_str_unique:
            if str_cls in li_str_unique:
                is_new = False; break;
        if is_new:
            li_id_very_unique.append(id_unique)
            gr = [str_cls]
            for group in li_group:
                if str_cls in group:
                    gr = group; break;
            li_li_str_unique.append(gr)
    for id_very_unique in li_id_very_unique:
        str_cls = li_str_class[id_very_unique]
        gr = [str_cls]
        for group in li_group:
            if str_cls in group:
                gr = group; break;
        li_ltrb_c_cc_same_cls = [ltrb_c_cc for ltrb_c_cc in li_ltrb_c_cc if li_str_class[int(ltrb_c_cc[0, -1])]      in gr]
        if 1 >= len(li_ltrb_c_cc_same_cls):
            li_ltrb_c_cc_filtered.append(li_ltrb_c_cc_same_cls[0]);   continue
        for i0, ltrb_c_cc_same_cls_0 in enumerate(li_ltrb_c_cc_same_cls):
            is_too_included = False
            for i1, ltrb_c_cc_same_cls_1 in enumerate(li_ltrb_c_cc_same_cls):
                print_indented(n_sp + 1, 'i1 :', i1, ' / ', len(li_ltrb_c_cc_same_cls))
                if i1 == i0: continue
                #ios = float(bbox_ios(torch.squeeze(ltrb_c_cc_same_cls_0), ltrb_c_cc_same_cls_1, 'ltrb', n_sp + 2))
                ios = float(bbox_ios(torch.squeeze(ltrb_c_cc_same_cls_0), ltrb_c_cc_same_cls_1, 'ltrb', -100))
                if ios > ios_threshold:
                    is_too_included = True; break;
            if not is_too_included:
                li_ltrb_c_cc_filtered.append(ltrb_c_cc_same_cls_0)
    print_indented(n_sp + 1, 'len(li_ltrb_c_cc) : ', len(li_ltrb_c_cc))
    print_indented(n_sp + 1, 'len(li_ltrb_c_cc_filtered) : ', len(li_ltrb_c_cc_filtered));   #exit()
    print_indented(n_sp, 'compensate_division END')
    return li_ltrb_c_cc_filtered



def non_max_suppression_4_mosaic(pred_xywh_c_cc, li_offset_xy, include_original, li_group, wh_net_input, wh_tile, wh_ori, li_str_class, ios_threshold, letterbox_type, bbox_type, n_sp, conf_thres=0.5, nms_thres=0.5): #pred_letterbox.type() : torch.cuda.FloatTensor

    """
    Input :
            bbox_type : one of 'ltrb' / 'ltwh' / 'xywh'
                pred_xywh_c_cc : tensor. shape (batch_size x N x 85 for yolov3)  85 (= x + y + w + h + 1 objectness + 80 class-confidences for yolov3)
                        li_offset_xy : ndarray.
                            Removes detections with lower object confidence score than 'conf_thres'
                                Non-Maximum Suppression to further filter detections.
                                    Returns detections with shape:
                                            (x1, y1, x2, y2, object_conf, class_conf, class)
                                                """
                                                    #print('type(li_offset_xy) : ', type(li_offset_xy))
                                                        #print('pred_xywh_c_cc.shape :', pred_xywh_c_cc.shape);  exit();
                                                            ##  type(li_offset_xy) :  <class 'numpy.ndarray'>
                                                                ##  pred_xywh_c_cc.shape : torch.Size([3, 5415, 85])
                                                                    
    #if 0 != li_offset_xy.size:
    #print('pred_xywh_c_cc.shape b4 :', pred_xywh_c_cc.shape);  exit();
    #t0 = pred_xywh_c_cc[-1].unsqueeze(0);   print('t0.shape :', t0.shape);  exit();
    #print_indented(n_sp, 'non_max_suppression_4_mosaic START')
    if li_offset_xy is not None and pred_xywh_c_cc.shape[0] > 1:
        
        #pred_xywh_c_cc = merge_divided_detections(pred_xywh_c_cc[-1].unsqueeze(0), 0, np.empty(shape=[0, 0]), include_original, wh_net_input, im_bgr_hwc_ori_np, letterbox_type, bbox_type, n_sp + 1) 
        pred_xywh_c_cc = merge_divided_detections(pred_xywh_c_cc, 0, li_offset_xy, include_original, wh_net_input, wh_tile, wh_ori, letterbox_type, bbox_type, n_sp + 1) 
    #print('pred_xywh_c_cc.shape after :', pred_xywh_c_cc.shape);  exit();
    min_wh = 2  # (pixels) minimum box width and height
    output = [None] * len(pred_xywh_c_cc)
    for image_i, xywh_c_cc in enumerate(pred_xywh_c_cc):
        print_indented(n_sp + 1, 'image_i :', image_i)
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])
        #print('xywh_c_cc[:, 4].max() :', xywh_c_cc[:, 4].max()
        #print('xywh_c_cc[:, 4].argmax() :', xywh_c_cc[:, 4].argmax())
        #print('xywh_c_cc[xywh_c_cc[:, 4].argmax(), 4] :', xywh_c_cc[xywh_c_cc[:, 4].argmax(), 4]);   exit()
        ##  xywh_c_cc[:, 4].max() : tensor(0.99119, device='cuda:0')
        ##  xywh_c_cc[:, 4].argmax() : tensor(3382, device='cuda:0')
        ##  xywh_c_cc[xywh_c_cc[:, 4].argmax(), 4] : tensor(0.99119, device='cuda:0')
        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = xywh_c_cc[:, 5:].max(1)
        #print('xywh_c_cc.shape :', xywh_c_cc.shape);
        #print('class_conf.shape :', class_conf.shape);
        #print('class_pred.shape :', class_pred.shape);  exit();
        
        ##  ixywh_c_cc.shape : torch.Size([16245, 85])  #   16245 = 3 x 5415

        ##  class_conf.shape : torch.Size([16245])
        ##  class_pred.shape : torch.Size([16245])
        
        print_indented(n_sp + 2, 'xywh_c_cc[:, 4].max() b4 :', xywh_c_cc[:, 4].max());
        xywh_c_cc[:, 4] *= class_conf
        print_indented(n_sp + 2, 'xywh_c_cc[:, 4].max() after :', xywh_c_cc[:, 4].max());
        print_indented(n_sp + 2, 'conf_thres :', conf_thres);
        
        # Select only suitable predictions
        i = (xywh_c_cc[:, 4] > conf_thres) & (xywh_c_cc[:, 2:4] > min_wh).all(1) & torch.isfinite(xywh_c_cc).all(1)
        print_indented(n_sp + 2, 'class_conf :', class_conf);
        print_indented(n_sp + 2, 'len(xywh_c_cc) b4 :', len(xywh_c_cc))
        xywh_c_cc = xywh_c_cc[i]
        print_indented(n_sp + 2, 'len(xywh_c_cc) after :', len(xywh_c_cc))
        #exit()
        # If none are remaining => process next image
        if len(xywh_c_cc) == 0:
            #print('None are remaining');    exit()
            continue
            
        #print('Some are remaining');    exit()
        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        ltrb_c_cc = xywh_c_cc.clone()
        #ltrb_c_cc[:, :4] = xywh2xyxy(xywh_c_cc[:, :4])
        ltrb_c_cc[:, :4] = xywh_2_ltrb(xywh_c_cc[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551
        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        #pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)
        ltrb_c_cc = torch.cat((ltrb_c_cc[:, :5], class_conf.unsqueeze(1), class_pred), 1)
        
        # Get detections sorted by decreasing confidence scores
        #pred = pred[(-pred[:, 4]).argsort()]
        ltrb_c_cc = ltrb_c_cc[(-ltrb_c_cc[:, 4]).argsort()]
        
        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        #for c in pred[:, -1].unique():
        for c in ltrb_c_cc[:, -1].unique():
            dc = ltrb_c_cc[ltrb_c_cc[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 100:
                dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117
                
            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind], 'ltrb') > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]
                
                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:], 'ltrb', n_sp + 3)  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold
                    
            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:], 'ltrb', n_sp + 3)  # iou with other boxes
                    if iou.max() > 0.5:                            
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold
                        
            elif nms_style == 'MERGE':  # weighted mixture box
                #print('len(dc) :', len(dc));    exit();
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc, 'ltrb', n_sp + 3) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]
            elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:], 'ltrb', n_sp + 3)  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
        
        print_indented(n_sp + 2, 'det_max[0].shape :', det_max[0].shape);   # exit();
        print_indented(n_sp + 2, 'len(det_max) :', len(det_max));    #exit();
        if include_original:
            print_indented(n_sp + 3, 'len(det_max) b4 :', len(det_max));    #exit();
            #print_indented(n_sp + 3, 'ios_threshold :', ios_threshold);    exit();
            det_max = compensate_division(det_max, li_group, li_str_class, ios_threshold, n_sp + 3)
            print_indented(n_sp + 3, 'len(det_max) after :', len(det_max));    #exit();
        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort
    #exit()

    print_indented(n_sp, 'non_max_suppression_4_mosaic END')
    return output



def ltrb_2_xywh(ltrb):
    xywh = ltrb.new(ltrb.shape)
    xywh[..., 0] = (ltrb[..., 0] + ltrb[..., 2]) / 2.0
    xywh[..., 1] = (ltrb[..., 1] + ltrb[..., 3]) / 2.0
    xywh[..., 2] = ltrb[..., 2] - ltrb[..., 0]
    xywh[..., 3] = ltrb[..., 3] - ltrb[..., 1]
    return xywh


def postprocess(x, anchors_tlbr, regression_yxhw, classification, regressBoxes, clipBoxes, threshold, iou_threshold, ios_threshold, n_sp, letterbox_type, wh_ori, wh_tile = None, li_offset_xy = None, include_original = None, li_group = None, li_str_class = None):
    print_indented(n_sp, "postprocess START")
    is_mosaic = li_offset_xy is not None
    batch_size, n_chn_input, h_input, w_input = x.shape
    wh_net_input = (w_input, h_input);  
    print_indented(n_sp + 1, 'wh_net_input :', wh_net_input);   #exit()
    print_indented(n_sp + 1, 'x.shape :', x.shape)
    print_indented(n_sp + 1, 'type(anchors_tlbr) :', type(anchors_tlbr))
    print_indented(n_sp + 1, 'anchors.shape :', anchors_tlbr.shape)
    print_indented(n_sp + 1, 'type(regression_yxhw) :', type(regression_yxhw))
    print_indented(n_sp + 1, 'regression_yxhw.shape :', regression_yxhw.shape)
    print_indented(n_sp + 1, 'type(classification) :', type(classification))
    print_indented(n_sp + 1, 'type(regressBoxes) : ', type(regressBoxes))
    print_indented(n_sp + 1, 'type(clipBoxes) :', type(clipBoxes)); #exit(); 
    # x.shape : torch.Size([1, 3, 512, 512])
    # anchor.shape : torch.Size(1, 49104, 4])
    # regressioin.shape : torch.Size(1, 49104, 4])
    transformed_anchors_ltrb = regressBoxes(anchors_tlbr, regression_yxhw)
    transformed_anchors_ltrb = clipBoxes(transformed_anchors_ltrb, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    print_indented(n_sp + 1, 'transformed_anchors_ltrb.shape :', transformed_anchors_ltrb.shape)
    print_indented(n_sp + 1, 'scores.shape :', scores.shape)
    print_indented(n_sp + 1, 'classification.shape :', classification.shape); #exit(); 
    ##  transformed_anchors_ltrb.shape : torch.Size([1, 49104, 4])
    ##  scores.shape : torch.Size(1, 49104, 1])
    ##  scores_over_thresh.shape : torch.Size(1, 49104])
    ##  classification.shape : torch.Size(1, 49104, 28])    #   28 classes for AI_challenge_2020

    #out = []
    li_di_ltrb_cls_score = []
    if is_mosaic:
        xywh = ltrb_2_xywh(transformed_anchors_ltrb)
        bbox_type = 'xywh'

        print_indented(n_sp + 1, 'xywh.shape :', xywh.shape);
        '''
        for iD in range(anchors.shape[1]):
            #if anchors[0, iD, 2] < 0:
            if anchors[0, iD, 0] < 0 and anchors[0, iD, 1] < 0:
                print('\niD :', iD)
                print('anchors[0, iD, :]', anchors[0, iD, :]);   
                print('transformed_anchors[0, iD, :]', transformed_anchors[0, iD, :]);   
                print('xywh[0, iD, :]', xywh[0, iD, :]);
        '''

        #print('\nanchors_tlbr[0, 49005, :] :', anchors_tlbr[0, 49005, :]);   
        #print('transformed_anchors_ltrb[0, 49005, :] :', transformed_anchors_ltrb[0, 49005, :]);   
        #print('xywh[0, 49005, :] :', xywh[0, 49005, :]);
        #print('scores[0, 49005, :] :', scores[0, 49005, :]);
        #print('scores.argmax() :', scores.argmax())
        #print(/'scores.max() :', scores.max())
        #print('scores.min() :', scores.min())
        #print('threshold :', threshold)
        #print('scores[0, scores.argmax(), :] :', scores[0, scores.argmax(), :]);
        #print('classification[0, scores.argmax(), :] :', classification[0, scores.argmax(), :]);
        #print('classification[0, 49005, :] :', classification[0, 49005, :]);
        #print('classification[0, scores.argmax(), :].sum() :', classification[0, scores.argmax(), :].sum());
        #print('classification[0, 49005, :].sum() :', classification[0, 49005, :].sum());
        ##  anchors_tlbr[0, 49005, :] : tensor([-64., -64., 448., 448.], device='cuda:0')
        ##  transformed_anchors_ltrb[0, 49005, :] : tensor([0.000, 0.000, 382.0305, 286.9348], device='cuda:0')
        ##  xywh[0, 49005, :] : tensor([191.0152, 143.4674 , 382.0305, 286.9348], device='cuda:0')
        ##  scores[0, 49005, :] : tensor([0.0002],  device='cuda:0')
        ##  scores.argmax() : tensor(19520, device='cuda:0')
        ##  scores.max() : tensor(0.2197,  device='cuda:0')
        ##  scores.min() : tensor(1.79799e-05,  device='cuda:0')
        ##  threshold : 0.05
        ##  scores[0, scores.argmax(), :] : tensor([0.2197], device='cuda:0')
        ##  classification[0, scores.argmax(), :] : tensor([0.0128, 0.0173, 0.0463, 0.0019, 0.0073, 0.0092, 0.2197, 0.0314, 0.0291, 0.0177, 0.0105, 0.0037, 0.0516, 0.0058, 0.0269, 0.0357, 0.0289, 0.0269, 0.0374, 0.0161, 0.0063, 0.0085, 0.0298, 0.0219, 0.0330, 0.0122, 0.0645, 0.0228], device='cuda:0')
        ## classification[0, 49005, :] : tensor([5.4680e-06, 1.4904e-04, 6.0246e-06, 5.5823e-10, 1.3952e-09, 8.1008e-05, 7.0045e-07, 4.5584e-06, 3.5451e-07, 4.3533e-07, 2.2724e-04, 1.0260e-06, 5.2864e-07, 1.2785e-06, 1.9779e-07, 7.5457e-08, 2.3940e-07, 3.7211e-10, 1.6166e-08, 6.5973e-07, 2.0706e-06, 7.1672e-07, 3.4279e-07, 2.4868e-07, 3.3471e-07, 6.8379e-07, 8.0108e-08, 8.5626e-06], device='cuda:0')
        ##  classification[0, scores.argmax(), :].sum() : tensor(0.8354, device='cuda:0')
        ##  classification[0, 49005, :].sum() : tensor(0.0005, device='cuda:0')
        #pred_xywh_c_cc = torch.dstack(xywh, scores, classification)
        pred_xywh_c_cc = torch.cat([xywh, scores, classification], axis = 2)
        #print('pred_xywh_c_cc.shape :', pred_xywh_c_cc.shape);  exit()
        ##  pred_xywh_c_cc.shape : torch.Size([1, 49104, 33])   #   33 = 4 + 1 + 28
        #print_indented(n_sp + 1, 'type(im_bgr_hwc_ori_np) :', type(im_bgr_hwc_ori_np)); exit()
        li_ltrb_c_cc = non_max_suppression_4_mosaic(pred_xywh_c_cc, li_offset_xy, include_original, li_group, wh_net_input, wh_tile, wh_ori, li_str_class, ios_threshold, letterbox_type, bbox_type, n_sp + 1, threshold, iou_threshold);
        #print('anchors_nms_idx.shape :', anchors_nms_idx.shape);   exit();
        #print_indented(n_sp + 1, 'li_ltrb_c_cc[0].shape :', li_ltrb_c_cc[0].shape);   exit();
        #   anchors_nms_idx[0].shape : torch.Size([6, 7])
        print_indented(n_sp + 1, 'len(li_ltrb_c_cc) :', len(li_ltrb_c_cc));   #exit();
        #   len(anchors_nms_idx) : 1
        for ltrb_c_cc in li_ltrb_c_cc:
            if 0 != ltrb_c_cc.shape[0]:
                classification_per = ltrb_c_cc[:, 5 : ].permute(1, 0)
                scores_, classes_ = classification_per.max(dim=0)
                boxes_ltrb_ = ltrb_c_cc[:, : 4]
                print_indented(n_sp + 2, 'classes_.shape 2 :', classes_.shape)
                print_indented(n_sp + 2, 'scores_.shape 2 :', scores_.shape)
                print_indented(n_sp + 2, 'boxes_ltrb_.shape :', boxes_ltrb_.shape);  #exit()
                #   classes_.shape 2 : torch.Size([6])
                #   scores_.shape 2 : torch.Size([6])
                #   boxes_ltrb_.shape : torch.Size([6, 4])
                li_di_ltrb_cls_score.append({
                    'rois_ltrb': boxes_ltrb_.cpu().numpy(),
                    'class_ids': classes_.cpu().numpy(),
                    'scores': scores_.cpu().numpy(),
                    })
            else:

                li_di_ltrb_cls_score.append({
                    'rois_ltrb': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                    })

    else:
        scores_over_thresh = (scores > threshold)[:, :, 0]
        print_indented(n_sp + 1, 'scores_over_thresh.shape :', scores_over_thresh.shape)
        for i in range(x.shape[0]):
            if scores_over_thresh[i].sum() == 0:
                li_di_ltrb_cls_score.append({
                    'rois_ltrb': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                    })
                continue

            classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_ltrb_per = transformed_anchors_ltrb[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            scores_, classes_ = classification_per.max(dim=0)
            print('classification_per.shape :', classification_per.shape, '\ntransformed_anchors_ltrb_per.shape :', transformed_anchors_ltrb_per.shape, '\nscores_per[:, 0].shape :', scores_per[:, 0].shape, '\nscores_.shape :', scores_.shape, '\nclasses_.shape :', classes_.shape);   #exit(); 
            #   classification_per.shape : torch.Size([28, 2193])
            #   transformed_anchors_ltrb_per.shape : torch.Size([2193, 4])
            #   scores_per[:, 0].shape : torch.Size([2193])
            #   scores_.shape : torch.Size([2193])
            #   classes_.shape : torch.Size([2193])
            anchors_nms_idx = batched_nms(transformed_anchors_ltrb_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)
            print('anchors_nms_idx.shape :', anchors_nms_idx.shape);   #exit();
            print('anchors_nms_idx :', anchors_nms_idx);   #exit();
            #   anchors_nms_idx.shape : torch.Size([718])
            if anchors_nms_idx.shape[0] != 0:
                classes_ = classes_[anchors_nms_idx]
                scores_ = scores_[anchors_nms_idx]
                boxes_ltrb_ = transformed_anchors_ltrb_per[anchors_nms_idx, :]
                print('classes_.shape 2 :', classes_.shape, '\nscores_.shape 2 :', scores_.shape, '\nboxes_ltrb_.shape :', boxes_ltrb_.shape);  #exit()
                #print('transformed_anchors_ltrb_per[10] :', transformed_anchors_ltrb_per[10]);  #exit()
                #for iB in range(transformed_anchors_ltrb_per.shape[0]):
                #    if 0 == iB % 10:
                #        print('iB :', iB, ', transformed_anchors_ltrb_per[iB] :', transformed_anchors_ltrb_per[iB]);  #exit()
                #exit()
                #   classes_.shape 2 : torch.Size([718])
                #   scores_.shape 2 : torch.Size([718])
                #   boxes_ltrb_.shape : torch.Size([718, 4])
                li_di_ltrb_cls_score.append({
                    'rois_ltrb': boxes_ltrb_.cpu().numpy(),
                    'class_ids': classes_.cpu().numpy(),
                    'scores': scores_.cpu().numpy(),
                    })
            else    :
                li_di_ltrb_cls_score.append({
                    'rois_ltrb': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                    })

    print_indented(n_sp, "postprocess END")
    return li_di_ltrb_cls_score


def display_just_box(ltrb, imgs):
    for i in range(len(imgs)):
        n_box = ltrb[i].shape[0]
        for iB in range(n_box):
            (x1, y1, x2, y2) = ltrb[iB, : 4].astype(np.int)
            plot_one_box(imgs[i], [x1, y1, x2, y2])
    return imgs



def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                    color=color_list[get_index_label(obj, obj_list)])
            if imshow:
                cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            path_img_random = f'test/{uuid.uuid4().hex}.jpg'
            print('path_img_random : ', path_img_random);
            cv2.imwrite(path_img_random, imgs[i])
            #exit(0);
            #cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


def replace_w_sync_bn(m):
    for var_name in dir(m):
        target_attr = getattr(m, var_name)
        if type(target_attr) == torch.nn.BatchNorm2d:
            num_features = target_attr.num_features
            eps = target_attr.eps
            momentum = target_attr.momentum
            affine = target_attr.affine

            # get parameters
            running_mean = target_attr.running_mean
            running_var = target_attr.running_var
            if affine:
                weight = target_attr.weight
                bias = target_attr.bias

            setattr(m, var_name,
                    SynchronizedBatchNorm2d(num_features, eps, momentum, affine))

            target_attr = getattr(m, var_name)
            # set parameters
            target_attr.running_mean = running_mean
            target_attr.running_var = running_var
            if affine:
                target_attr.weight = weight
                target_attr.bias = bias

    for var_name, children in m.named_children():
        replace_w_sync_bn(children)


class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')

        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
            inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
            for device_idx in range(len(devices))], \
                    [kwargs] * len(devices)


def get_last_weights(weights_path):
    weights_path = glob(weights_path + f'/*.pth')
    print('weights_path : ', weights_path); #exit(0);
    weights_path = sorted(weights_path,
            key=lambda x: int(x.rsplit('_')[-1].rsplit('.')[0]),
            reverse=True)[0]
    print(f'using weights {weights_path}')
    #exit(0);
    return weights_path


def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)


STANDARD_COLORS = [
        'LawnGreen', 'Chartreuse', 'Aqua','Beige', 'Azure','BlanchedAlmond','Bisque',
        'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
        'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
        'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
        'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
        'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
        'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
        'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
        'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        'WhiteSmoke', 'Yellow', 'YellowGreen'
        ]


def from_colorname_to_bgr(color):
    rgb_color=webcolors.name_to_rgb(color)
    result=(rgb_color.blue,rgb_color.green,rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard= []
    for i in range(len(list_color_name)-36): #-36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index


def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)),0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0]+s_size[0]+15, c1[1] - t_size[1] -3
        cv2.rectangle(img, c1, c2 , color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0],c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)


def genereate_random_color_list(n_class):
    li_color_bgr = []
    li_color_bgr.append((255, 0, 0))
    li_color_bgr.append((0, 255, 0))
    li_color_bgr.append((0, 0, 255))
    if len(li_color_bgr) < n_class:
        li_color_bgr.append((255, 255, 0))
        li_color_bgr.append((255, 0, 255))
        li_color_bgr.append((0, 255, 255))
        if len(li_color_bgr) < n_class:
            li_color_bgr.append((255, 128, 0))
            li_color_bgr.append((255, 0, 128))
            li_color_bgr.append((0, 255, 128))
            if len(li_color_bgr) < n_class:
                li_color_bgr.append((128, 255, 0))
                li_color_bgr.append((128, 0, 255))
                li_color_bgr.append((0, 128, 255))
                if len(li_color_bgr) < n_class:
                    li_color_bgr.append((128, 128, 0))
                    li_color_bgr.append((128, 0, 128))
                    li_color_bgr.append((0, 128, 128))
                    if len(li_color_bgr) < n_class:
                        more = n_class - len(li_color_bgr)                        
                        t1 = [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for k in range(more)]
                        li_color_bgr += t1 
    return li_color_bgr

#color_list = standard_to_bgr(STANDARD_COLORS)
color_list = genereate_random_color_list(len(STANDARD_COLORS))
