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

def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


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

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess(*image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
        means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


#def preprocess_video(*frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
def preprocess_video(ori_imgs, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    #ori_imgs = frame_from_video
    #print('type(ori_imgs[0]) :', type(ori_imgs[0]));  exit()
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
        means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

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


def merge_divided_detections(det_batch, idx_bbox_from, li_offset_xy, is_whole_included, im_bgr_hwc_ori_np, ratio_resize, bbox_type):
    '''
    bbox_type : one of 'ltrb' / 'ltwh' / 'xywh'
         for image_i, pred in enumerate(li_det):
                 pred[:, 0] += li_offset_xy[image_i][0]; 
                         pred[:, 1] += li_offset_xy[image_i][1];
                             '''
                                 #im_bgr_resized = cv2.resize(im_bgr_hwc_ori_np, None, fx = ratio_resize, fy = ratio_resize)
                                     
    print('is_whole_included :', is_whole_included);  #exit()
    print('li_offset_xy.shape :', li_offset_xy.shape);  #exit()
    print('det_batch.shape :', det_batch.shape);  #exit()
    if isinstance(det_batch, torch.Tensor):
        li_offset_xy = torch.from_numpy(li_offset_xy).to(det_batch)
    for image_i, offset_xy in enumerate(li_offset_xy):
        #print('offset_xy.shape :', offset_xy.shape);     #exit()
        #print('offset_xy b4 :', offset_xy);     #exit()
        #offset_xy[0] = 100; offset_xy[1] = 1000;
        #print('offset_xy after :', offset_xy);     #exit()
        #print('det_batch[image_i, :, :2] b4 :', det_batch[image_i, :, :2])
        det_batch[image_i, :, idx_bbox_from : idx_bbox_from + 2] += offset_xy[:]
        if 'ltrb' == bbox_type:
            det_batch[image_i, :, idx_bbox_from + 2 : idx_bbox_from + 4] += offset_xy[:]
        #print('det_batch[image_i, :, :2] after :', det_batch[image_i, :, :2])
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
    if is_whole_included:
        det_batch[-1, :, idx_bbox_from : idx_bbox_from + 4] /= ratio_resize
        h_ori, w_ori, _ = im_bgr_hwc_ori_np.shape
        #print('im_bgr_hwc_ori_np.shape : ', im_bgr_hwc_ori_np.shape) 
        #print('ratio_resize : ', ratio_resize) 
        if w_ori > h_ori:
            margin_y = 0.5 * float(w_ori) * ratio_resize * (1.0 - float(h_ori) / float(w_ori))
            #print('margin_y : ', margin_y); #exit()     
            det_batch[-1, :, idx_bbox_from + 1] -= margin_y
            if 'ltrb' == bbox_type:
                det_batch[-1, :, idx_bbox_from + 3] -= margin_y
        elif h_ori > w_ori:
            margin_x = 0.5 * h_ori * ratio_resize * (1.0 - w_ori / h_ori)
            #print('margin_x : ', margin_x); #exit()     
            det_batch[-1, :, idx_bbox_from] -= margin_x
            if 'ltrb' == bbox_type:
                det_batch[-1, :, idx_bbox_from + 2] -= margin_x
        '''
        ltrb = xywh2xyxy(li_det[-1, :, :4])# / (ratio_resize)
        for iR in range(len(ltrb)):
            #if 0 != iR % 10: continue
            if li_det[-1, iR, 4] < 0.6: continue
            cv2.rectangle(im_bgr_resized, (ltrb[iR, 0], ltrb[iR, 1]), (ltrb[iR, 2], ltrb[iR, 3]), (255, 0, 0))  
        cv2.imshow('im_bgr_resized', im_bgr_resized); cv2.waitKey(1); # exit()
        '''
    n_batch, n_det, n_attribute = tuple(det_batch.size())
    n_det_total = n_batch * n_det;
    #print('n_batch : ', n_batch);   print('n_det : ', n_det);   print('n_attribute : ', n_attribute);   print('n_det_total : ', n_det_total);  #exit();
    #det = np.vstack(li_det)
    det_merged = det_batch.view(-1, n_attribute);
    print('det_merged.size() : ', det_merged.size()); #exit()
    det_merged.unsqueeze_(0)
    return det_merged
                                                                                                                                                                                                                                                                                                                                                                                                                



def non_max_suppression_4_mosaic(pred_xywh_c_cc, li_offset_xy, include_original, li_group, im_bgr_hwc_ori_np, ratio_resize, li_str_class, too_included, bbox_type, conf_thres=0.5, nms_thres=0.5): #pred_letterbox.type() : torch.cuda.FloatTensor
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
    #print('pred_xywh_c_cc.shape b4 :', pred_xywh_c_cc.shape);  #exit();
    if li_offset_xy is not None and pred_xywh_c_cc.shape[0] > 1:
        pred_xywh_c_cc = merge_divided_detections(pred_xywh_c_cc, 0, li_offset_xy, include_original, im_bgr_hwc_ori_np, ratio_resize, bbox_type) 
    #print('pred_xywh_c_cc.shape after :', pred_xywh_c_cc.shape);  exit();
    min_wh = 2  # (pixels) minimum box width and height
    output = [None] * len(pred_xywh_c_cc)
    for image_i, xywh_c_cc in enumerate(pred_xywh_c_cc):
        #print('image_i : ', image_i)
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
        
        xywh_c_cc[:, 4] *= class_conf
        
        # Select only suitable predictions
        i = (xywh_c_cc[:, 4] > conf_thres) & (xywh_c_cc[:, 2:4] > min_wh).all(1) & torch.isfinite(xywh_c_cc).all(1)
        print('xywh_c_cc[:, 4].max() :', xywh_c_cc[:, 4].max());
        print('class_conf :', class_conf);
        print('i :', i);   exit()
        xywh_c_cc = xywh_c_cc[i]
        
        # If none are remaining => process next image
        if len(xywh_c_cc) == 0:
            continue
            
        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        ltrb_c_cc = xywh_c_cc.clone()
        ltrb_c_cc[:, :4] = xywh2xyxy(xywh_c_cc[:, :4])
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
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]
                
                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold
                    
            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:                            
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold
                        
            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
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
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                        
        if include_original:
            det_max = compensate_division(det_max, li_group, li_str_class, too_included)
            
            
        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort
    #exit()
    return output



def ltrb_2_xywh(ltrb):
    xywh = ltrb.new(ltrb.shape)
    xywh[..., 0] = (ltrb[..., 0] + ltrb[..., 2]) / 2.0
    xywh[..., 1] = (ltrb[..., 1] + ltrb[..., 3]) / 2.0
    xywh[..., 2] = ltrb[..., 2] - ltrb[..., 0]
    xywh[..., 3] = ltrb[..., 3] - ltrb[..., 1]
    return xywh


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold, li_offset_xy = None, include_original = None, li_group = None, im_bgr_hwc_ori_np = None, ratio_resize = None, li_str_class = None, too_included = None):
    is_mosaic = li_offset_xy is not None
    print('x.shape :', x.shape, '\ntype(anchors) :', type(anchors), '\nanchors.shape :', anchors.shape, '\ntype(regression) :', type(regression), '\nregression.shape :', regression.shape, '\ntype(classification) :', type(classification), '\ntype(regressBoxes) : ', type(regressBoxes), '\ntype(clipBoxes) :', type(clipBoxes)); #exit(); 
    # x.shape : torch.Size([1, 3, 512, 512])
    # anchor.shape : torch.Size(1, 49104, 4])
    # regressioin.shape : torch.Size(1, 49104, 4])
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    print(' transformed_anchors.shape :', transformed_anchors.shape, '\n scores.shape :', scores.shape, '\n scores_over_thresh.shape :', scores_over_thresh.shape, '\n classification.shape :', classification.shape); #exit(); 
    ##  transformed_anchors.shape : torch.Size([1, 49104, 4])
    ##  scores.shape : torch.Size(1, 49104, 1])
    ##  scores_over_thresh.shape : torch.Size(1, 49104])
    ##  classification.shape : torch.Size(1, 49104, 28])

    out = []
    if is_mosaic:
        xywh = ltrb_2_xywh(transformed_anchors)
        bbox_type = 'xywh'

        print('xywh.shape :', xywh.shape);
        '''
        for iD in range(anchors.shape[1]):
            #if anchors[0, iD, 2] < 0:
            if anchors[0, iD, 0] < 0 and anchors[0, iD, 1] < 0:
                print('\niD :', iD)
                print('anchors[0, iD, :]', anchors[0, iD, :]);   
                print('transformed_anchors[0, iD, :]', transformed_anchors[0, iD, :]);   
                print('xywh[0, iD, :]', xywh[0, iD, :]);
        '''

        #print('\nanchors[0, 49005, :] :', anchors[0, 49005, :]);   
        #print('transformed_anchors[0, 49005, :] :', transformed_anchors[0, 49005, :]);   
        #print('xywh[0, 49005, :] :', xywh[0, 49005, :]);
        #print('scores[0, 49005, :] :', scores[0, 49005, :]);
        #print('scores.argmax() :', scores.argmax())
        #print('scores.max() :', scores.max())
        #print('scores.min() :', scores.min())
        #print('threshold :', threshold)
        #print('scores[0, scores.argmax(), :] :', scores[0, scores.argmax(), :]);
        #print('classification[0, scores.argmax(), :] :', classification[0, scores.argmax(), :]);
        #print('classification[0, 49005, :] :', classification[0, 49005, :]);
        #print('classification[0, scores.argmax(), :].sum() :', classification[0, scores.argmax(), :].sum());
        #print('classification[0, 49005, :].sum() :', classification[0, 49005, :].sum());
        ##  anchors[0, 49005, :] : tensor([-64., -64., 448., 448.], device='cuda:0')
        ##  transformed_anchors[0, 49005, :] : tensor([0.000, 0.000, 382.0305, 286.9348], device='cuda:0')
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
        anchors_nms_idx = non_max_suppression_4_mosaic(pred_xywh_c_cc, li_offset_xy, include_original, li_group, im_bgr_hwc_ori_np, ratio_resize, li_str_class, too_included, bbox_type, threshold, iou_threshold);

    else:
        for i in range(x.shape[0]):
            if scores_over_thresh[i].sum() == 0:
                out.append({
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                    })
                continue

            classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            scores_, classes_ = classification_per.max(dim=0)
            print('transformed_anchors_per.shape :', transformed_anchors_per.shape, '\nscores_per[:, 0].shape :', scores_per[:, 0].shape, '\nclasses_.shape :', classes_.shape);   exit(); 

            anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

            if anchors_nms_idx.shape[0] != 0:
                classes_ = classes_[anchors_nms_idx]
                scores_ = scores_[anchors_nms_idx]
                boxes_ = transformed_anchors_per[anchors_nms_idx, :]

                out.append({
                    'rois': boxes_.cpu().numpy(),
                    'class_ids': classes_.cpu().numpy(),
                    'scores': scores_.cpu().numpy(),
                    })
            else    :
                out.append({
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                    })

                return out


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
