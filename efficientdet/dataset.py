import os
import torch
import numpy as np
from collections import defaultdict
import itertools

from torch.utils.data import Dataset, DataLoader
#from pycocotools.coco import COCO
import cv2

'''
class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations
'''

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def is_this_empty_string(strin):
    return (strin in (None, '')) or (not strin.stip())

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

def get_exact_file_name_from_path(str_path):
    return os.path.splitext(os.path.basename(str_path))[0]


def make_dict_of_image_file_name_and_path_under_this_dir(dir_im):   
    li_path = get_list_of_file_path_under_1st_with_2nd_extension(dir_im)
    di_img_fn_and_path = {}
    for path in li_path:
        di_img_fn_and_path[get_exact_file_name_from_path(path)] = path; 
    return di_img_fn_and_path

class ChallengeDataset(Dataset):
    def __init__(self, root_dir, input_side, set='train2017', transform=None, data_given = None):
        
        self.input_side = input_side
        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform
        self.dataset_given = data_given
        self.is_xywh = True if self.dataset_given is None else self.dataset_given.is_xywh
        self.load_classes()
        #self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        #self.image_ids = self.coco.getImgIds()
        #print('type(data_given) : ', type(data_given)); #exit(0);
        #print('type(data_given.imgs) : ', type(data_given.imgs)); #exit(0);
        #print('type(data_given.imgs[0]) : ', type(data_given.imgs[0])); #exit(0);
        #print('type(data_given.labels[0]) : ', type(data_given.labels[0])); #exit(0);
        #print('type(data_given.transforms) : ', type(data_given.transforms)); #exit(0);
        if data_given is None:
            self.imgs = make_dict_of_image_file_name_and_path_under_this_dir(os.path.join(self.root_dir, self.set_name));
        else:    
            #print('aaa'); exit(0);
            self.imgs = {}
            for path_im in data_given.imgs:
                aidi = get_exact_file_name_from_path(path_im)
                self.imgs[aidi] = {"id" : aidi, "file_name" : path_im} 
                
            self.im2anns = defaultdict(list)
            self.anns = {}
            count_bbox = 0                
            #print('type(self.dataset_given.labels) : ', type(self.dataset_given.labels));   #exit(0);
            for idx_label, label in enumerate(self.dataset_given.labels):
                if 0 == idx_label % 10000:
                    print('idx_label : ', idx_label, ' / ', len(self.dataset_given.labels))
                id_img = get_exact_file_name_from_path(label.attrib['name']) 
                for box_info in label.findall('./box') :
                    class_name, x1, y1, x2, y2 = box_info.attrib['label'], box_info.attrib['xtl'], box_info.attrib['ytl'], box_info.attrib['xbr'], box_info.attrib['ybr'] 
                    #print('type(x1) : ', type(x1)); exit(0)      
                    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                    #x1, y1, x2, y2 = map(int,map(float, [x1, y1, x2, y2]))
                        #boxes.append([x1,y1,x2,y2])
                        #class_names.append(class_num[class_name])
                    ann = {"image_id" : id_img, "category_id" : self.category_name_2_category_id[class_name], "id" : count_bbox, "bbox" : [x1, y1, x2, y2]}
                    self.im2anns[id_img].append(ann)
                    self.anns[count_bbox] = ann
                    #print('count_bbox : ', count_bbox)
                    count_bbox += 1

 


        self.image_ids = list(self.imgs.keys())
        #t0 = list(self.image_ids)#[0] : ', self.image_ids[0])
        #print('t0[0] : ', t0[0])
        #print('self.imgs[t0[0]] : ', self.imgs[t0[0]])
        #exit(0);
    
    def loadImgs(self, ids=[]):
        #print('type(ids) : ', type(ids));   exit(0);
        if (not isinstance(ids, str)) and (_isArrayLike(ids)):
            #print('ids array : ', ids) 
            return [self.imgs[aidi] for aidi in ids]
        else:
            #print('ids scalar : ', ids) 
            return [self.imgs[ids]]

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        #print('type(ids) : ', type(ids))
        #print('ids : ', ids);   exit(0);
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == str or type(ids) == int:
            return [self.anns[ids]]    

    def load_classes(self):

        # load class names (name -> label)
        if self.dataset_given is None:
            categories = self.coco.loadCats(self.coco.getCatIds())
            print('categories b4 : ', categories)
            categories.sort(key=lambda x: x['id'])
        else:
            categories = [{"id" : idx + 1, "name" : cat.find('name').text} for idx, cat in enumerate(self.dataset_given.categories)]
            #categories = [{"id" : idx + 1, "name" : cat.find('name').text} for idx, cat in enumerate(self.dataset_given.categories)]
        #print('categories after : ', categories);   exit(0)
        self.classes = {}
        self.category_name_2_category_id = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)
            self.category_name_2_category_id[c['name']] = c['id']

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
        print('self.classes : ', self.classes); #  exit(0);    
        print('self.labels : ', self.labels);   #exit(0);    

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        #print('START self.load_image(idx)')
        img = self.load_image(idx)
        if img is None:
            print('img is NONE !!!')
            annot = np.zeros((0, 5))
            sample = {'img': np.zeros([self.input_side, self.input_side, 3], dtype = np.float32), 'annot': annot}
        else:      
            #print('END self.load_image(idx)')
            #print('START self.load_annotatons(idx)')
            annot = self.load_annotations(idx)
            #print('END self.load_annotatons(idx)')
            #print('type(img) : ', type(img))
            #print('type(annot) : ', type(annot))
            #exit(0);
            sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        if self.dataset_given is None:
            image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        else:
            #t0 = self.image_ids[image_index]
            #print('type(t0) : ', type(t0))
            #print('t0 : ', t0)
            image_info = self.loadImgs(self.image_ids[image_index])[0]
            #print('image_info : ', image_info); exit(0); 
        #print('self.root_dir : ', self.root_dir);   print('self.set_name : ', self.set_name);   exit(0);
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        #print('path : ', path)
        #path = '/tf/notebooks/datasets/07_object_detection/val/ZED3_KSC_052263_L_P012028.png'
        img = cv2.imread(path)
        if(img is None):
            print('Can NOT read image from : ', path)
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.
    
    def getAnnIds(self, imgIds):
        
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]

        if not len(imgIds) == 0:
            lists = [self.im2anns[imgId] for imgId in imgIds if imgId in self.im2anns]
            anns = list(itertools.chain.from_iterable(lists))
        else:
            printA('Why len(imgIds) is zero?'); exit(0);
            #anns = self.dataset['annotations']
        #print('anns : ', anns)
        #print('type(anns) : ', type(anns)); #exit(0)
        ids = [ann['id'] for ann in anns]
        return ids



    def load_annotations(self, image_index):
        # get ground truth annotations
        if self.dataset_given is None:
            annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        else:
            #t0 = self.image_ids[image_index]
            #print('type(t0) : ', type(t0)); #exit(0);
            #print('t0 : ', t0); #exit(0);
            annotations_ids = self.getAnnIds(imgIds = [self.image_ids[image_index]])
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        #coco_annotations = self.coco.loadAnns(annotations_ids)
        challenge_annotations = self.loadAnns(annotations_ids)
        for idx, a in enumerate(challenge_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        if self.is_xywh:
            annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
            annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        #print('annotations : ', annotations)
        return annotations


def collater(data):
    '''
    print('type(data) : ', type(data));    #exit(0);
    print('len(data) : ', len(data));    #exit(0);
    print('type(data[0]) : ', type(data[0]));    #exit(0);
    print('len(data[0]) : ', len(data[0]));    #exit(0);
    print('type(data[0][0]) : ', type(data[0][0]));    #exit(0);
    print('type(data[0][1]) : ', type(data[0][1]));    #exit(0);
    print('data[0][1] : ',  data[0][1]);    exit(0);
    '''
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
