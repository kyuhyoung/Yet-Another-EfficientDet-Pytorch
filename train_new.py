import datetime
import os
import argparse
import traceback
import random 
import math

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
#from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.dataset import ChallengeDataset, Resizer, Normalizer, Augmenter, collater
from backbone import EfficientDetBackbone
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args(sys_argv = None):
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    #parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-p', '--project', type=str, default='ai_challenge_2020', help='project file that contains parameters')
    #parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='coefficients of efficientdet')
    #parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='num_workers of dataloader')
    #parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    #parser.add_argument('--batch_size', type=int, default=36, help='The number of images per batch among all devices')
    parser.add_argument('--batch_size', type=int, default=10, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    #parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    #parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    #parser.add_argument('--val_interval_step', type=int, default=5, help='Number of epoches between valing phases')
    parser.add_argument('--dir_pred', type=str, default='prediction', help='directory of predeicton result xml file.')
    parser.add_argument('--model_name', type=str, default='1', help='name of the model')
    parser.add_argument('--n_cls', type=int, default=29, help='Number of classes')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    #parser.add_argument('--save_interval', type=int, default=1, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    #parser.add_argument('-w', '--load_weights', type=str, default=None,
    parser.add_argument('-w', '--load_weights', type=str, default='last',
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    #parser.add_argument('--debug', type=boolean_string, default=False, help='whether visualize the predicted boxes of training, '
    parser.add_argument('--debug', type=boolean_string, default=True, help='whether visualize the predicted boxes of training. The output images will be in test/')
    parser.add_argument('--xywh', type=boolean_string, default=False, help='whether bounding box representation is left-top-width-height or not.')
    '''
    for ii in range(100):
        print('random() : ', random.random())
    exit(0);
    '''   
    #t0 = os.path.basename(sys_argv[0]); t1 = os.path.basename(__file__);
    #print('t0 : ', t0); print('t1 : ', t1);
    #exit(0);
    if os.path.basename(sys_argv[0]) == os.path.basename(__file__):
        #print("called from this py file")
        args = parser.parse_args()
    else:
        #print("called from another py file")
        args = parser.parse_args([])
    '''
    if sys_argv is None or 1 == len(sys_argv):
        #print('len(sys_argv) is 1')
        args = parser.parse_args()
        #print('args : ', args); exit()
    else:
        #print('sys_argv : ', sys_argv);   exit()
        args = parser.parse_args(sys_argv)
    '''
    return args


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, shall_display_current_result, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        #print('self.debug : ', self.debug); exit(0);
        if self.debug and shall_display_current_result:
            #print('forward WITH display')
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations, imgs=imgs, obj_list=obj_list)
            #cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations, imgs=imgs, obj_list=obj_list)
        else:
            #print('forward WITHOUT display')
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def load_weight_from_file(model, load_weights, saved_path):
    if load_weights.endswith('.pth'):
        weights_path = load_weights
    else:
        #print('saved_path : ', saved_path); exit(0);
        weights_path = get_last_weights(saved_path)
    try:
        last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
    except:
        last_step = 0
    print('weights_path :', weights_path);  print('last_step :', last_step);
    #exit(0);
    try:
        ret = model.load_state_dict(torch.load(weights_path), strict=False)
    except RuntimeError as e:
        print(f'[Warning] Ignoring {e}')
        print('[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

    print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    return model, last_step, weights_path


def train(opt, trainset_given = None, valset_given = None):
    #print('opt.project : ', opt.project);   exit(0);
    params = Params(f'projects/{opt.project}.yml')
    
    #print('params.num_gpus : ', params.num_gpus);   exit(0);
    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    '''
    if torch.cuda.is_available():
        #print('cuda is available'); exit(0);
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    '''
    #opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    print('opt.batch_size : ', opt.batch_size);
    print('opt.num_workers : ', opt.num_workers);
    #exit(0);
    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    #training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
    training_set = ChallengeDataset(input_side = input_sizes[opt.compound_coef], data_given = trainset_given, root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    #val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
    val_set = ChallengeDataset(input_side = input_sizes[opt.compound_coef], data_given = valset_given, root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)
    
    n_iter_val = len(val_generator)
    th_prob_display_val = 1.0 / 30.0
    #interval_display_val = int(n_iter_val * th_prob_display_val);
    print('n_iter_val : ', n_iter_val)
    #print('interval_display_val : ', interval_display_val)
    print('th_prob_display_val : ', th_prob_display_val)

    '''
    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    '''
    # load last weights
    if opt.model_name is not None:
        model, last_step, weigts_path = load_weight_from_file(model, opt.model_name, opt.saved_path)
        '''
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
        '''
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)
    print('opt.lr :', opt.lr)  
    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    val_interval_iter = int(num_iter_per_epoch / 5)
    #val_interval_iter = 5
    print('val_interval_iter : ', val_interval_iter)
    #check parameter of model
    print("------------------------------------------------------------")
    total_params = sum(p.numel() for p in model.parameters())
    print("num of parameter : ", total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num of trainable_ parameter : ", trainable_params)
    print("------------------------------------------------------------")



    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                '''
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                '''     
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    th_prob_display = 1.0 / math.sqrt(step - last_step + 1.0) 
                    prob_rand = random.random()
                    #print('prob_rand : ', prob_rand, ' / th_prob_display : ', th_prob_display)
                    shall_display_current_result = prob_rand < th_prob_display and opt.debug  
                    cls_loss, reg_loss = model(imgs, annot, shall_display_current_result, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1
                    #print('step : ', step);
                    #print('opt.save_interval : ', opt.save_interval);  # exit(0);
                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model,  os.path.join(opt.saved_path, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth'))
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

                #if epoch % opt.val_interval == 0:
                if step > 0 and step % val_interval_iter == 0:
                    print('validating ...') 
                    scheduler.step(np.mean(epoch_loss))
                    epoch_loss = []
                    model.eval()
                    loss_regression_ls = []
                    loss_classification_ls = []
                    #print('BBB eval') 
                    for iter, data in enumerate(val_generator):
                        '''
                        if 0 == iter % 100:
                            print('iter eval : ', iter, ' / ', len(val_generator)) 
                        if 200 < iter:
                            break    
                        '''     
                        with torch.no_grad():
                            imgs = data['img']
                            annot = data['annot']

                            if params.num_gpus == 1:
                                imgs = imgs.cuda()
                                annot = annot.cuda()

                            #th_prob_display_val = 1.0 / math.sqrt(step - last_step + 1.0) 
                            prob_rand = random.random()
                            shall_display_result_val = prob_rand < th_prob_display_val  
                            cls_loss, reg_loss = model(imgs, annot, shall_display_result_val, obj_list = params.obj_list)
                            cls_loss = cls_loss.mean()
                            reg_loss = reg_loss.mean()

                            loss = cls_loss + reg_loss
                            if loss == 0 or not torch.isfinite(loss):
                                continue

                            loss_classification_ls.append(cls_loss.item())
                            loss_regression_ls.append(reg_loss.item())
                                
                    cls_loss = np.mean(loss_classification_ls)
                    reg_loss = np.mean(loss_regression_ls)
                    loss = cls_loss + reg_loss

                    print(
                        'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                            epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                    writer.add_scalars('Loss', {'val': loss}, step)
                    writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                    if loss + opt.es_min_delta < best_loss:
                        best_loss = loss
                        best_epoch = epoch

                        save_checkpoint(model, os.path.join(opt.saved_path, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth'))

                    model.train()
                           
                # Early stopping
                    if epoch - best_epoch > opt.es_patience > 0:
                        print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                        break
                    #exit(0)       


            '''
            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, 0 == iter % 50, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, os.path.join(opt.saved_path, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth'))

                model.train()
                           
                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
            '''                         
    except KeyboardInterrupt:
        save_checkpoint(model, os.path.join(opt.saved_path, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth'))
        writer.close()
    writer.close()


def save_checkpoint(model, path_checkpoint):
    #print('name : ', name);
    if isinstance(model, CustomDataParallel):
        #print('it is CustomDataParallel');  exit(0);
        torch.save(model.module.model.state_dict(), path_checkpoint)
    else:
        #print('it is NOT CustomDataParallel');  #exit(0);
        #print('path_checkpoint : ', path_checkpoint)
        torch.save(model.model.state_dict(), path_checkpoint)
        #exit(0)

#'''
if __name__ == '__main__':
    opt = get_args()
    train(opt)
#'''    
