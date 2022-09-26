# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib
matplotlib.use('Agg')
#import UnetsegLSTM
#matplotlib.use('pdf')

import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference

from monai.handlers.utils import from_engine
from monai import transforms, data
from monai.data import  decollate_batch ,load_decathlon_datalist
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)

import os
import time
import shutil
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import distributed_all_gather
import torch.utils.data.distributed
from monai.data import decollate_batch

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def train_epoch(model,
                loader,
                optimizer,
                scaler,
                epoch,
                loss_func,
                args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        data, target = data.cuda(args.rank), target.cuda(args.rank)

        #data_write=

        #img_name = data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
        #if '_' in img_name:
            # flip the image 
            # b,c, h,w,t
        #    data=torch.flip(data, 2) 
        #    data=torch.flip(data, 1)

        #    target=torch.flip(target,2) 
        #    target=torch.flip(target,1) 

        #
        #data=torch.zeros([1,1,160, 192,224])#, dtype=torch.int32)
        #target=torch.zeros([1,1,160, 192,224])
        #data=data.cuda().float()
        #target=target.cuda().float()

        for param in model.parameters(): param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0 and idx % 10==0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg

def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )

        dice_metric.reset()

    mean_dice_val = np.mean(dice_vals)
    return mean_dice_val


def val_epoch_organ_acc(model,
              loader,
              epoch,
              acc_func,
              args,
              model_inferer=None,
              post_label=None,
              post_pred=None):
    model.eval()
    start_time = time.time()
    acc_all=[]
    acc_avg_all=[]
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            #print ('*'*50)
            #print (target.max())
            #print (target.min())
            #print (logits.max())
            #print (logits.min())            
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)
            #acc=acc[:,1:14]
            #print (acc.size())

            if args.distributed:
                acc_list = distributed_all_gather([acc],
                                                  out_numpy=True,
                                                  is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
            acc_avg_all.append(avg_acc)
            #print (acc_all)
            #if args.rank == 0:
            #    print('Val {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
            #          'acc', avg_acc,
            #          'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()


            acc=torch.squeeze(acc).detach().cpu().numpy()
            
            #acc_all.append(avg_acc)
            acc_all.append(acc)
            
            start_time = time.time()
    acc_all=np.array(acc_all)
    #acc_all=np.mean(acc_all, axis=0)
    #print (acc_all)
    #print (acc_all.shape)
    return acc_all,np.average(acc_avg_all)

def val_epoch(model,
              loader,
              epoch,
              acc_func,
              args,
              model_inferer=None,
              post_label=None,
              post_pred=None):
    model.eval()
    start_time = time.time()
    acc_all=[]
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            #print ('*'*50)
            #print (target.max())
            #print (target.min())
            #print (logits.max())
            #print (logits.min())            
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)
            #acc=acc[:,1:14]
            #print (acc.size())
            if args.distributed:
                acc_list = distributed_all_gather([acc],
                                                  out_numpy=True,
                                                  is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
            acc_all.append(avg_acc)
            #print (acc_all)
            #if args.rank == 0:
            #    print('Val {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
            #          'acc', avg_acc,
            #          'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()
    acc_all=np.array(acc_all)
    #print (acc_all)
    #print (acc_all.shape)
    return np.average(acc_all)

    
def val_epoch_ori_size(model,loader,acc_func,args,model_inferer=None,post_label=None,post_pred=None,post_transform=None):
    model.eval()
    start_time = time.time()
    acc_all=[]
    with torch.no_grad():
        for val_data in loader:
            #print ('info: validate data')
            val_inputs = val_data["image"].cuda()
            roi_size = (args.roi_x, args.roi_y, args.roi_z)
            sw_batch_size = 4
            val_data["pred"] = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model)
            val_data = [post_transform(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            # compute metric for current iteration
            #print (val_outputs[0].size())
            #print (val_labels[0].size())
            acc_func(y_pred=val_outputs, y=val_labels)
            dice = acc_func.aggregate().item()
            #print ('info: validate data acc', dice)
            acc_all.append(dice)
    # aggregate the final mean dice result
        
    # reset the status for next validation round
    acc_func.reset()
    return np.mean(acc_all)

def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(args.logdir, filename)
    torch.save(state_dict, filename)
    #torch.save(model.state_dict(), PATH)
    print('Saving checkpoint', filename)

def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 acc_func,
                 args,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 post_label=None,
                 post_pred=None
                 ):
    

   
    
    #print ('info: val acc is ',val_avg_acc)
    writer = None
    epoch_loss_values=[]
    val_loss_values=[]
                     
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0: print('Writing Tensorboard logs to ', args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.

    val_acc_max1=0.
    val_acc_organ_max=np.zeros(13)
    organ_list=['sp','rk','lk','gall','eso','lv','aorta','stoma','infe','portal','pancreas','ra','la']
    # start to write the validation accuracy
    wt_path=os.path.join(args.logdir, 'validation accuracy.csv')
    fd_results = open(wt_path, 'w')
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        #print(args.rank, time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()
         
        train_loss = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 scaler=scaler,
                                 epoch=epoch,
                                 loss_func=loss_func,
                                 args=args)
        epoch_loss_values.append(train_loss)
        if args.rank == 0 :
            print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time))
        if args.rank==0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
        b_new_best = False
        if (epoch+1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()

            
            # val_avg_acc = val_epoch(model,
            #                         val_loader,
            #                         epoch=epoch,
            #                         acc_func=acc_func,
            #                         model_inferer=model_inferer,
            #                         args=args,
            #                         post_label=post_label,
            #                         post_pred=post_pred)

            # val_avg_acc = val_epoch_ori_size(model,
            #                 val_org_loader,
            #                 acc_func=acc_func,
            #                 model_inferer=model_inferer,
            #                 args=args,
            #                 post_label=post_label,
            #                 post_pred=post_pred,
            #                 post_transform=post_transforms)    

            val_acc_all,val_avg_acc = val_epoch_organ_acc(model,
                                        val_loader,
                                        epoch=epoch,
                                        acc_func=acc_func,
                                        model_inferer=model_inferer,
                                        args=args,
                                        post_label=post_label,
                                        post_pred=post_pred)
                
                
            val_organ_wise=np.nanmean(val_acc_all,axis=0) # numpy.nanmean
            #print (val_organ_wise)
            fd_results.write(str(epoch )+',')
            for organ_id in range(len(val_organ_wise)):
                fd_results.write(str(val_organ_wise[organ_id])+ ',')

                if val_organ_wise[organ_id]>val_acc_organ_max[organ_id]: # max organ-wise achieved
                    save_nm='best_acc_organ_'+organ_list[organ_id]+'.pt'
                    print ('info: best accuracy of organ ',organ_list[organ_id])
                    print (val_organ_wise[organ_id])
                    save_checkpoint(model,
                                    epoch,
                                    args,
                                    best_acc=val_acc_max,
                                    filename=save_nm)
                    val_acc_organ_max[organ_id]=val_organ_wise[organ_id]
            fd_results.write(str(val_avg_acc)+ ',') 
            fd_results.write('\n')
            fd_results.flush() 
            mean_acc1=np.average(val_organ_wise)
            if mean_acc1>val_acc_max1:
                val_acc_max1=mean_acc1
                #save_nm='best_acc_organ_AVG.pt'
                #print ('info: best accuracy of average organ ',mean_acc1)
                #save_checkpoint(model,
                #                epoch,
                #                args,
                #                best_acc=mean_acc1,
                #                filename=save_nm)
            #print ('info: Validation Done!!!')

            #tep_acc=np.mean(val_acc_all,axis=1)
            #print (tep_acc)
            #val_avg_acc=np.mean(tep_acc,axis=0)  
            #print (val_avg_acc)
            val_loss_values.append(val_avg_acc)

            #val_loss_values.append(val_avg_acc)
            #fd_results.write(str(epoch ) + ',' + str(val_avg_acc)+'\n')
            #fd_results.flush() 
            if args.rank == 0:
                print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                      'acc', val_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time))
                if writer is not None:
                    writer.add_scalar('val_acc', val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(model, epoch, args,
                                        best_acc=val_acc_max,
                                        optimizer=optimizer,
                                        scheduler=scheduler)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model,
                                epoch,
                                args,
                                best_acc=val_acc_max,
                                filename='model_final.pt')
                if b_new_best:
                    print('Copying to model.pt new best model!!!!')
                    shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))
            
            show_img=False
            if 1>0:
                if show_img:
                    plt.figure(1, figsize=(8, 8))
                    plt.subplot(2, 2, 1)
                    plt.plot(epoch_loss_values)
                    plt.grid()
                    plt.title('Training Loss')

                    plt.subplot(2, 2, 2)
                    plt.plot(val_loss_values)
                    plt.grid()
                    plt.title('Validation Loss')


                    plt.savefig(os.path.join(args.logdir, 'train_val_loss_plots.png'))
                    plt.close(1)


        if scheduler is not None:
            scheduler.step()

    print('Training Finished !, Best Accuracy: ', val_acc_max)

    return val_acc_max

