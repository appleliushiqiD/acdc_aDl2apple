import os
import sys
import glob
import h5py
from multiprocessing import cpu_count
import statsmodels.api as sm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup

from data_acdc import create_dataloader
from tt import train, test, save_load_model, saveBetter

import traincfg as tcfg

import logging


if not os.path.exists(tcfg.log_dir):
    os.mkdir(tcfg.log_dir)

logging.basicConfig(filename=tcfg.log_path, level=logging.INFO, format='%(asctime)s %(message)s')

train_h5 = None

cpu_num = cpu_count()
torch.set_num_threads(cpu_num)

# if test_folder == None, use scale
# create_dataset(train_folder, test_folder, crop=True, size=(128,128), batch_size=10, scale=0.8, addDiff=False):
train_dataloader, test_dataloader = create_dataloader(tcfg.train_folder, tcfg.test_folder, tcfg.crop, tcfg.img_size, tcfg.batch_size, tcfg.split_scale, tcfg.addDiff)

device = ("cuda" if torch.cuda.is_available() else "cpu")

model = tcfg.model


if tcfg.train_type == "metr":
    model_path = glob.glob(tcfg.model_load_path)
    if len(model_path) ==0:
        logging.warning('no model to load')
        sys.exit(0)
    else:
        model = save_load_model(model, model_path[0], "load")


optimizer = AdamW(model.parameters(), lr=tcfg.lossrate, eps=1e-6)
loss_fn = CrossEntropyLoss()
# weights = torch.tensor([0.1, 0.3, 0.3, 0.3]).to(device)
# loss_fn = CrossEntropyLoss(weight=weights)

# 线性学习率预热
iteration = len(train_dataloader)
warm_up_ratio = 0.1
total_steps = iteration * tcfg.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*warm_up_ratio, num_training_steps=total_steps)

logging.info('-'*40)
logging.info('Data summary:')
logging.info(' - Train images:')
logging.info(len(train_dataloader.dataset))
logging.info(' - Test:')
logging.info(len(test_dataloader.dataset))

if tcfg.train_type =="metr":
    metr_h5_file = tcfg.metr_path
    mhd5f = h5py.File(metr_h5_file, "w")

    logging.info('*'*40)
    for mn,mf in tcfg.metricset.items():
        lss, ms = test(train_dataloader, model, loss_fn, mf, device)
        mhd5f.create_dataset('train%s'%mn, data=ms)
        
        clsn = ['BG', 'RV', 'Myo', 'LV']
        logging.info('train:')
        logging.info('%s loss(%3d): %.5f (%.5f - %.5f)'%(mn, len(lss), np.mean(lss), np.min(lss), np.max(lss)))
        for i in range(4):
            logging.info('%s %s(%3d): %.5f (%.5f - %.5f)'%(mn, clsn[i], len(ms), np.mean([x[i] for x in ms]), np.min([x[i] for x in ms]), np.max([x[i] for x in ms])))
        logging.info('*'*20)
        
        lss, ms = test(test_dataloader, model, loss_fn, mf, device)
        mhd5f.create_dataset('test%s'%mn, data=ms)
        logging.info('test:')
        logging.info('%s loss(%3d): %.5f (%.5f - %.5f)'%(mn, len(lss), np.mean(lss), np.min(lss), np.max(lss)))
        for i in range(4):
            logging.info('%s %s(%3d): %.5f (%.5f - %.5f)'%(mn, clsn[i], len(ms), np.mean([x[i] for x in ms]), np.min([x[i] for x in ms]), np.max([x[i] for x in ms])))
        logging.info('*'*20)
    mhd5f.close()
    sys.exit(0)

train_all_loss = []
train_all_metric = []
test_all_loss = []
test_all_metric = []

train_min_loss = 0.1
test_min_loss = 0.1

logging.info('-'*40)
logging.info('start train on %s'%device)


for epoch in range(tcfg.epochs):
    logging.info('-'*40)
    logging.info('Epoch %02d...'%(epoch+1+tcfg.start_epoch))

    losses_tr, mss_tr = train(train_dataloader, model, loss_fn, tcfg.metric_fn, optimizer, scheduler, device)
    train_all_loss.extend(losses_tr)
    train_all_metric.extend(mss_tr)

    logging.info('-'*20)                                                              
    logging.info('Train loss -3: %.5f %.5f %.5f'%(losses_tr[-3], losses_tr[-2], losses_tr[-1]))
    logging.info('Train metric -2: ')
    logging.info('BG: %.5f, RV: %.5f, Myo: %.5f, LV: %.5f'%(mss_tr[-2][0], mss_tr[-2][1], mss_tr[-2][2], mss_tr[-2][3]))
    logging.info('BG: %.5f, RV: %.5f, Myo: %.5f, LV: %.5f'%(mss_tr[-1][0], mss_tr[-1][1], mss_tr[-1][2], mss_tr[-1][3]))
    logging.info('-'*20)

    losses_te, mss_te = test(test_dataloader, model, loss_fn, tcfg.metric_fn, device)
    test_all_loss.extend(losses_te)
    test_all_metric.extend(mss_te)

    logging.info('Test loss -3: %.5f %.5f %.5f'%(losses_te[-3], losses_te[-2], losses_te[-1])) 
    logging.info('Test metric -2: ')
    logging.info('BG: %.5f, RV: %.5f, Myo: %.5f, LV: %.5f'%(mss_te[-2][0], mss_te[-2][1], mss_te[-2][2], mss_te[-2][3]))
    logging.info('BG: %.5f, RV: %.5f, Myo: %.5f, LV: %.5f'%(mss_te[-1][0], mss_te[-1][1], mss_te[-1][2], mss_te[-1][3]))

    min_trls = min(losses_tr)
    min_tels = min(losses_te)
    if min_trls < train_min_loss:
        train_min_loss = min_trls
        saveBetter(model, losses_tr, min_trls, (epoch+1+tcfg.start_epoch), 'train', tcfg)
    if min_tels < test_min_loss:
        test_min_loss = min_tels
        saveBetter(model, losses_te, min_tels, (epoch+1+tcfg.start_epoch), 'test', tcfg)
    logging.info('-'*40)

""" save model """
save_path = tcfg.model_save_path+'-e%d-l%.5f.pth'%(tcfg.start_epoch+tcfg.epochs, train_all_loss[-1])
save_load_model(model, save_path)

""" save loss & metric data """
if tcfg.train_type == "train":
    train_h5_file = tcfg.metr_path
    train_h5 = h5py.File(train_h5_file, "w")

    train_h5.create_dataset("trls", data=train_all_loss)
    train_h5.create_dataset("tels", data=test_all_loss)
    train_h5.create_dataset("trmt", data=train_all_metric)
    train_h5.create_dataset("temt", data=test_all_metric)
    train_h5.close()

""" draw image """
smooth = lambda data: sm.nonparametric.lowess (
        data, list(range(len(data))), 0.05
        )[:, 1]

plt.figure()
plt.plot(smooth(train_all_loss), alpha=0.7, label='train')
plt.plot(smooth(test_all_loss), alpha=0.7, label='test')
plt.ylim(0, 0.7)
plt.legend()
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig(tcfg.img_path+'-loss.png')
plt.clf()

class_num = 4
line_labels = ['BG', 'RV', 'Myo', 'LV']

plt.figure()
for i in range(class_num):
    plt.plot(smooth([x[i] for x in train_all_metric]), label=line_labels[i], alpha=0.7)
plt.plot(smooth([np.mean(x) for x in train_all_metric]), label='Mean', alpha=0.7)
plt.ylim(0.4, 1)
plt.legend()
plt.xlabel('step')
plt.ylabel('train %s'%(tcfg.metric_name))
plt.savefig(tcfg.img_path+f'-train_%s.png'%(tcfg.metric_name))
plt.clf()

plt.figure()
for i in range(class_num):
    plt.plot(smooth([x[i] for x in test_all_metric]), label=line_labels[i], alpha=0.7)
plt.plot(smooth([np.mean(x) for x in test_all_metric]), label='Mean', alpha=0.7)
plt.ylim(0.4, 1)
plt.legend()
plt.xlabel('step')
plt.ylabel('test %s'%(tcfg.metric_name))
plt.savefig(tcfg.img_path+'-test_%s.png'%(tcfg.metric_name))
plt.clf()
plt.close()

