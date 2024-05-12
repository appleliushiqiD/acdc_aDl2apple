import torch
from torch import nn
import numpy as np

import logging


def train(dataloader, model, loss_fn, metric_fn, optimizer, scheduler=None, device="cpu"):
    model.to(device).train()
    data_len = len(dataloader.dataset)

    losses = []
    mss = []

    for itera, (img, label) in enumerate(dataloader):
        img, label = img.to(device), label.to(device)
        # img: (batch_size, 1, H, W)
        # label: (batch_size, H, W)
        pre = model(img)
        # pre: (batch, 4, H, W)
        loss = loss_fn(pre, label.long())

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        optimizer.zero_grad()

        with torch.no_grad():
            pred = torch.argmax(pre, dim=1)
            mss.append(metric_fn(label, pred))

        if itera % 10 == 0:
            current = (itera + 1) * len(img)
            logging.info('loss: %.5f  [%5d / %5d]'%(losses[-1], current, data_len))
            logging.info('BG: %.5f, RV: %.5f, Myo: %.5f, LV: %.5f'%(mss[-1][0], mss[-1][1], mss[-1][2], mss[-1][3]))

    return losses, mss

def test(dataloader, model, loss_fn, metric_fn, device="cpu"):
    model.to(device).eval()
    losses = []
    mss = []

    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            pre = model(img)
            loss = loss_fn(pre, label.long())
            losses.append(loss.item())

            pred = torch.argmax(pre, dim=1)
            mss.append(metric_fn(label, pred))

    return losses, mss

def save_load_model(model, model_path, sl="save"):
    if sl=="save":
        torch.save(model.state_dict(), model_path)
        logging.info('Saved PyTorch Model State to %s'%model_path)
    elif sl=="load":
        model.load_state_dict(torch.load(model_path))
        logging.info('Loaded PyTorch Model State from %s'%model_path)
        return model
    else:
        logging.warning('func: tt.save_load_model, argument wrong!(not "save" or "load")')


def saveBetter(model, loss_ls, min_ls, epoch, tt, cfg):
    step = loss_ls.index(min_ls)
    logging.info('find a better %s loss: %.5f'%(tt, min_ls))

    save_path = cfg.model_save_path+'-%s-e%02d-s%02d-l%.5f.pth'%(tt, epoch, step, min_ls)
    save_load_model(model, save_path)






