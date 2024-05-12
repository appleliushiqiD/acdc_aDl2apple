import os
import glob

import cv2
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import logging


def load_nii(file):
    """
    file: nii / nii.gz file(full path)    
    return:
        get_fdata: type=np.ndarray
        affine: type=np.ndarray
        header: type=nib......
    """
    nimg = nib.load(file)
    return nimg.get_fdata(), nimg.affine, nimg.header

def niigz2png(in_file, out_folder, gt_flag=False):
    """
    in_foler: 'training/patien...'
    out_folder: '/p...png'
    gt_flag: boolean, is label?(gt * 255//num_class)

    save '.../fn.nii.gz' --> 'fn.png'
    """
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    patient_frame = os.path.basename(in_file).split('.nii.gz')[0]
    
    fdata, _, _ = load_nii(in_file)

    for i in range(fdata.shape[2]):
        ''' [slice, x, y] '''
        # clean data, make sure no wrong class 
        out_name = patient_frame + '_slice%02d.png'%(i+1)
        out_path = os.path.join(out_folder, out_name)
        if gt_flag:
            if np.min(fdata[:,:,i]) >= 0 and np.max(fdata[:,:,i]) < 4:
                cv2.imwrite(out_path, fdata[:, :, i] * 85)
        else:
            cv2.imwrite(out_path, fdata[:, :, i])
    logging.info('writed %s...'%patient_frame)
        
    
def load_data2fp_list(in_folder, pattern='.nii.gz'):
    """
    in_folder: training/patient...
    return:
        fp_list: nii file
        fp_list_gt: nii gt file
    """
    fp_list = []
    fp_list_gt = []
    pnum = 0
    for patient in os.listdir(in_folder):
        folder_path = os.path.join(in_folder, patient)
        if os.path.isdir(folder_path):
            pnum += 1
            for file in glob.glob(os.path.join(folder_path, 'patient???_frame??'+pattern)):
                gt_file = file.split('.nii.gz')[0] + '_gt'+pattern
                if os.path.exists(gt_file):
                    fp_list.append(file)
                    fp_list_gt.append(gt_file)

    return fp_list, fp_list_gt, pnum

def training2png(in_folder, out_folder, pattern='.nii.gz'):
    """
    prepare png for training img
    """

    fp_list, fp_list_gt, pnum = load_data2fp_list(in_folder, pattern)
    # fp_list, fp_list_gt, pnum = load_data2fp_list(in_folder, '.nii')
    logging.info('created fp list for %d patients..........'%pnum)

    for file,file_gt in zip(fp_list, fp_list_gt):
        niigz2png(file, out_folder, False)
        niigz2png(file_gt, out_folder, True)
    logging.info('writed all file to png, %d......'%pnum)


def testing2png(in_folder, out_folder, pattern='.nii.gz'):
    """
    prepare png for training img
    """

    fp_list, fp_list_gt, pnum = load_data2fp_list(in_folder, pattern)
    # fp_list, fp_list_gt, pnum = load_data2fp_list(in_folder, '.nii')
    logging.info('created fp list for %d patients..........'%pnum)

    for file,file_gt in zip(fp_list, fp_list_gt):
        niigz2png(file, out_folder, False)
        niigz2png(file_gt, out_folder, True)
    logging.info('writed all file to png, %d......'%pnum)

def create_dataloader(train_folder, test_folder, crop=True, size=(128,128), batch_size=10, scale=0.8, addDiff=False):
    """
    in_folder: png images path
    """
    train_fl = glob.glob(os.path.join(train_folder,'patient???_frame??_slice??.png'))
    train_gt_fl = glob.glob(os.path.join(train_folder,'patient???_frame??_gt_slice??.png'))
    if test_folder!=None:
        test_fl = glob.glob(os.path.join(test_folder,'patient???_frame??_slice??.png'))
        test_gt_fl = glob.glob(os.path.join(test_folder,'patient???_frame??_gt_slice??.png'))

    imgs = []
    labels = []
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []

    logging.info('start load from png files....')

    logging.info('start load from train files....')    
    for gt_file in train_gt_fl:
        file = gt_file.split('_gt')[0] + gt_file.split('_gt')[1]
        if file in train_fl:
            img = cv2.imread(file, 0)
            label = cv2.imread(gt_file, 0)
            if crop and img.shape[0] < size[0]:
                d = (size[0] - img.shape[0]) // 2 +1
                img = np.pad(img, ((d,d), (0,0)), 'constant')
            if crop and img.shape[1] < size[1]:
                d = (size[1] - img.shape[1]) // 2 +1
                img = np.pad(img, ((0,0), (d,d)), 'constant')
            if crop and label.shape[0] < size[0]:
                d = (size[0] - label.shape[0]) // 2 +1
                label = np.pad(label, ((d,d), (0,0)), 'constant')
            if crop and img.shape[1] < size[1]:
                d = (size[1] - label.shape[1]) // 2 +1
                label = np.pad(label, ((0,0), (d,d)), 'constant')

            imgs.append(img)
            labels.append(label // 85)
    logging.info('loaded all train files....')
    
    if test_folder != None:
        train_imgs = imgs
        train_labels = labels
        logging.info('start load from test files....')    
        for gt_file in test_gt_fl:
            file = gt_file.split('_gt')[0] + gt_file.split('_gt')[1]
            if file in test_fl:
                img = cv2.imread(file, 0)
                if crop and img.shape[0] < size[0]:
                    d = (size[0] - img.shape[0]) // 2 +1
                    img = np.pad(img, ((d,d), (0,0)), 'constant')
                if crop and img.shape[1] < size[1]:
                    d = (size[1] - img.shape[1]) // 2 +1
                    img = np.pad(img, ((0,0), (d,d)), 'constant')
                label = cv2.imread(gt_file, 0)
                if crop and label.shape[0] < size[0]:
                    d = (size[0] - label.shape[0]) // 2 +1
                    label = np.pad(label, ((d,d), (0,0)), 'constant')
                if crop and img.shape[1] < size[1]:
                    d = (size[1] - label.shape[1]) // 2 +1
                    label = np.pad(label, ((0,0), (d,d)), 'constant')

                test_imgs.append(img)
                test_labels.append(label // 85)
        logging.info('loaded all test files....')
    else:
        train_num = int(len(imgs) * scale)
        train_imgs = imgs[:train_num]
        test_imgs = imgs[train_num:]
        train_labels = labels[:train_num]
        test_labels = labels[train_num:]


    preprocTS = transforms.CenterCrop((size[0], size[1])) if crop else transforms.Resize((size[0], size[1]))
    preprocTSs = transforms.Compose([preprocTS])
    train_data = ACDCDataset(train_imgs, train_labels, preprocTSs)
    test_data = ACDCDataset(test_imgs, test_labels, preprocTSs)
    
    if addDiff:
        addTransformHV = transforms.Compose([
            preprocTS,
            transforms.RandomHorizontalFlip(1),
            transforms.RandomVerticalFlip(1)
        ])
        add_dataHV = ACDCDataset(train_imgs, train_labels, addTransformHV)
        train_data.imgs.extend(add_dataHV.imgs)
        train_data.labels.extend(add_dataHV.labels)

    train_dataLoader = DataLoader(train_data, batch_size, shuffle=True)
    test_dataLoader = DataLoader(test_data, batch_size, shuffle=True)
        
    return train_dataLoader, test_dataLoader
    

class ACDCDataset(Dataset):
    def __init__(self, imgs, labels, transforms):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img_tensor = torch.tensor(img, dtype=torch.float).unsqueeze(0)
        img_tensor = self.transforms(img_tensor)
        label_tensor = torch.tensor(label).unsqueeze(0)
        label_tensor = self.transforms(label_tensor)

        return img_tensor, label_tensor.squeeze(0)

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
    train_in = '/home/adl/acdc_segment/ACDC/database/training'
    train_out = '/home/adl/acdc_segment/ACDC/database/trainingPng'
    test_in = '/home/adl/acdc_segment/ACDC/database/testing'
    test_out = '/home/adl/acdc_segment/ACDC/database/testingPng'

    pattern='.nii.gz'
    training2png(train_in, train_out, pattern)
    testing2png(test_in, test_out, pattern)

    logging.info('wrote train and test file to png...')
