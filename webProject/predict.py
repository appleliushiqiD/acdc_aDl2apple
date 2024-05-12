import os
import cv2
import numpy as np
import glob

import torch
from torchvision import transforms

from models import FCN8s, UNet, AUNet
from tt import save_load_model

def predict(model, img, device):
	with torch.no_grad():
		x = img.to(device)
		pre = model(x)
		pred = torch.argmax(pre, dim=1)
		return pred

def pre_crop_rescale(img, cs, hwratio=1):
	'''
	x, y = img.shape
	nx, ny = x, y
	if (x/y) > hwratio:
		ny = y-(y % 32)
		nx = ny * hwratio
		nx = nx-(nx % 32)    
	elif (ny/nx) > hwratio:
		nx = x-(x % 32)
		ny = nx * hwratio
		ny = ny-(ny % 32)
	else:
		nx = x-(x % 32)
		ny = y-(y % 32)
	'''
	img_tensor = torch.tensor(img, dtype=torch.float)
	img_tensor = torch.unsqueeze(torch.unsqueeze(img_tensor, 0) ,0)
	if cs == "crop":
		# img_tensor = transforms.CenterCrop((nx,ny))(img_tensor)
		img_tensor = transforms.CenterCrop((128,128))(img_tensor)
	elif cs == "rescale":
		# img_tensor = transforms.Resize((nx,ny))(img_tensor)
		img_tensor = transforms.Resize((224,224))(img_tensor)
	# return shape (1, 1, nx, ny)
	return img_tensor

def enColour(img, mask):
	img_rgb = np.repeat(np.expand_dims(img, 2), 3, 2)
	img_mask = np.repeat(np.expand_dims(mask, 2), 3, 2)

	img_mask = np.where(img_rgb==[1,1,1], [240,133,45], img_mask)
	img_mask = np.where(img_rgb==[2,2,2], [60,67,244], img_mask)
	img_mask = np.where(img_rgb==[3,3,3], [88,168,10], img_mask)
	return img_mask


def runPredict(file_path, model, cs="crop", cotype="gray", cfg=None):
	"""
	cfg: a dict type, include: hwratio, save_path,
	"""
	if not (os.path.exists(file_path) and os.path.isfile(file_path)):
		return False
	img = cv2.imread(file_path, 0)

	device = ("cuda" if torch.cuda.is_available() else "cpu")
	img_tensor = pre_crop_rescale(img, cs, cfg['hwratio']).to(device)

	pre = predict(model, img_tensor, device)
	img_pre = pre.numpy().squeeze()

	img_cs = img_tensor.numpy().squeeze()
	img_co = img_pre * 85
	if cotype == "colourful":
		img_co = enColour(img_pre, img_pre)
	elif cotype == "colourfulmask":
		img_co = enColour(img_pre, img_cs)

	cv2.imwrite(cfg['save_cs'], img_cs)
	cv2.imwrite(cfg['save_co'], img_co)
	return True


if __name__ == '__main__':

	"""
	load a gray image to predict
	"""

	modelset = { 'unet': UNet(), 'aunet': AUNet(), 'fcn': FCN8s() }

	model_dir = './modelbest'
	for mn in modelset.keys():
		model_path = os.path.join(model_dir, '%s-*.pth'%(mn))
		model_paths = glob.glob(model_path)
		if len(model_paths) ==0:
			print('no model for %s'%mn)
		else:
			modelset[mn] = save_load_model(modelset[mn], model_paths[0], "load")

	file_path = "./images/in.png"

	pcfg = { 'hwratio': 1.5,
			'save_cs': './images/pre_proc.png',
			'save_co': './images/pre_out.png',}

	fPredict(file_path, modelset['fcn'], cs="crop", cotype="colourfulmask", cfg=pcfg)





