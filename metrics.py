import numpy as np
import torch
from scipy.spatial.distance import cdist


def _dice(label, pred, smooth=1e-5):

	intersection = torch.mul(label, pred).sum()
	return (2 * intersection + smooth) / (label.sum() + pred.sum() + smooth)


def dice_multi(label, pred, classes=4, smooth=1e-5):
	# shape: (batch, H, W)
	batch_size = label.shape[0]
	batch_dices = np.zeros((batch_size, classes))
	for i in range(batch_size):
		for j in range(classes):
			t = torch.where(label[i,:,:] ==j, 1, 0)
			p = torch.where(pred[i,:,:] ==j, 1, 0)
			batch_dices[i][j] = _dice(t, p, smooth)
	return np.mean(batch_dices, axis=0) # shape: (4,)

def iou_multi(label, pred, classes=4):
	# shape: (batch, H, W)
	ious = [0, 0, 0, 0]
	predf = pred.view(-1)
	labelf = label.view(-1)

	for cls in range(classes):
		pred_inds = predf == cls
		label_inds = labelf == cls

		intersection = (pred_inds[label_inds]).sum()
		union = pred_inds.sum() + label_inds.sum() - intersection
		ious[cls] = float(intersection) / float(max(union, 1))
	return ious


def _directed_hd95(c1, c2):
	if len(c1) == 0 or len(c2) == 0:
		return 0
	dists = cdist(c1, c2)
	# shape: [ c1.shape[0], c2.shape[0] ]
	min_dists = np.min(dists, axis=1)
	return np.percentile(min_dists, 95)

def _hd95(label, pred):
	# shape: [ H, W ]

	label_axis = np.transpose(np.nonzero(label))
	pred_axis = np.transpose(np.nonzero(pred))
	# shape: [n, 2]
	forward_hd95 = _directed_hd95(label_axis, pred_axis)
	reverse_hd95 = _directed_hd95(pred_axis, label_axis)

	return max(forward_hd95, reverse_hd95)


def hd95_multi(label, pred, classes=4):
	# shape: [batch, H, W]

	batch = label.shape[0]
	hd95s = np.zeros((batch, classes))
	for i in range(batch):
		for j in range(classes):
			t = torch.where(label[i,:,:] ==j, 1, 0)
			p = torch.where(pred[i,:,:] ==j, 1, 0)
			hd95s[i][j] = _hd95(t.to('cpu').numpy(), p.to('cpu').numpy())

	return np.mean(hd95s, axis=0)

