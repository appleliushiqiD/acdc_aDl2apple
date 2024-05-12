import os
from models import FCN8s, UNet, UNetM, UNetM12, UNetM34, AUNet, AUNetM12, AUNetM34
from metrics import dice_multi, iou_multi, hd95_multi

project_path = '/mnt/workspace'

train_folder = '/mnt/workspace/ACDC/database/trainingPng'
test_folder = '/mnt/workspace/ACDC/database/testingPng'

modelset = {'fcn': FCN8s(), 'unet': UNet(), 'unetm': UNetM(), 'unetm12': UNetM12(), 'unetm34': UNetM34(), 'aunet': AUNet(), 'aunetm12': AUNetM12(), 'aunetm34': AUNetM34()}
model_name = 'aunetm34'
model = modelset[model_name]

metricset = {'dice': dice_multi, 'iou': iou_multi, 'hd95': hd95_multi}
metric_name = 'dice'
metric_fn = metricset[metric_name]

img_size = (128,128)
# img_size = (224,224)
crop = True
split_scale = 0.8
addDiff = False

lossrate = 1e-3
# batch_size = 64
batch_size = 32
epochs = 60


# type: train, metr
train_type = "metr"

start_epoch = 0

base_name = '%s-%s-%s'%(model_name, "crop" if crop else "rescale", "diff" if addDiff else "nodiff")

model_load_path = os.path.join(project_path, 'model/%s-e%02d-l*.pth'%(base_name, start_epoch+60))
# model_load_path = os.path.join(project_path, 'model/%s-*-test-e%02d-*.pth'%(base_name, ))

model_save_path = os.path.join(project_path, 'model/%s'%(base_name))

img_path = os.path.join(project_path, 'img/%s-e%02d'%(base_name, start_epoch+epochs))

metr_path = os.path.join(project_path, 'metricdata/%s-e%02d'%(base_name, start_epoch+epochs))

log_dir = os.path.join(project_path, "logdir" if \
                       train_type=="train" else "mtlogdir-train")
log_path = os.path.join(log_dir, '%s-e%02d-log.txt'%(base_name, start_epoch+epochs) if \
                        train_type=="train" else '%s-e%02d-metric-log.txt'%(base_name, start_epoch+epochs))





