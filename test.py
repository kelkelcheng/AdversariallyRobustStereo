from __future__ import print_function
import argparse
import os
import torch.nn.parallel
from torch.autograd import Variable
from dataloader import readPFM
import numpy as np
from PIL import Image
import skimage.io

# Training settings
parser = argparse.ArgumentParser(description='Evaluation of Stereo Matching Models')
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--left_image', type=str, default='examples/left.png', help="left image")
parser.add_argument('--right_image', type=str, default='examples/right.png', help="right image")
parser.add_argument('--crop_height', type=int, default=384, help="crop height")
parser.add_argument('--crop_width', type=int, default=1248, help="crop width")
parser.add_argument('--backbone', type=bool, default=False, help='if the backbone is used')
parser.add_argument('--pretrained', type=bool, default=False, help='if the pretrained model is used')
opt = parser.parse_args()

if opt.backbone:
    from models.MCTNet_backbone import Model
    opt.resume = 'checkpoint/MCTNet/MCTNet_backbone_kitti.pth'
    if opt.pretrained:
        opt.resume = 'checkpoint/MCTNet/MCTNet_backbone_sf_epoch_20.pth'
else:
    from models.MCTNet import Model
    opt.resume = 'checkpoint/MCTNet/MCTNet_kitti.pth'
    if opt.pretrained:
        opt.resume = 'checkpoint/MCTNet/MCTNet_sf_epoch_20.pth'

model = Model(opt.max_disp)
model.training = False
print(opt)

# load model
print("loading the model")
model = torch.nn.DataParallel(model).cuda()
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

left = Image.open(opt.left_image)
right = Image.open(opt.right_image)

# cast to float
size = np.shape(left)
height = size[0]
width = size[1]
temp_data = np.zeros([6, height, width], 'float32')
left = np.asarray(left).astype(float)
right = np.asarray(right).astype(float)

# normalization
mean_left = mean_right = np.array([0.485, 0.456, 0.406])
std_left = std_right = np.array([0.229, 0.224, 0.225])
left /= 255.
right /= 255.

# crop data
temp_data[0:3, :, :] = np.moveaxis((left[:,:,:3] - mean_left) / std_left, -1, 0)
temp_data[3:6, :, :] = np.moveaxis((right[:,:,:3] - mean_right) / std_right, -1, 0)

if height <= opt.crop_height and width <= opt.crop_width:
    temp = temp_data
    temp_data = np.zeros([6, opt.crop_height, opt.crop_width], 'float32')
    temp_data[:, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width] = temp

input1_np = np.expand_dims(temp_data[0:3], axis=0)
input2_np = np.expand_dims(temp_data[3:6], axis=0)

# start to evaluate
model.eval()

# to gpu
input1 = Variable(torch.from_numpy(input1_np), requires_grad=False).cuda()
input2 = Variable(torch.from_numpy(input2_np), requires_grad=False).cuda()

# compute disparity
disp = model(input1, input2).detach()

# store the disparity map
temp = disp.cpu().numpy()
if height <= opt.crop_height and width <= opt.crop_width:
    temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
else:
    temp = temp[0, :, :]
skimage.io.imsave("examples/disp.png", (temp * 256).astype('uint16'))