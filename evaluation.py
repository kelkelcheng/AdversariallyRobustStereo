from __future__ import print_function
import argparse
import os
import torch.nn.parallel
from torch.autograd import Variable
from dataloader import readPFM
import numpy as np
from PIL import Image

# test settings
parser = argparse.ArgumentParser(description='Evaluation of Stereo Matching Models')
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--data_path', type=str, default='../../data/', help="data root")
parser.add_argument('--dataset', type=int, default=1, help='1: sceneflow, 2: kitti. 3: kitti 2015')
parser.add_argument('--whichModel', type=int, default=2, help='0 for GANet, 1 for PSMNet, 2 for this method')
parser.add_argument('--backbone', type=bool, default=False, help='if the backbone is used')
parser.add_argument('--pretrained', type=bool, default=False, help='if the pretrained model is used')
opt = parser.parse_args()

if opt.dataset == 1:
    opt.test_data_path = opt.data_path + 'FlyingThings3D/'
    opt.val_list = './lists/sceneflow_test.list'
    opt.crop_height = 576
    opt.crop_width = 960
elif opt.dataset == 2:
    opt.test_data_path = opt.data_path + 'KITTI2012/training/'
    opt.val_list = './lists/kitti2012_train.list'
    opt.crop_height = 384
    opt.crop_width = 1248
else:
    opt.test_data_path = opt.data_path + 'KITTI2015/training/'
    opt.val_list = './lists/kitti2015_train.list'
    opt.crop_height = 384
    opt.crop_width = 1248

if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
torch.cuda.manual_seed(123)

# select the model
print('===> Building model')
if opt.whichModel == 0:
    from models.GANet_deep import GANet
    model = GANet(opt.max_disp)
    opt.resume = 'checkpoint/GANet/kitti2015_final.pth'
    if opt.dataset == 1 or opt.pretrained:
        opt.resume = 'checkpoint/GANet/sceneflow_epoch_10.pth'
elif opt.whichModel == 1:
    from models.PSMNet import *
    model = stackhourglass(opt.max_disp)
    opt.resume = 'checkpoint/PSMNet/pretrained_model_KITTI2015.tar'
    if opt.dataset == 1 or opt.pretrained:
        opt.resume = 'checkpoint/PSMNet/pretrained_sceneflow.tar'
        opt.psm_constant = 1.17
    if opt.crop_height == 240:
        opt.crop_height = 256
else:
    if opt.backbone:
        from models.MCTNet_backbone import Model
        opt.resume = 'checkpoint/MCTNet/MCTNet_backbone_kitti.pth'
        if opt.dataset == 1:
            opt.resume = 'checkpoint/MCTNet/MCTNet_backbone_sf_epoch_20.pth'
    else:
        from models.MCTNet import Model
        opt.resume = 'checkpoint/MCTNet/MCTNet_kitti.pth'
        if opt.dataset == 1:
            opt.resume = 'checkpoint/MCTNet/MCTNet_sf_epoch_20.pth'

    model = Model(opt.max_disp)
    model.training = False
print(opt)

model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def fetch_data(A, crop_height, crop_width):
    ''' self-contained data extraction '''
    # parse data name
    if opt.dataset == 1:
        filename_l = opt.test_data_path + 'frames_finalpass/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'frames_finalpass/' + A[0: len(A) - 14] + 'right/' + A[len(A) - 9:len(A) - 1]
        filename_disp = opt.test_data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
        disp_left, height, width = readPFM(filename_disp)
    elif opt.dataset == 2:
        filename_l = opt.test_data_path + 'colored_0/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'colored_1/' + A[0: len(A) - 1]
        filename_disp = opt.test_data_path + 'disp_occ/' + A[0: len(A) - 1]
        disp_left = np.asarray(Image.open(filename_disp)).astype(float)
    elif opt.dataset == 3:
        filename_l = opt.test_data_path + 'image_2/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'image_3/' + A[0: len(A) - 1]
        filename_disp = opt.test_data_path + 'disp_occ_0/' + A[0: len(A) - 1]
        # filename_disp = opt.test_data_path + 'disp_noc_0/' + A[0: len(A) - 1]
        disp_left = np.asarray(Image.open(filename_disp)).astype(float)

    left = Image.open(filename_l)
    right = Image.open(filename_r)

    # cast to float
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left).astype(float)
    right = np.asarray(right).astype(float)

    # normalization
    if opt.whichModel == 0:
        mean_left = np.array([np.mean(left[:, :, 0]), np.mean(left[:, :, 1]), np.mean(left[:, :, 2])])
        std_left = np.array([np.std(left[:, :, 0]), np.std(left[:, :, 1]), np.std(left[:, :, 2])])

        mean_right = np.array([np.mean(right[:, :, 0]), np.mean(right[:, :, 1]), np.mean(right[:, :, 2])])
        std_right = np.array([np.std(right[:, :, 0]), np.std(right[:, :, 1]), np.std(right[:, :, 2])])
    else:
        mean_left = mean_right = np.array([0.485, 0.456, 0.406])
        std_left = std_right = np.array([0.229, 0.224, 0.225])
        left /= 255.
        right /= 255.

    temp_data[0:3, :, :] = np.moveaxis((left[:,:,:3] - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right[:,:,:3] - mean_right) / std_right, -1, 0)

    # ignore disparities that are smaller than a threshold
    disp_left[disp_left < 0.01] = width * 2 * 256
    if opt.dataset != 1:
        disp_left = disp_left / 256.

    # range mask
    mask_min = (disp_left < opt.max_disp).astype(float)

    temp_data[6, :, :] = width * 2
    temp_data[6, :, :] = disp_left
    temp_data[7, :, :] = mask_min

    # crop data
    if height <= crop_height and width <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - height: crop_height, crop_width - width: crop_width] = temp
    else:
        start_x = int((width - crop_width) / 2)
        start_y = int((height - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    input1_np = np.expand_dims(temp_data[0:3], axis=0)
    input2_np = np.expand_dims(temp_data[3:6], axis=0)
    target_np = np.expand_dims(temp_data[6:7], axis=0)
    mask_min_np = np.expand_dims(temp_data[7:8], axis=0).astype(bool)

    return input1_np, input2_np, target_np, mask_min_np

if __name__ == '__main__':
    # initialize
    f = open(opt.val_list, 'r')
    file_list = f.readlines()
    file_list.sort()

    # thresholds for the error rates
    thr_list = [1,2,3]

    data_total = len(file_list)
    mask_min_loss_list = np.zeros(data_total)
    mask_max_loss_list = np.zeros(data_total)
    whole_loss_list = np.zeros(data_total)

    err_mask_min_list = np.zeros((len(thr_list), data_total))
    err_mask_max_list = np.zeros((len(thr_list), data_total))
    err_whole_list = np.zeros((len(thr_list), data_total))

    # start to evaluate
    model.eval()
    for data_num in range(data_total):
        A = file_list[data_num]
        input1_np, input2_np, target_np, mask_min_np = fetch_data(A, opt.crop_height, opt.crop_width)

        # from np to torch
        input1 = Variable(torch.from_numpy(input1_np), requires_grad=False)
        input2 = Variable(torch.from_numpy(input2_np), requires_grad=False)
        target = Variable(torch.from_numpy(target_np), requires_grad=False)
        target = torch.squeeze(target, 1)
        mask_min = Variable(torch.from_numpy(mask_min_np), requires_grad=False)
        mask_min = torch.squeeze(mask_min, 1)

        # to gpu
        input1 = input1.cuda()
        input2 = input2.cuda()
        target = target.cuda()
        mask_min = mask_min.cuda()

        # compute disparity
        disp = model(input1, input2).detach()
        if opt.whichModel==1 and opt.pretrained:
            disp = disp * opt.psm_constant

        # compute EPE, bad 1.0, and bad 3.0
        diff_mask_min = torch.abs(disp[mask_min] - target[mask_min]).detach()
        mask_min_loss = torch.mean(diff_mask_min).detach()
        mask_min_loss_list[data_num] = mask_min_loss.item()
        mask_min_total = mask_min_np.sum()
        diff_mask_min = diff_mask_min.cpu().numpy()

        for idx, thr in enumerate(thr_list):
            # over threshold error rates
            err_mask_min_thr = (diff_mask_min > thr).sum() / mask_min_total
            err_mask_min_list[idx, data_num] = err_mask_min_thr

            if thr==3:
                print("data", data_num+1, A[:-1], "EPE", mask_min_loss.item(), "| error rate (" + str(thr) + " px):", err_mask_min_thr)

    print("number of nans:", np.isnan(mask_min_loss_list).sum())
    print("loss mean:", mask_min_loss_list[~np.isnan(mask_min_loss_list)].mean())

    for idx, thr in enumerate(thr_list):
        print("mean error rate for threshold (" + str(thr) + " px):", err_mask_min_list[idx, :][~np.isnan(err_mask_min_list[idx, :])].mean())
