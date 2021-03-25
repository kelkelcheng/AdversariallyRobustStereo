from __future__ import print_function
import argparse
import os
import torch
import torch.nn.parallel
from torch.autograd import Variable
from dataloader import readPFM
import numpy as np
from PIL import Image

# settings
parser = argparse.ArgumentParser(description='Stereo-Constrained PGD Attack')
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--crop_height', type=int, default=240, help="crop height")
parser.add_argument('--crop_width', type=int, default=384, help="crop width")
parser.add_argument('--data_path', type=str, default='../../data/', help="data root")
parser.add_argument('--dataset', type=int, default=3, help='1: sceneflow, 3: kitti 2015')
parser.add_argument('--whichModel', type=int, default=2, help='0 for GANet, 1 for PSMNet, 2 for this method')
parser.add_argument('--total_iter', type=int, default=20, help='iterations of PGD attack')
parser.add_argument('--e', type=float, default=0.03, help='epsilon of PGD attack')
parser.add_argument('--a', type=float, default=0.01, help='step size of PGD attack')
parser.add_argument('--double_occ', type=bool, default=False, help='if occlusion of the right image is excluded')
parser.add_argument('--backbone', type=bool, default=False, help='if the backbone is used')
parser.add_argument('--unconstrained_attack', type=bool, default=False, help='if the backbone is used')
opt = parser.parse_args()

# select file list according to dataset
if opt.dataset == 1:
    opt.test_data_path = opt.data_path + 'FT_subset/val/'
    opt.val_list = './lists/sceneflow_subset_val_1000.list'
elif opt.dataset == 2:
    opt.test_data_path = opt.data_path + 'KITTI2012/training/'
    opt.val_list = './lists/kitti2012_train.list'
else:
    opt.test_data_path = opt.data_path + 'KITTI2015/training/'
    opt.val_list = './lists/kitti2015_train.list'

if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# select the model
print('===> Building model')
if opt.whichModel == 0:
    from models.GANet_deep import GANet
    model = GANet(opt.max_disp)
    opt.resume = 'checkpoint/GANet/kitti2015_final.pth'
    if opt.dataset == 1:
        opt.resume = 'checkpoint/GANet/sceneflow_epoch_10.pth'
elif opt.whichModel == 1:
    from models.PSMNet import *
    model = stackhourglass(opt.max_disp)
    opt.resume = 'checkpoint/PSMNet/pretrained_model_KITTI2015.tar'
    if opt.dataset == 1:
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
print("load parameters:", opt.resume)
model = torch.nn.DataParallel(model).cuda()

# load trained parameters
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

def fetch_data(A, crop_height=240, crop_width=576):
    if opt.dataset == 1:
        filename_l = opt.test_data_path + 'frames_finalpass/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'frames_finalpass/' + 'right/' + A[5:len(A) - 1]

        filename_disp = opt.test_data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
        disp_left, height, width = readPFM(filename_disp)
        disp_left = -disp_left

        filename_disp_r = opt.test_data_path + 'disparity/' + 'right/' + A[5:len(A) - 4] + 'pfm'
        disp_right, height, width = readPFM(filename_disp_r)

        filename_occ = opt.test_data_path + 'disparity_occlusions/' + A[0: len(A) - 1]
        occ_left = Image.open(filename_occ)
        occ_left = np.asarray(occ_left)
        occ_left = occ_left | (disp_left >= opt.max_disp)

        filename_occ_r = opt.test_data_path + 'disparity_occlusions/' + 'right/' + A[5:len(A) - 1]
        occ_right  = Image.open(filename_occ_r)
        occ_right = np.asarray(occ_right)
        occ_right = occ_right | (occ_right >= opt.max_disp)

    elif opt.dataset == 3:
        filename_l = opt.test_data_path + 'image_2/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'image_3/' + A[0: len(A) - 1]
        filename_disp = opt.test_data_path + 'disp_noc_0/' + A[0: len(A) - 1]
        filename_disp_2 = opt.test_data_path + 'disp_noc_1/' + A[0: len(A) - 1]
        disp_left = np.asarray(Image.open(filename_disp)).astype(float)
        disp_right = np.asarray(Image.open(filename_disp_2)).astype(float)

    left = Image.open(filename_l)
    right = Image.open(filename_r)

    # cast to float
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([10, height, width], 'float32')
    left = np.asarray(left).astype(float)
    right = np.asarray(right).astype(float)

    # generate masks for KITTI2015
    if opt.dataset != 1:
        disp_left[disp_left < 0.01] = width * 2 * 256
        disp_left = disp_left / 256.
        occ_left = (disp_left >= opt.max_disp).astype(float)

        disp_right[disp_right < 0.01] = width * 2 * 256
        disp_right = disp_right / 256.
        occ_right = (disp_right >= opt.max_disp).astype(float)

    # normalization
    scale = 1.0
    if opt.whichModel == 0:
        mean_left = np.array([np.mean(left[:, :, 0]), np.mean(left[:, :, 1]), np.mean(left[:, :, 2])])
        std_left = np.array([np.std(left[:, :, 0]), np.std(left[:, :, 1]), np.std(left[:, :, 2])])

        mean_right = np.array([np.mean(right[:, :, 0]), np.mean(right[:, :, 1]), np.mean(right[:, :, 2])])
        std_right = np.array([np.std(right[:, :, 0]), np.std(right[:, :, 1]), np.std(right[:, :, 2])])

        scale = 255.0
    else:
        mean_left = mean_right = np.array([0.485, 0.456, 0.406])
        std_left = std_right = np.array([0.229, 0.224, 0.225])
        left /= 255.
        right /= 255.

    # set 0 and 255 as boundary values for attacks
    rgb_min_l = -mean_left / std_left
    rgb_max_l = (scale - mean_left) / std_left

    rgb_min_r = -mean_right / std_right
    rgb_max_r = (scale - mean_right) / std_right

    temp_data[0:3, :, :] = np.moveaxis((left - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right - mean_right) / std_right, -1, 0)

    temp_data[6, :, :] = width * 2
    temp_data[6, :, :] = disp_left

    temp_data[7, :, :] = occ_left.astype(float)
    temp_data[8, :, :] = occ_right.astype(float)

    temp_data[9, :, :] = width * 2
    temp_data[9, :, :] = disp_right

    # crop data
    if height <= crop_height and width <= crop_width:
        temp = temp_data
        temp_data = np.zeros([9, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - height: crop_height, crop_width - width: crop_width] = temp

        # set the filled-in areas as occluded to avoid to count as results
        temp_data[7, 0:crop_height - height, :] = 1.0
        temp_data[7, :, 0:crop_width - width] = 1.0
        temp_data[8, 0:crop_height - height, :] = 1.0
        temp_data[8, :, 0:crop_width - width] = 1.0
    else:
        start_x = int((width - crop_width) / 2)
        start_y = int((height - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]


    input1_np = np.expand_dims(temp_data[0:3], axis=0)
    input2_np = np.expand_dims(temp_data[3:6], axis=0)
    target_np = np.expand_dims(temp_data[6:7], axis=0)
    occ_np = np.expand_dims(temp_data[7:8], axis=0)
    occ_np = occ_np.astype(bool)
    occ_2_np = np.expand_dims(temp_data[8:9], axis=0)
    occ_2_np = occ_2_np.astype(bool)
    target_2_np = np.expand_dims(temp_data[9:10], axis=0)
    info = {'rgb_min_l': rgb_min_l, 'rgb_min_r': rgb_min_r,
            'rgb_max_l': rgb_max_l, 'rgb_max_r': rgb_max_r,
            'mean_right': mean_right, 'mean_left': mean_left,
            'std_right': std_right, 'std_left': std_left}

    return input1_np, input2_np, target_np, target_2_np, occ_np, occ_2_np, info


def unconstrained_projected_gradient_descent(model, x1, x2, y, occ, num_steps, step_size, step_norm, eps, eps_norm,
                               rgb_min_l, rgb_max_l, rgb_min_r, rgb_max_r):
    """Performs the projected gradient descent attack on a batch of images."""
    # new
    batch_size, channels, im_h, im_w = x1.detach().cpu().numpy().shape
    x_adv = torch.zeros([batch_size, channels, im_h, im_w], requires_grad=True, device='cuda')
    x2_adv = torch.zeros([batch_size, channels, im_h, im_w], requires_grad=True, device='cuda')
    zero_plane = torch.zeros([batch_size, channels, im_h, im_w], requires_grad=False, device='cuda')
    err_list = np.zeros(num_steps)

    _x_adv = x_adv.clone().detach().requires_grad_(True)
    _x2_adv = x2_adv.clone().detach().requires_grad_(True)

    for i in range(num_steps):
        # new
        _x_adv = x_adv.clone().detach().requires_grad_(True)
        _x2_adv = x2_adv.clone().detach().requires_grad_(True)

        input1 = x1 + _x_adv
        input2 = x2 + _x2_adv
        # clamp out of range values
        input1 = input1.reshape(im_h, im_w, channels)
        input1 = torch.max(torch.min(input1, rgb_max_l), rgb_min_l)
        input1 = input1.reshape(batch_size, channels, im_h, im_w)

        input2 = input2.reshape(im_h, im_w, channels)
        input2 = torch.max(torch.min(input2, rgb_max_r), rgb_min_r)
        input2 = input2.reshape(batch_size, channels, im_h, im_w)

        # compute disp and loss
        if opt.whichModel==2:
            prediction = model(input1, input2, attack=True)
        elif opt.whichModel==1 and opt.dataset==1: # according to their repo, they disp need to *1.17 for SceneFLow
            prediction = model(input1, input2) * opt.psm_constant
        else:
            prediction = model(input1, input2)
        abs_diff = torch.abs(prediction[~occ] - y[~occ])
        loss = torch.mean(abs_diff)

        # bad 3.0 [%]
        thr = 3
        diff_over_thr = ((abs_diff > thr).sum()).item() / ((~occ).sum()).item()
        print("iter", i, "loss:", loss.item(), "err rate:", diff_over_thr)
        err_list[i] = diff_over_thr

        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step_size
                gradients_2 = _x2_adv.grad.sign() * step_size

            x_adv += gradients
            x2_adv += gradients_2

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            x_adv = torch.max(torch.min(x_adv, zero_plane + eps), zero_plane - eps)
            x2_adv = torch.max(torch.min(x2_adv, zero_plane + eps), zero_plane - eps)

    input1 = x1 + _x_adv
    input2 = x2 + _x2_adv

    input1 = input1.reshape(im_h, im_w, channels)
    input1 = torch.max(torch.min(input1, rgb_max_l), rgb_min_l)
    input1 = input1.reshape(batch_size, channels, im_h, im_w)

    input2 = input2.reshape(im_h, im_w, channels)
    input2 = torch.max(torch.min(input2, rgb_max_r), rgb_min_r)
    input2 = input2.reshape(batch_size, channels, im_h, im_w)

    return input1.detach(), input2.detach(), err_list


def projected_gradient_descent(model, x1, x2, y, occ, mask, occ_mask, occ_2_mask, num_steps, step_size, step_norm, eps, eps_norm,
                               rgb_min_l, rgb_max_l, rgb_min_r, rgb_max_r):
    """Performs the projected gradient descent attack on a batch of images."""
    # initialization
    batch_size, channels, im_h, im_w = x1.detach().cpu().numpy().shape
    noise = torch.zeros([batch_size, channels, im_h, im_w], requires_grad=True, device='cuda')
    zero_plane = torch.zeros([batch_size, channels, im_h, im_w], requires_grad=False, device='cuda')
    assert (channels == 3)
    num_channels = noise.shape[1]
    err_list = np.zeros(num_steps)

    # start attack
    for i in range(num_steps):
        _x_adv = noise.clone().detach().requires_grad_(True)

        input2_noise = x2 + _x_adv

        # fetch correspondence from the right image
        noise_extract = torch.gather(_x_adv, channels, mask)
        noise_extract[occ_mask] = zero_plane[occ_mask]
        input_extract = x1 + noise_extract

        # clamp out of range values
        input_extract = input_extract.reshape(im_h, im_w, channels)
        input2_noise = input2_noise.reshape(im_h, im_w, channels)

        input_extract = torch.max(torch.min(input_extract, rgb_max_l), rgb_min_l)
        input2_noise = torch.max(torch.min(input2_noise, rgb_max_r), rgb_min_r)

        input_extract = input_extract.reshape(batch_size, channels, im_h, im_w)
        input2_noise = input2_noise.reshape(batch_size, channels, im_h, im_w)

        # compute disp and loss
        if opt.whichModel==2:
            # attack=True to enable attacking census transform
            prediction = model(input_extract, input2_noise, attack=True)
        elif opt.whichModel==1 and opt.dataset==1: # according to their repo, they disp need to *1.17 for SceneFLow
            prediction = model(input_extract, input2_noise) * opt.psm_constant
        else:
            prediction = model(input_extract, input2_noise)

        # only compute errors of non-occluded regions
        abs_diff = torch.abs(prediction[~occ] - y[~occ])
        loss = torch.mean(abs_diff)

        # bad 3.0 [%]
        thr = 3
        diff_over_thr = ((abs_diff > thr).sum()).item() / ((~occ).sum()).item()
        print("iter", i, "loss:", loss.item(), "err rate:", diff_over_thr)
        err_list[i] = diff_over_thr

        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step_size
            else:
                # Note .view() assumes batched image data as 4D tensor
                gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1) \
                    .norm(step_norm, dim=-1) \
                    .view(-1, num_channels, 1, 1)

            # update the perturbation
            noise += gradients

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            noise = torch.max(torch.min(noise, zero_plane + eps), zero_plane - eps)
            # double_occ=True disable the occluded regions of the right image
            if opt.double_occ:
                noise[occ_2_mask] = zero_plane[occ_2_mask]
        else:
            mask = noise.view(noise.shape[0], -1).norm(eps_norm, dim=1) <= eps

            scaling_factor = noise.view(noise.shape[0], -1).norm(eps_norm, dim=1)
            scaling_factor[mask] = eps

            # .view() assumes batched images as a 4D Tensor
            noise *= eps / scaling_factor.view(-1, 1, 1, 1)

    # apply the final iteration
    if opt.double_occ:
        noise[occ_2_mask] = zero_plane[occ_2_mask]

    input2_noise = x2.clone()
    input2_noise += noise.detach()

    noise_extract = torch.gather(noise, channels, mask)
    noise_extract[occ_mask] = zero_plane[occ_mask]
    input_extract = x1 + noise_extract

    # clamp
    input_extract = input_extract.reshape(im_h, im_w, channels)
    input2_noise = input2_noise.reshape(im_h, im_w, channels)

    input_extract = torch.max(torch.min(input_extract, rgb_max_l), rgb_min_l)
    input2_noise = torch.max(torch.min(input2_noise, rgb_max_r), rgb_min_r)

    input_extract = input_extract.reshape(batch_size, channels, im_h, im_w)
    input2_noise = input2_noise.reshape(batch_size, channels, im_h, im_w)

    return input_extract.detach(), input2_noise.detach(), err_list


def my_mean(temp):
    return temp[~np.isnan(temp)].mean()


if __name__ == '__main__':

    # preprocessing
    f = open(opt.val_list, 'r')
    file_list = f.readlines()

    # initialize lists to keep records
    data_total = len(file_list)
    before_loss_list = np.zeros(data_total)
    after_loss_list = before_loss_list.copy()
    diff_over_thr_3_ori_list = before_loss_list.copy()
    diff_over_thr_1_ori_list = before_loss_list.copy()
    diff_over_thr_3_list = before_loss_list.copy()
    diff_over_thr_1_list = before_loss_list.copy()
    err_iter_list = np.zeros((data_total, opt.total_iter))

    # start to loop through data
    model.eval()
    for data_num in range(data_total):
        A = file_list[data_num]

        # fetch data
        input1_np, input2_np, target_np, target_2_np, occ_np, occ_2_np, info = fetch_data(A, opt.crop_height, opt.crop_width)

        # fetch min and max values to clamp attack perturbations
        rgb_min_l, rgb_min_r = info['rgb_min_l'], info['rgb_min_r']
        rgb_max_l, rgb_max_r = info['rgb_max_l'], info['rgb_max_r']
        rgb_min_l = torch.tensor(rgb_min_l).cuda().float()
        rgb_min_r = torch.tensor(rgb_min_r).cuda().float()
        rgb_max_l = torch.tensor(rgb_max_l).cuda().float()
        rgb_max_r = torch.tensor(rgb_max_r).cuda().float()
        mean_right, mean_left = info['mean_right'], info['mean_left']
        std_right, std_left = info['std_right'], info['std_left']

        # from np to torch
        input1 = Variable(torch.from_numpy(input1_np), requires_grad=False)
        input2 = Variable(torch.from_numpy(input2_np), requires_grad=True)
        target = Variable(torch.from_numpy(target_np), requires_grad=False)
        target2 = Variable(torch.from_numpy(target_2_np), requires_grad=False)
        occ = Variable(torch.from_numpy(occ_np), requires_grad=False)
        occ_2 = Variable(torch.from_numpy(occ_2_np), requires_grad=False)

        # mask is the indices for fetching noise
        mask = torch.linspace(0, opt.crop_width - 1, steps=opt.crop_width, requires_grad=True)
        mask = mask.repeat(target.size()[0], target.size()[1], target.size()[2], 1)
        mask = mask - target
        mask = mask.round().long()

        # set those with out-of-crop correspondence as occluded
        occ = occ | (mask < 0)
        occ = torch.squeeze(occ, 1)

        mask = torch.clamp(mask, 0, opt.crop_width - 1)
        mask = mask.repeat(1, 3, 1, 1)

        # for the right image
        mask2 = torch.linspace(0, opt.crop_width-1, steps=opt.crop_width, requires_grad=True)
        mask2 = mask2.repeat(target2.size()[0], target2.size()[1], target2.size()[2], 1)
        mask2 = mask2 + target2

        occ_2 = occ_2 | (mask2 >= opt.crop_width)
        occ_2 = torch.squeeze(occ_2, 1)

        # occ_mask is occ repeated for RGB channels
        occ_mask = occ.repeat(1, 3, 1, 1)
        occ_2_mask = occ_2.repeat(1, 3, 1, 1)

        # to gpu
        input1 = input1.cuda()
        input2 = input2.cuda()
        target = target.cuda()
        occ = occ.cuda()
        mask = mask.cuda()
        occ_mask = occ_mask.cuda()
        occ_2_mask = occ_2_mask.cuda()

        target = torch.squeeze(target, 1)

        # the error before attacks
        ori_disp = model(input1, input2).detach()
        if opt.whichModel==1 and opt.dataset==1: # according to their repo, they disp need to *1.17 for SceneFLow
            ori_disp = ori_disp * opt.psm_constant
        before_loss = torch.mean(torch.abs(ori_disp[~occ] - target[~occ])).detach()
        print("data", data_num, "before_loss", before_loss.item())
        before_loss_list[data_num] = before_loss.item()

        # stereo-constrained attack - x1 and x2 are the adversarial inputs
        if not opt.unconstrained_attack:
            x1, x2, err_list = projected_gradient_descent(model, input1, input2, target, occ, mask, occ_mask, occ_2_mask,
                                                num_steps=opt.total_iter, step_size=opt.a,
                                                eps=opt.e, eps_norm='inf',
                                                step_norm='inf',
                                                rgb_min_l=rgb_min_l, rgb_max_l=rgb_max_l,
                                                rgb_min_r=rgb_min_r, rgb_max_r=rgb_max_r)
        # unconstrained attack
        else:
            x1, x2, err_list = unconstrained_projected_gradient_descent(model, input1, input2, target, occ,
                                                                        num_steps=opt.total_iter, step_size=opt.a,
                                                                        eps=opt.e, eps_norm='inf',
                                                                        step_norm='inf',
                                                                        rgb_min_l=rgb_min_l, rgb_max_l=rgb_max_l,
                                                                        rgb_min_r=rgb_min_r, rgb_max_r=rgb_max_r)
        err_iter_list[data_num, :] = err_list

        # the error after attacks
        attack_disp = model(x1, x2).detach()
        if opt.whichModel==1 and opt.dataset==1: # according to their repo, they disp need to *1.17 for SceneFLow
            attack_disp = attack_disp * opt.psm_constant
        after_loss = torch.mean(torch.abs(attack_disp[~occ] - target[~occ])).detach()
        print("data", data_num, "after_loss", after_loss.item())
        after_loss_list[data_num] = after_loss.item()

        # record EPE, bad 1.0, bad 3.0
        thr = 3
        diff_ori = torch.abs(ori_disp[~occ] - target[~occ]).detach().cpu().numpy()
        diff_over_thr_ori = (diff_ori > thr).sum() / (~occ_np).sum()
        print("data", data_num, "Original error rate (3 px):", diff_over_thr_ori)
        diff_over_thr_3_ori_list[data_num] = diff_over_thr_ori

        diff = torch.abs(attack_disp[~occ] - target[~occ]).detach().cpu().numpy()
        diff_over_thr = (diff > thr).sum() / (~occ_np).sum()
        print("data", data_num, "After attack error rate (3 px):", diff_over_thr)
        diff_over_thr_3_list[data_num] = diff_over_thr

        thr = 1
        diff_ori = torch.abs(ori_disp[~occ] - target[~occ]).detach().cpu().numpy()
        diff_over_thr_ori = (diff_ori > thr).sum() / (~occ_np).sum()
        print("data", data_num, "Original error rate (1 px):", diff_over_thr_ori)
        diff_over_thr_1_ori_list[data_num] = diff_over_thr_ori

        diff = torch.abs(attack_disp[~occ] - target[~occ]).detach().cpu().numpy()
        diff_over_thr = (diff > thr).sum() / (~occ_np).sum()
        print("data", data_num, "After attack error rate (1 px):", diff_over_thr)
        diff_over_thr_1_list[data_num] = diff_over_thr

        # show current averages
        current_total = data_num+1
        print("avg loss:", my_mean(after_loss_list[:current_total]))
        print("avg 1px:", my_mean(diff_over_thr_1_list[:current_total]))
        print("avg 3px:", my_mean(diff_over_thr_3_list[:current_total]))

    print("number of nans:", np.isnan(before_loss_list).sum())
    print("before_loss mean:", my_mean(before_loss_list))
    print("after_loss mean:", my_mean(after_loss_list))
    print("diff_over_thr_3_ori mean:", my_mean(diff_over_thr_3_ori_list))
    print("diff_over_thr_1_ori_list mean:", my_mean(diff_over_thr_1_ori_list))
    print("diff_over_thr_3_list mean:", my_mean(diff_over_thr_3_list))
    print("diff_over_thr_1_list mean:", my_mean(diff_over_thr_1_list))