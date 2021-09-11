from __future__ import print_function
import argparse
import os
import torch
import torch.nn.parallel
from torch.autograd import Variable
from dataloader import readPFM
import numpy as np
from PIL import Image
import skimage.io
import skimage.color

# test settings
parser = argparse.ArgumentParser(description='Adversarial Patch Attack')
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--a', type=float, default=0.01, help='step size of PGD attack')
parser.add_argument('--dataset', type=int, default=3, help='1: sceneflow, 2: kitti. 3: kitti 2015')
parser.add_argument('--whichModel', type=int, default=2, help='0 for GANet, 1 for PSMNet, 2 for this method')
parser.add_argument('--total_iter', type=int, default=50, help='iterations of PGD attack')
parser.add_argument('--backbone', type=bool, default=False, help='if the backbone is used')
parser.add_argument('--adv_train', type=bool, default=False, help='use adv-trained params')
parser.add_argument('--save', type=bool, default=False, help='save outputs')
parser.add_argument('--out_top_dir', type=str, default='patch_attack_results/', help='output directories')
opt = parser.parse_args()

# setup output directory
if opt.whichModel == 0:
    opt.out_dir = opt.out_top_dir + 'GANet/'
elif opt.whichModel == 1:
    opt.out_dir = opt.out_top_dir + 'PSMNet/'
elif opt.whichModel == 2:
    if opt.backbone:
        opt.out_dir = opt.out_top_dir + 'MCTNet_backbone/'
    else:
        opt.out_dir = opt.out_top_dir + 'MCTNet/'
elif opt.whichModel == 5:
    opt.out_dir = opt.out_top_dir + 'TwoBackbone/'

# setup dataset
if opt.dataset == 1:
    opt.test_data_path = '/home/kbcheng/data/FT_subset/val/'
    opt.val_list = './lists/sceneflow_subset_val.list'
    opt.crop_height = 240
    opt.crop_width = 576
elif opt.dataset == 2:
    opt.test_data_path = '/home/kbcheng/data/KITTI2012/training/'
    opt.val_list = './lists/kitti2012_train.list'
    opt.crop_height = 240
    opt.crop_width = 384
else:
    opt.test_data_path = '/home/kbcheng/data/KITTI2015/training/'
    opt.val_list = './lists/kitti2015_train.list'
    opt.crop_height = 240
    opt.crop_width = 384

# setup cuda
cuda = 1
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.cuda.manual_seed(123)
torch.manual_seed(123)

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
    if opt.adv_train:
        opt.resume = 'checkpoint/PSMNet/PSMNet_adv-3_epoch_20.pth'
    if opt.dataset == 1:
        opt.resume = 'checkpoint/PSMNet/pretrained_sceneflow.tar'
        opt.psm_constant = 1.17
    if opt.crop_height == 240:
        opt.crop_height = 256
elif opt.whichModel == 2:
    if opt.backbone:
        from models.MCTNet_backbone import Model
        opt.resume = 'checkpoint/MCTNet/MCTNet_backbone_kitti.pth'
        if opt.adv_train:
            opt.resume = 'checkpoint/MCTNet/MCTNet_backbone_adv-2_epoch_20.pth'
        if opt.dataset == 1:
            opt.resume = 'checkpoint/MCTNet/MCTNet_backbone_sf_epoch_20.pth'
    else:
        from models.MCTNet import Model
        opt.resume = 'checkpoint/MCTNet/MCTNet_kitti.pth'
        if opt.adv_train:
            opt.resume = 'checkpoint/MCTNet/MCTNet_adv-3_epoch_20.pth'
        if opt.dataset == 1:
            opt.resume = 'checkpoint/MCTNet/MCTNet_sf_epoch_20.pth'
    model = Model(opt.max_disp)
    model.training = False
elif opt.whichModel == 5:
    from models.CompMatchDS3Feat import Model
    opt.resume = 'checkpoint/CompMatchDS3Feat/kitti_epoch_413_best.pth'
    if opt.dataset == 1:
        opt.resume = 'checkpoint/CompMatchDS3Feat/_epoch_20.pth'
    if opt.adv_train:
        opt.resume = 'checkpoint/CompMatchDS3Feat/adv-3_epoch_20.pth'
    model = Model(opt.max_disp)    
print(opt)


if cuda:
    model = torch.nn.DataParallel(model).cuda()

# load model
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def inverse_normalize(img, mean_rgb, std_rgb):
    '''inverse the normalization for saving figures'''
    height, width, _ = img.shape
    temp_data = np.zeros([height, width, 3], 'float32')
    temp_data[:, :, 0] = img[:, :, 0] * std_rgb[0] + mean_rgb[0]
    temp_data[:, :, 1] = img[:, :, 1] * std_rgb[1] + mean_rgb[1]
    temp_data[:, :, 2] = img[:, :, 2] * std_rgb[2] + mean_rgb[2]
    return temp_data

def fetch_data(A, crop_height=240, crop_width=576):
    ''' self-contained data extraction '''
    # parse data name
    if opt.dataset == 1:
        filename_l = opt.test_data_path + 'frames_finalpass/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'frames_finalpass/' + 'right/' + A[5:len(A) - 1]
        filename_disp = opt.test_data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
        disp_left, height, width = readPFM(filename_disp)
        disp_left = -disp_left
        filename_occ = opt.test_data_path + 'disparity_occlusions/' + A[0: len(A) - 1]
        occ_left = Image.open(filename_occ)
        occ_left = np.asarray(occ_left)
        occ_left = occ_left | (disp_left >= opt.max_disp)
    elif opt.dataset == 2:
        filename_l = opt.test_data_path + 'colored_0/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'colored_1/' + A[0: len(A) - 1]
        filename_disp = opt.test_data_path + 'disp_occ/' + A[0: len(A) - 1]
        disp_left = np.asarray(Image.open(filename_disp)).astype(float)
    elif opt.dataset == 3:
        filename_l = opt.test_data_path + 'image_2/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'image_3/' + A[0: len(A) - 1]
        filename_disp = opt.test_data_path + 'disp_occ_0/' + A[0: len(A) - 1]
        disp_left = np.asarray(Image.open(filename_disp)).astype(float)

    # read data
    left = Image.open(filename_l)
    right = Image.open(filename_r)

    # initialize for normalization and cropping
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([9, height, width], 'float32')
    left = np.asarray(left).astype(float)
    right = np.asarray(right).astype(float)

    # for KITTI2012 and KITTI2015
    if opt.dataset != 1:
        disp_left[disp_left < 0.01] = width * 2 * 256
        disp_left = disp_left / 256.
        occ_left = (disp_left >= opt.max_disp).astype(float)

    # normalization
    scale = 1.0
    if opt.whichModel == 0:
        # for GANet and LEAStereo
        mean_left = np.array([np.mean(left[:, :, 0]), np.mean(left[:, :, 1]), np.mean(left[:, :, 2])])
        std_left = np.array([np.std(left[:, :, 0]), np.std(left[:, :, 1]), np.std(left[:, :, 2])])

        mean_right = np.array([np.mean(right[:, :, 0]), np.mean(right[:, :, 1]), np.mean(right[:, :, 2])])
        std_right = np.array([np.std(right[:, :, 0]), np.std(right[:, :, 1]), np.std(right[:, :, 2])])

        scale = 255.0
    else:
        # for PSMNet and our method
        mean_left = mean_right = np.array([0.485, 0.456, 0.406])
        std_left = std_right = np.array([0.229, 0.224, 0.225])
        left /= 255.
        right /= 255.

    # get min and max for the normalized values
    rgb_min_l = -mean_left / std_left
    rgb_max_l = (scale - mean_left) / std_left
    rgb_min_r = -mean_right / std_right
    rgb_max_r = (scale - mean_right) / std_right

    # normalization
    temp_data[0:3, :, :] = np.moveaxis((left - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right - mean_right) / std_right, -1, 0)

    temp_data[6, :, :] = width * 2
    temp_data[6, :, :] = disp_left
    temp_data[7, :, :] = occ_left
    
    # crop data
    if height <= crop_height and width <= crop_width:
        temp = temp_data.copy()
        temp_data = np.zeros([9, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - height: crop_height, crop_width - width: crop_width] = temp

        # set the filled-in areas as occluded to avoid to count as results
        temp_data[7, 0:crop_height - height, :] = 1.0
        temp_data[7, :, 0:crop_width - width] = 1.0
        temp_data[8, 0:crop_height - height, :] = 1.0
        temp_data[8, :, 0:crop_width - width] = 1.0
    else:
        # crop the center
        start_x = int((width - crop_width) / 2)
        start_y = int((height - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]


    input1_np = np.expand_dims(temp_data[0:3], axis=0)
    input2_np = np.expand_dims(temp_data[3:6], axis=0)
    target_np = np.expand_dims(temp_data[6:7], axis=0)
    occ_np = np.expand_dims(temp_data[7:8], axis=0)
    occ_np = occ_np.astype(bool)

    info = {'rgb_min_l': rgb_min_l, 'rgb_min_r': rgb_min_r,
            'rgb_max_l': rgb_max_l, 'rgb_max_r': rgb_max_r,
            'mean_right': mean_right, 'mean_left': mean_left,
            'std_right': std_right, 'std_left': std_left}

    return input1_np, input2_np, target_np, occ_np, info


def patch_pgd(patch_info, model, x1, x2, y, occ, mask, occ_mask, num_steps, step_size, step_norm, eps, eps_norm,
                               rgb_min_l, rgb_max_l, rgb_min_r, rgb_max_r):
    """Performs the projected gradient descent attack on a patch."""
    ph_size = patch_info[0]
    ph_x = patch_info[1]
    ph_y = patch_info[2]

    # addjust patch location for PSMNet as its cropped height is 256
    if opt.whichModel == 1:
        ph_y += 8 
    
    # get x,y coordinates for the crop
    y_a, y_b, x_a, x_b = ph_y - ph_size, ph_y + ph_size + 1, ph_x - ph_size, ph_x + ph_size + 1

    # initialize
    batch_size, channels, im_h, im_w = x1.detach().cpu().numpy().shape
    noise = torch.zeros([batch_size, channels, im_h, im_w], requires_grad=True, device='cuda')
    noise_patch = torch.zeros([batch_size, channels, ph_size*2+1, ph_size*2+1], requires_grad=True, device='cuda')
    zero_plane = torch.zeros([batch_size, channels, im_h, im_w], requires_grad=False, device='cuda')
    zero_patch = torch.zeros([batch_size, channels, ph_size*2+1, ph_size*2+1], requires_grad=False, device='cuda')
    assert (channels == 3)
    num_channels = noise.shape[1]
    _x_adv = noise.clone().detach().requires_grad_(True)

    for i in range(num_steps):
        # fetch adversarial patch
        _x_adv = noise.clone().detach().requires_grad_(False)
        _x_adv_patch = noise_patch.clone().detach().requires_grad_(True)
        _x_adv[:, :, y_a:y_b, x_a:x_b] = _x_adv_patch

        # apply perturbation on the right image
        input2_noise = x2 + _x_adv

        # apply perturbation on the left image
        noise_extract = torch.gather(_x_adv, channels, mask)
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
            prediction = model(input_extract, input2_noise, attack=True)
        else:
            prediction = model(input_extract, input2_noise)

        # print loss
        noc_crop = (~occ)[0, y_a:y_b, x_a:x_b]
        loss = torch.mean(torch.abs(prediction[~occ] - y[~occ]))
        print("iter", i, "loss:", loss.item())
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                gradients = _x_adv_patch.grad.sign() * step_size

            noise_patch += gradients
            noise[0, :, y_a:y_b, x_a:x_b] = noise_patch

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # clamp values outside the norm
            noise = torch.max(torch.min(noise, zero_plane + eps), zero_plane - eps)
            noise_patch = torch.max(torch.min(noise_patch, zero_patch + eps), zero_patch - eps)

    _x_adv = noise.clone().detach().requires_grad_(False)
    _x_adv_patch = noise_patch.clone().detach().requires_grad_(True)
    _x_adv[:, :, y_a:y_b, x_a:x_b] = _x_adv_patch

    input2_noise = x2 + _x_adv

    noise_extract = torch.gather(noise, channels, mask)
    input_extract = x1 + noise_extract

    # clamp out-of-range values
    input_extract = input_extract.reshape(im_h, im_w, channels)
    input2_noise = input2_noise.reshape(im_h, im_w, channels)

    input_extract = torch.max(torch.min(input_extract, rgb_max_l), rgb_min_l)
    input2_noise = torch.max(torch.min(input2_noise, rgb_max_r), rgb_min_r)

    input_extract = input_extract.reshape(batch_size, channels, im_h, im_w)
    input2_noise = input2_noise.reshape(batch_size, channels, im_h, im_w)

    prediction = model(input_extract, input2_noise)
    noc_crop = (~occ)[0, y_a:y_b, x_a:x_b]
    # loss = torch.mean(torch.abs(prediction[~occ] - y[~occ]))
    loss = torch.mean(torch.abs(prediction[0, y_a:y_b, x_a:x_b][noc_crop] - y[0, y_a:y_b, x_a:x_b][noc_crop])).detach()
    print("after loss within the patch:", loss.item())

    return input_extract.detach(), input2_noise.detach()


def my_mean(temp):
    return temp[~np.isnan(temp)].mean()


if __name__ == '__main__':

    # input the picked 10 data
    file_list = [['000167_10.png ', (20, 100, 195)]]
    file_list += [['000004_10.png ', (20, 190, 110)]]
    file_list += [['000130_10.png ', (20, 170, 180)]]
    file_list += [['000133_10.png ', (20, 50, 200)]]
    file_list += [['000141_10.png ', (15, 175, 165)]]
    file_list += [['000144_10.png ', (15, 170, 160)]]
    file_list += [['000150_10.png ', (20, 180, 150)]]
    file_list += [['000165_10.png ', (15, 125, 120)]]
    file_list += [['000166_10.png ', (20, 120, 160)]]
    file_list += [['000199_10.png ', (20, 160, 145)]]

    # initialize
    data_total = len(file_list)
    before_loss_list = np.zeros(data_total)
    after_loss_list = before_loss_list.copy()
    diff_over_thr_3_ori_list = before_loss_list.copy()
    diff_over_thr_1_ori_list = before_loss_list.copy()
    diff_over_thr_3_list = before_loss_list.copy()
    diff_over_thr_1_list = before_loss_list.copy()

    model.eval()
    for data_num in range(data_total):
        A = file_list[data_num][0]
        patch_info = file_list[data_num][1]

        input1_np, input2_np, target_np, occ_np, info = fetch_data(A, opt.crop_height, opt.crop_width)

        # fetch min and max normalized color values
        rgb_min_l, rgb_min_r = info['rgb_min_l'], info['rgb_min_r']
        rgb_max_l, rgb_max_r = info['rgb_max_l'], info['rgb_max_r']
        rgb_min_l = torch.tensor(rgb_min_l).cuda().float()
        rgb_min_r = torch.tensor(rgb_min_r).cuda().float()
        rgb_max_l = torch.tensor(rgb_max_l).cuda().float()
        rgb_max_r = torch.tensor(rgb_max_r).cuda().float()

        # from np to torch
        input1 = Variable(torch.from_numpy(input1_np), requires_grad=False)
        input2 = Variable(torch.from_numpy(input2_np), requires_grad=True)
        target = Variable(torch.from_numpy(target_np), requires_grad=False)
        occ = Variable(torch.from_numpy(occ_np), requires_grad=False)
        # occ = torch.squeeze(occ, 1)

        # mask is the indices for fetching noise
        mask = torch.linspace(0, opt.crop_width - 1, steps=opt.crop_width, requires_grad=True)
        mask = mask.repeat(target.size()[0], target.size()[1], target.size()[2], 1)
        mask = mask - target
        mask = mask.round().long()

        # set out-of-crop pixels to be occluded
        occ = occ | (mask < 0)
        occ = torch.squeeze(occ, 1)

        mask = torch.clamp(mask, 0, opt.crop_width - 1)
        mask = mask.repeat(1, 3, 1, 1)

        # occ_mask is occ repeated for RGB channels
        occ_mask = occ.repeat(1, 3, 1, 1)

        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
            occ = occ.cuda()
            mask = mask.cuda()
            occ_mask = occ_mask.cuda()

        target = torch.squeeze(target, 1)

        # after attack error
        ori_disp = model(input1, input2)
        before_loss = torch.mean(torch.abs(ori_disp[~occ] - target[~occ])).detach()
        print("data", data_num, "before_loss", before_loss.item())
        before_loss_list[data_num] = before_loss.item()

        ori_disp = ori_disp.detach()

        # attack
        x1, x2 = patch_pgd(patch_info, model, input1, input2, target, occ, mask, occ_mask,
                                            num_steps=opt.total_iter, step_size=opt.a,
                                            eps=1000.0, eps_norm='inf',
                                            step_norm='inf',
                                            rgb_min_l=rgb_min_l, rgb_max_l=rgb_max_l,
                                            rgb_min_r=rgb_min_r, rgb_max_r=rgb_max_r)

        # after attack error
        attack_disp = model(x1, x2).detach()
        after_loss = torch.mean(torch.abs(attack_disp[~occ] - target[~occ])).detach()
        print("data", data_num, "after_loss", after_loss.item())
        after_loss_list[data_num] = after_loss.item()

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

        # save outputs
        if opt.save:
            if opt.whichModel==0:
                mean_left = info['mean_left']
                mean_right = info['mean_right']
                std_left = info['std_left']
                std_right = info['std_right']
            else:
                mean_left = mean_right = np.array([0.485, 0.456, 0.406])
                std_left = std_right = np.array([0.229, 0.224, 0.225])

            # save the perturbed left image
            temp = x1.detach().cpu().numpy().squeeze()
            temp = np.moveaxis(temp, 0, -1)
            temp = inverse_normalize(temp, mean_left, std_left)

            model_str = 'm' + str(opt.whichModel) + '_'
            if opt.adv_train:
                model_str += 'adv_'

            savename = opt.out_dir + model_str + 'attack_left_' + A[0: len(A) - 1] 
            if opt.whichModel==0:
                skimage.io.imsave(savename, temp.astype('uint8'))
            else:
                skimage.io.imsave(savename, (temp * 255).astype('uint8'))

            # save the attaperturbedcked right image
            temp = x2.detach().cpu().numpy().squeeze()
            temp = np.moveaxis(temp, 0, -1)
            temp = inverse_normalize(temp, mean_right, std_right)
            print(temp.shape)
            savename = opt.out_dir + model_str + 'attack_right_' + A[0: len(A) - 1]
            if opt.whichModel==0:
                skimage.io.imsave(savename, temp.astype('uint8'))
            else:
                skimage.io.imsave(savename, (temp * 255).astype('uint8'))
            
            # save the after attack disparity
            temp = attack_disp.detach().cpu().numpy().squeeze()
            print(temp.shape)
            savename = opt.out_dir + model_str + 'attack_disp_' + A[0: len(A) - 1]
            skimage.io.imsave(savename, (temp * 256).astype('uint16'))
            
            # save the before attack disparity
            temp = ori_disp.detach().cpu().numpy().squeeze()
            print(temp.shape)
            savename = opt.out_dir + model_str + 'ori_disp_' + A[0: len(A) - 1]
            skimage.io.imsave(savename, (temp * 256).astype('uint16'))

    # print errors
    print("number of nans:", np.isnan(before_loss_list).sum())
    print("before_loss mean:", my_mean(before_loss_list))
    print("after_loss mean:", my_mean(after_loss_list))
    print("diff_over_thr_3_ori mean:", my_mean(diff_over_thr_3_ori_list))
    print("diff_over_thr_1_ori_list mean:", my_mean(diff_over_thr_1_ori_list))
    print("diff_over_thr_3_list mean:", my_mean(diff_over_thr_3_list))
    print("diff_over_thr_1_list mean:", my_mean(diff_over_thr_1_list))
