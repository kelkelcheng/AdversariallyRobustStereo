from __future__ import print_function
import argparse
import os
import torch
import torch.nn.parallel
from torch.autograd import Variable
import numpy as np
import skimage.io
import skimage.color

# settings
parser = argparse.ArgumentParser(description='Synthetic Patch Attack')
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--whichModel', type=int, default=2, help='0 for GANet, 1 for PSMNet, 2 for this method')
parser.add_argument('--total_iter', type=int, default=30, help='iterations of PGD attack')
parser.add_argument('--crop_height', type=int, default=240, help="crop height")
parser.add_argument('--crop_width', type=int, default=384, help="crop width")
parser.add_argument('--ph_size', type=int, default=30, help='half of the patch size')
parser.add_argument('--ph_x', type=int, default=100, help='patch position x')
parser.add_argument('--ph_y', type=int, default=100, help='patch position x')
parser.add_argument('--gt', type=int, default=20, help='patch ground-truth disparity level')
parser.add_argument('--a', type=float, default=0.1, help='attack step size')
parser.add_argument('--backbone', type=bool, default=False, help='if the backbone is used')
parser.add_argument('--dataset', type=int, default=1, help='which parameters to used: 1. sceneflow, 3. kitti 2015')
parser.add_argument('--test_patch_shift', type=bool, default=False, help='test on shifting the generated patch from disp 10-180')
parser.add_argument('--adv_train', type=bool, default=False, help='use adv-trained params')
opt = parser.parse_args()

if not torch.cuda.is_available():
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

model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def inverse_normalize(img, mean_rgb, std_rgb):
    height, width, _ = img.shape
    temp_data = np.zeros([height, width, 3], 'float32')
    temp_data[:, :, 0] = img[:, :, 0] * std_rgb[0] + mean_rgb[0]
    temp_data[:, :, 1] = img[:, :, 1] * std_rgb[1] + mean_rgb[1]
    temp_data[:, :, 2] = img[:, :, 2] * std_rgb[2] + mean_rgb[2]
    # return (temp_data * 255).astype(np.uint16)
    return temp_data


def projected_gradient_descent(model, x1, x2, y,  num_steps, step_size, step_norm, eps, eps_norm,
                               rgb_min_l, rgb_max_l, rgb_min_r, rgb_max_r):
    """Performs the projected gradient descent attack on a batch of images."""
    ph_size = opt.ph_size#30#12
    ph_x = opt.ph_x#150
    ph_y = opt.ph_y#150
    y_a, y_b, x_a, x_b = ph_y - ph_size, ph_y + ph_size + 1, ph_x - ph_size, ph_x + ph_size + 1

    # initialization
    batch_size, channels, im_h, im_w = x1.detach().cpu().numpy().shape
    noise = torch.zeros([batch_size, channels, im_h, im_w], requires_grad=True, device='cuda')
    noise_patch = torch.zeros([batch_size, channels, ph_size*2+1, ph_size*2+1], requires_grad=True, device='cuda')
    zero_plane = torch.zeros([batch_size, channels, im_h, im_w], requires_grad=False, device='cuda')
    zero_patch = torch.zeros([batch_size, channels, ph_size*2+1, ph_size*2+1], requires_grad=False, device='cuda')
    assert (channels == 3)
    num_channels = noise.shape[1]
    _x_adv = noise.clone().detach().requires_grad_(True)

    for i in range(num_steps):
        _x_adv = noise.clone().detach().requires_grad_(False)
        _x_adv_patch = noise_patch.clone().detach().requires_grad_(True)
        _x_adv[:, :, y_a:y_b, x_a:x_b] = _x_adv_patch

        # attack the right image
        input2_noise = x2 + _x_adv

        # apply the same adversarial patch to the left image
        noise_extract = zero_plane.clone()
        noise_extract[:, :, y_a:y_b, x_a+opt.gt:x_b+opt.gt] = _x_adv_patch
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
        elif opt.whichModel == 1 and opt.dataset==1:
            prediction = model(input_extract, input2_noise) * opt.psm_constant
        else:
            prediction = model(input_extract, input2_noise)

        # only compute loss of the patch area
        loss = torch.mean(torch.abs(prediction[0, y_a:y_b, x_a + opt.gt:x_b + opt.gt] - y[0, y_a:y_b, x_a + opt.gt:x_b + opt.gt]))
        print("iter", i, "loss:", loss.item())

        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                # only consider the patch
                gradients = _x_adv_patch.grad.sign() * step_size

            # update the patch perturbation
            noise_patch += gradients
            noise[0, :, y_a:y_b, x_a:x_b] = noise_patch

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            noise = torch.max(torch.min(noise, zero_plane + eps), zero_plane - eps)
            noise_patch = torch.max(torch.min(noise_patch, zero_patch + eps), zero_patch - eps)

    # do it again for the last iteration
    _x_adv = noise.clone().detach().requires_grad_(False)
    _x_adv_patch = noise_patch.clone().detach().requires_grad_(False)
    _x_adv[:, :, y_a:y_b, x_a:x_b] = _x_adv_patch
    input2_noise = x2 + _x_adv

    noise_extract = zero_plane.clone()
    noise_extract[:, :, y_a:y_b, x_a + opt.gt:x_b + opt.gt] = noise_patch
    input_extract = x1 + noise_extract

    # clamp
    input_extract = input_extract.reshape(im_h, im_w, channels)
    input2_noise = input2_noise.reshape(im_h, im_w, channels)

    input_extract = torch.max(torch.min(input_extract, rgb_max_l), rgb_min_l)
    input2_noise = torch.max(torch.min(input2_noise, rgb_max_r), rgb_min_r)

    input_extract = input_extract.reshape(batch_size, channels, im_h, im_w)
    input2_noise = input2_noise.reshape(batch_size, channels, im_h, im_w)

    prediction = model(input_extract, input2_noise)
    if opt.whichModel == 1 and opt.dataset == 1:
        prediction = prediction * opt.psm_constant

    # show after loss within the patch
    loss = torch.mean(torch.abs(prediction[0, y_a:y_b, x_a+opt.gt:x_b+opt.gt] - y[0, y_a:y_b, x_a+opt.gt:x_b+opt.gt])).detach()
    print("after loss inside the patch:", loss.item())

    return input_extract.detach(), input2_noise.detach()


def my_mean(temp):
    return temp[~np.isnan(temp)].mean()


if __name__ == '__main__':

    model.eval()
    # initialize inputs
    input1_np = np.ones([1, 3, opt.crop_height, opt.crop_width], 'float32')
    input2_np = np.ones_like(input1_np)
    target_np = np.zeros([1, 1, opt.crop_height, opt.crop_width], 'float32')
    occ_np = np.ones_like(target_np).astype(bool)

    ph_size = opt.ph_size
    ph_x = opt.ph_x
    ph_y = opt.ph_y
    gt = opt.gt

    # set coordinates
    y_a, y_b, x_a, x_b = ph_y - ph_size, ph_y + ph_size + 1, ph_x - ph_size, ph_x + ph_size + 1

    # intialize the adversarial patch
    np.random.seed(0)
    noise_patch = np.random.randn(1, 3, ph_size * 2 + 1, ph_size * 2 + 1)
    input1_np[:, :, y_a:y_b, x_a + gt:x_b + gt] = noise_patch
    input2_np[:, :, y_a:y_b, x_a :x_b] = noise_patch
    occ_np[:, :, y_a:y_b, x_a + gt:x_b + gt] = False
    target_np[:, :, y_a:y_b, x_a + gt:x_b + gt] = gt

    # set 0 and 1 as boundary values
    mean_left = mean_right = np.array([0.485, 0.456, 0.406])
    std_left = std_right = np.array([0.229, 0.224, 0.225])

    # rgb_min_l = rgb_min_r = np.array([0.0, 0.0, 0.0])
    # rgb_max_l = rgb_max_r = np.array([1.0, 1.0, 1.0])

    rgb_min_l = rgb_min_r = -mean_left / std_left
    rgb_max_l = rgb_max_r = (1.0 - mean_left) / std_left

    rgb_min_l = torch.tensor(rgb_min_l).cuda().float()
    rgb_min_r = torch.tensor(rgb_min_r).cuda().float()
    rgb_max_l = torch.tensor(rgb_max_l).cuda().float()
    rgb_max_r = torch.tensor(rgb_max_r).cuda().float()

    # from np to torch
    input1 = Variable(torch.from_numpy(input1_np), requires_grad=False)
    input2 = Variable(torch.from_numpy(input2_np), requires_grad=True)
    target = Variable(torch.from_numpy(target_np), requires_grad=False)
    occ = Variable(torch.from_numpy(occ_np), requires_grad=False)
    occ = torch.squeeze(occ, 1)

    # to gpu
    input1 = input1.cuda()
    input2 = input2.cuda()
    target = target.cuda()
    occ = occ.cuda()

    target = torch.squeeze(target, 1)

    # the error before attacks
    if opt.whichModel == 1 and opt.dataset==1:
        ori_disp = model(input1, input2) * opt.psm_constant
    else:
        ori_disp = model(input1, input2)
    ori_disp[occ] = 0
    print((~occ).sum())

    before_loss = torch.mean(torch.abs(ori_disp[~occ] - target[~occ])).detach()
    print("before_loss", before_loss.item())
    ori_disp = ori_disp.detach()

    # attack
    x1, x2 = projected_gradient_descent(model, input1, input2, target,
                                        num_steps=opt.total_iter, step_size=opt.a,
                                        eps=1000.0, eps_norm='inf',
                                        step_norm='inf',
                                        rgb_min_l=rgb_min_l, rgb_max_l=rgb_max_l,
                                        rgb_min_r=rgb_min_r, rgb_max_r=rgb_max_r)

    # the error after attacks
    if opt.whichModel == 1 and opt.dataset==1:
        attack_disp = model(x1, x2).detach() * opt.psm_constant
    else:
        attack_disp = model(x1, x2).detach()

    after_loss = torch.mean(torch.abs(attack_disp[~occ] - target[~occ])).detach()
    print("after_loss", after_loss.item())

    temp_loss = torch.mean(torch.abs(attack_disp[:, y_a:y_b, x_a + gt:x_b + gt] - torch.ones_like(
        attack_disp[:, y_a:y_b, x_a + gt:x_b + gt]) * gt)).detach()
    print("disp", opt.gt, "after_loss", temp_loss.item())

    # test the same patch on different shift
    if opt.test_patch_shift:
        patch_attack_list = np.zeros(opt.max_disp) - 1
        for i in range(10, 181, 1):
            x1_new = torch.ones_like(x1)
            x2_new = torch.ones_like(x2)
            x2_new[:, :, y_a:y_b, x_a:x_b] = x2[:, :, y_a:y_b, x_a:x_b].clone()
            x1_new[:, :, y_a:y_b, x_a + i:x_b + i] = x2_new[:, :, y_a:y_b, x_a:x_b]

            temp_disp = model(x1_new, x2_new).detach()
            if opt.whichModel == 1:
                temp_disp *= opt.psm_constant
            temp_loss = torch.mean(torch.abs(temp_disp[:, y_a:y_b, x_a + i:x_b + i] - torch.ones_like(temp_disp[:, y_a:y_b, x_a + i:x_b + i]) * i)).detach()
            print("disp", i, "after_loss", temp_loss.item())
            patch_attack_list[i - 1] = temp_loss.item()
        np.savetxt("patch_attack_list_" + str(opt.whichModel) + ".csv", patch_attack_list, delimiter=",")

    # store attacked images
    # mean_left = mean_right = np.array([0.485, 0.456, 0.406])
    # std_left = std_right = np.array([0.229, 0.224, 0.225])

    # temp = x1.detach().cpu().numpy().squeeze()
    # temp = np.moveaxis(temp, 0, -1)
    # temp = inverse_normalize(temp, mean_left, std_left)
    # print(temp.shape)

    # post_fname = 'model_' + str(opt.whichModel) + '.png'
    # savename = './' + 'patch_attack_left_' + post_fname
    # skimage.io.imsave(savename, (temp * 255).astype('uint8'))

    # temp = x2.detach().cpu().numpy().squeeze()
    # temp = np.moveaxis(temp, 0, -1)#.astype('uint16')
    # temp = inverse_normalize(temp, mean_right, std_right)
    # print(temp.shape)
    # savename = './' + 'patch_attack_right_' + post_fname
    # skimage.io.imsave(savename, (temp * 255).astype('uint8'))
