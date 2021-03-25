from __future__ import print_function
import argparse
import sys
import os
import torch
import torch.nn.parallel
import torch.distributed as dist
from apex.parallel import DistributedDataParallel
from apex import amp
from apex.parallel import convert_syncbn_model
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader import DatasetFromList

# Training settings
parser = argparse.ArgumentParser(description='Train Stereo Matching Models')
parser.add_argument('--crop_height', type=int, default=240, help="crop height")
parser.add_argument('--crop_width', type=int, default=576, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--start_ep', type=int, default=1, help="start epoch")
parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--dataset', type=int, default=1, help='0: sceneflow_subset, 1: sceneflow, 3: kitti 2015')
parser.add_argument('--whichModel', type=int, default=1, help='0: GANet, 1: ours')
parser.add_argument('--data_path', type=str, default='../../data/', help="data root")
parser.add_argument('--save_path', type=str, default='./checkpoint/MCTNet/', help="location to save models")
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--backbone', type=bool, default=False, help='if the backbone is used')
opt = parser.parse_args()

if opt.dataset == 1:
    opt.train_data_path = opt.data_path + 'FlyingThings3D/'
    opt.training_list = './lists/sceneflow_train.list'
    opt.val_list = './lists/sceneflow_test_select.list'
    opt.test_height = 576
    opt.test_width = 960
elif opt.dataset == 3:
    opt.train_data_path = opt.data_path + 'KITTI2015/training/'
    opt.training_list = './lists/kitti2015_train_new.list'
    opt.val_list = './lists/kitti2015_val_new.list'
    opt.test_height = 384
    opt.test_width = 1248

print(opt)

dist.init_process_group(backend='nccl')
torch.cuda.set_device(opt.local_rank)

if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

train_set = DatasetFromList(opt.train_data_path, opt.training_list, [opt.crop_height, opt.crop_width], training=True, dataset=opt.dataset, method=opt.whichModel)
val_set = DatasetFromList(opt.train_data_path, opt.val_list, [opt.test_height, opt.test_width], training=False, dataset=opt.dataset, method=opt.whichModel)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False,
                                  drop_last=True, sampler=train_sampler, pin_memory=True)
testing_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

# select model
print('===> Building model')
if opt.whichModel == 0:
    from models.GANet_deep import GANet
    model = GANet(opt.max_disp)
else:
    if opt.backbone:
        from models.MCTNet_backbone import Model
    else:
        from models.MCTNet import Model
    model = Model(opt.max_disp)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of parameters:", pytorch_total_params)

# use apex for acceleration
model = convert_syncbn_model(model).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
model = DistributedDataParallel(model)

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if opt.whichModel==1:
            amp.load_state_dict(checkpoint['amp'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

def train(epoch):
    train_sampler.set_epoch(epoch)
    model.train()

    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1],requires_grad=True), Variable(batch[2], requires_grad=False)

        # to gpu
        input1 = input1.cuda()
        input2 = input2.cuda()
        target = target.cuda()

        target = torch.squeeze(target, 1)

        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid > 0:
            optimizer.zero_grad()

            # get disparity and compute errors
            disp1, disp2, disp3 = model(input1, input2)

            disp1_loss = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
            disp2_loss = F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
            disp3_loss = F.smooth_l1_loss(disp3[mask], target[mask], reduction='mean')

            loss = 0.5 * disp1_loss + 0.7 * disp2_loss + disp3_loss
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if opt.local_rank == 0:
                print("===> Epoch[{}]({}/{}): Loss1: {:.4f}, Loss2: ({:.4f})".format(epoch, iteration,
                                                                                     len(training_data_loader),
                                                                                     disp1_loss.item(), disp3_loss.item()))
                sys.stdout.flush()


def val():
    epoch_error = 0
    valid_iteration = 0
    three_px_acc_all = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)

        input1 = input1.cuda()
        input2 = input2.cuda()
        target = target.cuda()
        target = torch.squeeze(target, 1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid > 0:
            with torch.no_grad():
                disp = model(input1, input2)
                abs_diff = torch.abs(disp[mask] - target[mask])
                error = torch.mean(abs_diff)

                valid_iteration += 1
                epoch_error += error.item()

                # computing 3-px error#
                abs_diff_np = abs_diff.cpu().detach().numpy()
                mask_np = mask.cpu().detach().numpy()
                three_px_acc = (abs_diff_np > 1).sum() / mask_np.sum()
                three_px_acc_all += three_px_acc

                print("===> Test({}/{}): Error: ({:.4f} {:.4f})".format(iteration, len(testing_data_loader), error.item(),
                                                                      three_px_acc))
                sys.stdout.flush()

    print("===> Test: Avg. Error: ({:.4f} {:.4f})".format(epoch_error / valid_iteration,
                                                          three_px_acc_all / valid_iteration))
    return three_px_acc_all / valid_iteration

def save_checkpoint(save_path, epoch, state, is_best):
    filename = save_path + "_epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "_epoch_{}_best.pth".format(epoch))
    print("Checkpoint saved to {}".format(filename))

if __name__ == '__main__':
    error = 100
    start_epoch = opt.start_ep
    for epoch in range(start_epoch, start_epoch + opt.nEpochs):
        train(epoch)

        if opt.local_rank == 0:
            is_best = False

            # for SceneFlow
            if opt.dataset == 1:
                if epoch>=0:
                    save_checkpoint(opt.save_path, epoch,{
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'amp': amp.state_dict()
                    }, is_best)
            # for KITTI2015
            else:
                # use validation error to keep track
                loss = val()
                if loss < error:
                    error = loss
                    is_best = True

                if epoch % 100 == 0 and epoch >= 300:
                    save_checkpoint(opt.save_path, epoch, {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }, is_best)
                if epoch >= 100 and is_best: # select the best one
                    save_checkpoint(opt.save_path, epoch, {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }, is_best)