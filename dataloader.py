import torch.utils.data as data
from PIL import Image
import numpy as np
import random
from struct import unpack
import re
import sys

def readPFM(file): 
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
        #        quit()
    return img, height, width

def train_transform(temp_data, crop_height, crop_width, left_right=False, shift=0):
    _, h, w = np.shape(temp_data)

    if h > crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, h + shift, crop_width + shift], 'float32')
        temp_data[6:7, :, :] = 1000
        temp_data[:, h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = temp
        _, h, w = np.shape(temp_data)

    start_x = random.randint(0, w - crop_width)
    start_y = random.randint(0, h - crop_height)
    temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]
    return left, right, target

def test_transform(temp_data, crop_height, crop_width, left_right=False):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8,crop_height,crop_width], 'float32')
        temp_data[6: 7, :, :] = 1000
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w-crop_width)/2)
        start_y = int((h-crop_height)/2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
   
    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]

    return left, right, target

def train_transform_occ(temp_data, crop_height, crop_width, left_right=False, shift=0):
    _, h, w = np.shape(temp_data)

    if h > crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, h+shift, crop_width + shift], 'float32')
        temp_data[6:7,:,:] = 1000
        temp_data[:, h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = temp
        _, h, w = np.shape(temp_data)

    if shift > 0:
        start_x = random.randint(0, w - crop_width)
        shift_x = random.randint(-shift, shift)
        if shift_x + start_x < 0 or shift_x + start_x + crop_width > w:
            shift_x = 0
        start_y = random.randint(0, h - crop_height)
        left = temp_data[0: 3, start_y: start_y + crop_height, start_x + shift_x: start_x + shift_x + crop_width]
        right = temp_data[3: 6, start_y: start_y + crop_height, start_x: start_x + crop_width]
        target = temp_data[6: 7, start_y: start_y + crop_height, start_x + shift_x : start_x+shift_x + crop_width]
        target = target - shift_x
        return left, right, target

    start_x = random.randint(0, w - crop_width)
    start_y = random.randint(0, h - crop_height)
    temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]
    occ = temp_data[7: 8, :, :]
    occ = occ.astype(bool)
    return left, right, target, occ

def test_transform_occ(temp_data, crop_height, crop_width, left_right=False):
    _, h, w = np.shape(temp_data)

    start_x = int((w-crop_width)/2)
    start_y = int((h-crop_height)/2)
    temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
   
    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]
    occ = temp_data[7: 8, :, :]
    occ = occ.astype(bool)
    return left, right, target, occ
    
def test_transform_full(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    start_x = int((w-crop_width)/2)
    start_y = int((h-crop_height)/2)
    temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    
    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]
    occ = temp_data[7: 8, :, :]
    occ = occ.astype(bool)
    return left, right, target, occ

def load_data(data_path, current_file, method, transform=None):
    A = current_file
    filename = data_path + 'frames_finalpass/' + A[0: len(A) - 1]
    left  =Image.open(filename)
    filename = data_path + 'frames_finalpass/' + A[0: len(A) - 14] + 'right/' + A[len(A) - 9:len(A) - 1]
    right = Image.open(filename)
    filename = data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
    disp_left, height, width = readPFM(filename)

    left = np.asarray(left).astype(float)
    right = np.asarray(right).astype(float)

    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')

    if method == 0:
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

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left
    # temp_data[7, :, :] = disp_right
    return temp_data

def load_subset_data(data_path, current_file, method):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    A = current_file
    # filename = data_path + 'frames_finalpass/' + A[0: len(A) - 1]
    filename = data_path + 'image_clean/' + A[0: len(A) - 1]
    left  =Image.open(filename)
    # filename = data_path + 'frames_finalpass/' + A[0: len(A) - 14] + 'right/' + A[len(A) - 9:len(A) - 1]
    filename = data_path + 'image_clean/' + 'right/' + A[5:len(A) - 1]
    right = Image.open(filename)
    filename = data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
    disp_left, height, width = readPFM(filename)
    disp_left = -disp_left
    
    filename = data_path + 'disparity_occlusions/' + A[0: len(A) - 1]
    occ_left  = Image.open(filename)

    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left) / 255.
    right = np.asarray(right) / 255.
    occ_left  = np.asarray(occ_left).astype(float)

    temp_data[0, :, :] = (left[:, :, 0] - imagenet_mean[0]) / imagenet_std[0]
    temp_data[1, :, :] = (left[:, :, 1] - imagenet_mean[1]) / imagenet_std[1]
    temp_data[2, :, :] = (left[:, :, 2] - imagenet_mean[2]) / imagenet_std[2]
	
    temp_data[3, :, :] = (right[:, :, 0] - imagenet_mean[0]) / imagenet_std[0]
    temp_data[4, :, :] = (right[:, :, 1] - imagenet_mean[1]) / imagenet_std[1]
    temp_data[5, :, :] = (right[:, :, 2] - imagenet_mean[2]) / imagenet_std[2]
    
    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left
    temp_data[7, :, :] = occ_left
    return temp_data

def load_kitti_data(file_path, current_file):
    """ load current file from the list"""
    filename = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
    left = Image.open(filename)
    filename = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
    right = Image.open(filename)
    filename = file_path + 'disp_occ/' + current_file[0: len(current_file) - 1]

    disp_left = Image.open(filename)
    temp = np.asarray(disp_left)
    size = np.shape(left)

    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    disp_left = np.asarray(disp_left)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]

    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]

    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.1] = width * 2 * 256
    temp_data[6, :, :] = temp / 256.

    return temp_data

def load_kitti2015_data(file_path, current_file, method):
    """ load current file from the list"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    filename = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
    left = Image.open(filename)
    filename = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
    right = Image.open(filename)
    filename = file_path + 'disp_occ_0/' + current_file[0: len(current_file) - 1]

    disp_left = Image.open(filename)
    temp = np.asarray(disp_left)
    size = np.shape(left)

    height = size[0]
    width = size[1]

    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left).astype(float) # / 255.0
    right = np.asarray(right).astype(float) # / 255.0

    disp_left = np.asarray(disp_left)

    if method == 0:
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

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.01] = width * 2 * 256
    temp_data[6, :, :] = temp / 256.
    
    return temp_data


def load_data_md(file_path, current_file, method=1):
    """ load current file from the list"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    imgl = file_path + current_file[0: len(current_file) - 1]
    gt_l = imgl.replace('im0.png', 'disp0GT.pfm')
    imgr = imgl.replace('im0.png', 'im1.png')

    left = Image.open(imgl)
    right = Image.open(imgr)

    disp_left, height, width = readPFM(gt_l)

    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left).astype(float)
    right = np.asarray(right).astype(float)
    disp_left = np.asarray(disp_left)

    if method == 0:
        mean_left = np.array([np.mean(left[:, :, 0]), np.mean(left[:, :, 1]), np.mean(left[:, :, 2])])
        std_left = np.array([np.std(left[:, :, 0]), np.std(left[:, :, 1]), np.std(left[:, :, 2])])

        mean_right = np.array([np.mean(right[:, :, 0]), np.mean(right[:, :, 1]), np.mean(right[:, :, 2])])
        std_right = np.array([np.std(right[:, :, 0]), np.std(right[:, :, 1]), np.std(right[:, :, 2])])
    else:
        mean_left = mean_right = np.array(imagenet_mean)
        std_left = std_right = np.array(imagenet_std)
        left /= 255.
        right /= 255.

    temp_data[0:3, :, :] = np.moveaxis((left[:,:,:3] - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right[:,:,:3] - mean_right) / std_right, -1, 0)

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.01] = width * 2 # * 256
    temp_data[6, :, :] = temp  # / 256.

    return temp_data

class DatasetFromList(data.Dataset): 
    def __init__(self, data_path, file_list, crop_size=[256, 256], training=True, left_right=False, dataset=0, shift=0, method=0):
        super(DatasetFromList, self).__init__()
        f = open(file_list, 'r')
        self.data_path = data_path
        self.file_list = f.readlines()
        self.training = training
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.left_right = left_right
        self.dataset = dataset
        self.shift = shift
        self.method = method

    def __getitem__(self, index):
        if self.dataset==0:
            temp_data = load_subset_data(self.data_path, self.file_list[index], self.method)
            if self.training:
                input1, input2, target, occ = train_transform_occ(temp_data, self.crop_height, self.crop_width, self.left_right, self.shift)
            else:
                input1, input2, target, occ = test_transform_occ(temp_data, self.crop_height, self.crop_width)
            return input1, input2, target, occ

        elif self.dataset==1:

            temp_data = load_data(self.data_path, self.file_list[index], self.method)

            if self.training:
                input1, input2, target = train_transform(temp_data, self.crop_height, self.crop_width, self.left_right, self.shift)
            else:
                input1, input2, target = test_transform(temp_data, self.crop_height, self.crop_width)
            return input1, input2, target

        elif self.dataset==2: #load kitti2012 dataset
            temp_data = load_kitti_data(self.data_path, self.file_list[index])
            if self.training:
                input1, input2, target = train_transform(temp_data, self.crop_height, self.crop_width, self.left_right, self.shift)
                return input1, input2, target
            else:
                input1, input2, target = test_transform(temp_data, self.crop_height, self.crop_width)
                return input1, input2, target
        elif self.dataset==3: #load kitti2015 dataset
            temp_data = load_kitti2015_data(self.data_path, self.file_list[index], self.method)
            if self.training:
                input1, input2, target = train_transform(temp_data, self.crop_height, self.crop_width, self.left_right, self.shift)
                return input1, input2, target
            else:
                input1, input2, target = test_transform(temp_data, self.crop_height, self.crop_width)
                return input1, input2, target
        else: #load scene flow dataset
            temp_data = load_data(self.data_path, self.file_list[index])
                
        if self.training:
            input1, input2, target = train_transform(temp_data, self.crop_height, self.crop_width, self.left_right, self.shift)
            return input1, input2, target
        else:
            input1, input2, target = test_transform(temp_data, self.crop_height, self.crop_width)
            return input1, input2, target

    def __len__(self):
        return len(self.file_list)
