#from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
import h5py
from torch.utils.data import DataLoader
from skimage import metrics

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/%06d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('data'))
            data = torch.FloatTensor(data.astype(float))
        return data

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    dataset_dir = args.testset_dir
    data_list = os.listdir(dataset_dir)
    test_Loaders = []
    length_of_tests = 0

    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name)
        length_of_tests += len(test_Dataset)
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL'):
        super(TestSetDataLoader, self).__init__()
        self.angin = args.angin
        self.dataset_dir = args.testset_dir + data_name
        self.file_list = []
        tmp_list = os.listdir(self.dataset_dir)
        for index, _ in enumerate(tmp_list):
            tmp_list[index] = tmp_list[index]
        self.file_list.extend(tmp_list)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = self.dataset_dir + '/' + self.file_list[index]
        with h5py.File(file_name, 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            data = torch.FloatTensor(data.astype(float))
            label = torch.FloatTensor(label.astype(float))

        return data, label

    def __len__(self):
        return self.item_num


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5: # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label


def LFdivide(data, angRes, patch_size, stride):
    uh, vw = data.shape
    h0 = uh // angRes
    w0 = vw // angRes
    bdr = (patch_size - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1
    hE = stride * (numU-1) + patch_size
    wE = stride * (numV-1) + patch_size

    dataE = torch.zeros(hE*angRes, wE*angRes)
    for u in range(angRes):
        for v in range(angRes):
            Im = data[u*h0:(u+1)*h0, v*w0:(v+1)*w0]
            dataE[u*hE : u*hE+h, v*wE : v*wE+w] = ImageExtend(Im, bdr)
    subLF = torch.zeros(numU, numV, patch_size*angRes, patch_size*angRes)
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u*hE + kh*stride
                    vv = v*wE + kw*stride
                    subLF[kh, kw, u*patch_size:(u+1)*patch_size, v*patch_size:(v+1)*patch_size] = dataE[uu:uu+patch_size, vv:vv+patch_size]
    return subLF


def ImageExtend(Im, bdr):
    h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out


def LFintegrate(subLF, angRes, pz, stride, h0, w0):
    numU, numV, pH, pW = subLF.shape
    ph, pw = pH //angRes, pW //angRes
    bdr = (pz - stride) //2
    temp = torch.zeros(stride*numU, stride*numV)
    outLF = torch.zeros(angRes, angRes, h0, w0)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    temp[ku*stride:(ku+1)*stride, kv*stride:(kv+1)*stride] = subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride]

            outLF[u, v, :, :] = temp[0:h0, 0:w0]

    return outLF
def LFintegrate_onedim(subLF, angRes, pz, stride, h0, w0):
    numU, numV, n, pH, pW = subLF.shape
    ph, pw = pH, pW
    bdr = (pz - stride) //2
    temp = torch.zeros(stride*numU, stride*numV)
    outLF = torch.zeros(n, h0, w0)
    for i in range(n):
        for ku in range(numU):
            for kv in range(numV):
                temp[ku*stride:(ku+1)*stride, kv*stride:(kv+1)*stride] = subLF[ku, kv, i, bdr:bdr+stride, bdr:bdr+stride]

        outLF[i, :, :] = temp[0:h0, 0:w0]

    return outLF

def cal_psnr(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)

def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    #return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True, data_range=1.0)
    return metrics.structural_similarity((img1_np*255.0).astype(np.uint8), (img2_np*255.0).astype(np.uint8), gaussian_weights=True)

def cal_metrics(img1, img2):
    n,h,w = img1.shape

    PSNR = np.zeros(shape=(n), dtype='float32')
    SSIM = np.zeros(shape=(n), dtype='float32')
    
    #indicate = ind_cal(8,2)
    bd = 22

    for u in range(n):

        PSNR[u] = cal_psnr(img1[u, bd:-bd, bd:-bd], img2[u, bd:-bd, bd:-bd])
        SSIM[u] = cal_ssim(img1[u, bd:-bd, bd:-bd], img2[u, bd:-bd, bd:-bd])

        pass
    # print(SSIM.sum())
    # print(np.sum(SSIM > 0))

    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean

def ind_cal(angout, angin):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    ind_all = np.arange(angout*angout).reshape(angout, angout)        
    delt = (angout-1) // (angin-1)
    ind_source = ind_all[0:angout:delt, 0:angout:delt]
    ind_source = torch.from_numpy(ind_source.reshape(-1))
    return ind_source
def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st#.squeeze(2)

def LFassemble(data_in, data_syn, angin, angout):
    ind_all = np.arange(angout*angout).reshape(angout, angout)        
    delt = (angout-1) // (angin-1)
    ind_source = ind_all[0:angout:delt, 0:angout:delt].reshape(-1)
    data_syn = np.array(data_syn.permute(1,0,2,3,4).cpu())
    data_in = np.array(data_in.permute(1,0,2,3,4).cpu())
    for i in range(len(ind_source)):
        #print(data_syn.shape)
        data_syn = np.insert(data_syn,ind_source[i], np.expand_dims(data_in[i,...], axis=0),axis = 0)
    data_syn = torch.from_numpy(data_syn)
    data_syn = data_syn.permute(1,0,2,3,4)

    return data_syn
def LFassemble2(data_in, data_syn):
    LF = []
    LF.extend([data_in[:,0], data_syn[:,0], data_in[:,1], data_syn[:,1], data_in[:,2]])
    LF.extend([data_syn[:,2], data_syn[:,3], data_syn[:,4], data_syn[:,5], data_syn[:,6]])
    LF.extend([data_in[:,3], data_syn[:,7], data_in[:,4], data_syn[:,8], data_in[:,5]])
    LF.extend([data_syn[:,9], data_syn[:,10], data_syn[:,11], data_syn[:,12], data_syn[:,13]])
    LF.extend([data_in[:,6], data_syn[:,14], data_in[:,7], data_syn[:,15], data_in[:,8]])
    LF= torch.stack(LF, 1)
    return LF
def FormOutput(x_sv):
    b, n, c, h, w = x_sv.shape
    angRes = int(math.sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out

def SplitInput(data):
    b, a_h, a_w, h, w = data.shape
    data = data.unsqueeze(3)
    #print(data.shape)

    h_1_list = []
    h_1_1 = torch.stack((data[:,0,0,...], data[:,0,1,...]), 1)
    h_1_2 = torch.stack((data[:,0,1,...], data[:,0,2,...]), 1)
    h_1_3 = torch.stack((data[:,0,0,...], data[:,0,2,...]), 1)
    h_1_list.extend([h_1_1, h_1_2, h_1_3])

    h_2_list = []
    h_2_1 = torch.stack((data[:,1,0,...], data[:,1,1,...]), 1)
    h_2_2 = torch.stack((data[:,1,1,...], data[:,1,2,...]), 1)
    h_2_3 = torch.stack((data[:,1,0,...], data[:,1,2,...]), 1)
    h_2_list.extend([h_2_1, h_2_2, h_2_3])

    h_3_list = []
    h_3_1 = torch.stack((data[:,2,0,...], data[:,2,1,...]), 1)
    h_3_2 = torch.stack((data[:,2,1,...], data[:,2,2,...]), 1)
    h_3_3 = torch.stack((data[:,2,0,...], data[:,2,2,...]), 1)
    h_3_list.extend([h_3_1, h_3_2, h_3_3])

    w_1_list = []
    w_1_1 = torch.stack((data[:,0,0,...], data[:,1,0,...]), 1)
    w_1_2 = torch.stack((data[:,1,0,...], data[:,2,0,...]), 1)
    w_1_3 = torch.stack((data[:,0,0,...], data[:,2,0,...]), 1)
    w_1_list.extend([w_1_1, w_1_2, w_1_3])

    w_2_list = []
    w_2_1 = torch.stack((data[:,0,1,...], data[:,1,1,...]), 1)
    w_2_2 = torch.stack((data[:,1,1,...], data[:,2,1,...]), 1)
    w_2_3 = torch.stack((data[:,0,1,...], data[:,2,1,...]), 1)
    w_2_list.extend([w_2_1, w_2_2, w_2_3])

    w_3_list = []
    w_3_1 = torch.stack((data[:,0,2,...], data[:,1,2,...]), 1)
    w_3_2 = torch.stack((data[:,1,2,...], data[:,2,2,...]), 1)
    w_3_3 = torch.stack((data[:,0,2,...], data[:,2,2,...]), 1)
    w_3_list.extend([w_3_1, w_3_2, w_3_3])

    u_list = []
    u_1_1 = torch.stack((data[:,1,0,...], data[:,0,1,...]), 1)
    u_2_1 = torch.stack((data[:,2,0,...], data[:,1,1,...]), 1)
    u_2_2 = torch.stack((data[:,1,1,...], data[:,0,2,...]), 1)
    u_2_3 = torch.stack((data[:,2,0,...], data[:,0,2,...]), 1)
    u_3_1 = torch.stack((data[:,2,1,...], data[:,1,2,...]), 1)
    u_list.extend([u_1_1, u_2_1, u_2_2, u_2_3, u_3_1])

    v_list = []
    v_1_1 = torch.stack((data[:,1,0,...], data[:,2,1,...]), 1)
    v_2_1 = torch.stack((data[:,0,0,...], data[:,1,1,...]), 1)
    v_2_2 = torch.stack((data[:,1,1,...], data[:,2,2,...]), 1)
    v_2_3 = torch.stack((data[:,0,0,...], data[:,2,2,...]), 1)
    v_3_1 = torch.stack((data[:,0,1,...], data[:,1,2,...]), 1)
    v_list.extend([v_1_1, v_2_1, v_2_2, v_2_3, v_3_1])

    return h_1_list, h_2_list, h_3_list, w_1_list, w_2_list, w_3_list, u_list, v_list

# def MergeOut(tmp, h_1_1, h_1_2, h_2_1, h_2_2, h_3_1, h_3_2, w_1_1, w_1_2, w_2_1, w_2_2, w_3_1, w_3_2, u_1, u_2, v_1, v_2, ind_source, ang=5):
#     b,n,c,h,w = tmp.shape
#     out = torch.zeros((b,ang*ang,c,h,w)).to(tmp.device)
#     out[]
def CropPatches_w(image,len,crop):
    #image [1,an2,4,ph,pw]
    #left [1,an2,4,h,lw]
    #middles[n,an2,4,h,mw]
    #right [1,an2,4,h,rw]
    an,f,h,w = image.shape[1:5]
    left = image[:,:,:,:,0:len+crop]
    num = math.floor((w-len-crop)/len)
    middles = torch.Tensor(num,an,f,h,len+crop*2).to(image.device)
    for i in range(num):
        middles[i] = image[0,:,:,:,(i+1)*len-crop:(i+2)*len+crop]      
    right = image[:,:,:,:,-(len+crop):]
    return left,middles,right
def MergePatches_w(left,middles,right,h,w,len,crop):
    #[N,4,h,w]
    n,a = left.shape[0:2]
    out = torch.Tensor(n,a,h,w).to(left.device)
    out[:,:,:,:len] = left[:,:,:,:-crop]
    for i in range(middles.shape[0]): 
        out[:,:,:,len*(i+1):len*(i+2)] = middles[i:i+1,:,:,crop:-crop]        
    out[:,:,:,-len:]=right[:,:,:,crop:]
    return out
def Image2patch(data, patchsize=64, stride=32):
    b, U, V, H, W = data.shape
    patch = []
    for h in range(0, H-patchsize+1, stride):
        for w in range(0, W-patchsize+1, stride):
            temp = data[:, :, :, h:h+patchsize, w:w+patchsize]
            patch.append(temp)
    return patch


    

