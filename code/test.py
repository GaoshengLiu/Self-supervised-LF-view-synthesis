import time
import argparse
import scipy.misc
import torch.backends.cudnn as cudnn
from utils import *
from model_light import Net
from tqdm import tqdm
import scipy.io as sio
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angin", type=int, default=3, help="angular resolution")
    parser.add_argument("--angout", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument('--testset_dir', type=str, default='../Data/TestData_SIG_3x3_ASR_5x5/')

    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=32, help="The stride between two test patches is set to patchsize/2")

    parser.add_argument('--modelh_path', type=str, default='../checkpoint/self-ASRNet_h3xSR_x5_epoch_12.pth.tar')
    parser.add_argument('--modelw_path', type=str, default='../checkpoint/self-ASRNet_w3xSR_x5_epoch_12.pth.tar')
    #parser.add_argument('--modelu_path', type=str, default='../checkpoint/self-ASRNet_u3xSR_x5_epoch_12.pth.tar')
    #parser.add_argument('--modelv_path', type=str, default='../checkpoint/self-ASRNet_v3xSR_x5_epoch_12.pth.tar')
    parser.add_argument('--save_path', type=str, default='../Results/')

    return parser.parse_args()


def test(cfg, test_Names, test_loaders):

    net_h = Net(2)#cfg.angin, cfg.angout)#, cfg.upscale_factor)
    net_w = Net(2)#cfg.angin, cfg.angout)#, cfg.upscale_factor)
    #net_u = Net(2)#cfg.angin, cfg.angout)#, cfg.upscale_factor)
    #net_v = Net(2)#cfg.angin, cfg.angout)#, cfg.upscale_factor)
    net_h.to(cfg.device)
    net_w.to(cfg.device)
    #net_u.to(cfg.device)
    #net_v.to(cfg.device)
    cudnn.benchmark = True

    if os.path.isfile(cfg.modelh_path) and  os.path.isfile(cfg.modelw_path) and os.path.isfile(cfg.modelu_path) and os.path.isfile(cfg.modelv_path):
        model_h = torch.load(cfg.modelh_path, map_location={'cuda:0': cfg.device})
        net_h.load_state_dict(model_h['state_dict'])
        model_w = torch.load(cfg.modelw_path, map_location={'cuda:0': cfg.device})
        net_w.load_state_dict(model_w['state_dict'])
        #model_u = torch.load(cfg.modelu_path, map_location={'cuda:0': cfg.device})
        #net_u.load_state_dict(model_u['state_dict'])
        #model_v = torch.load(cfg.modelv_path, map_location={'cuda:0': cfg.device})
        #net_v.load_state_dict(model_v['state_dict'])
    else:
        if not os.path.isfile(cfg.modelh_path):
            print("=> no model found at '{}'".format(cfg.modelh_path))
        if not os.path.isfile(cfg.modelw_path):
            print("=> no model found at '{}'".format(cfg.modelw_path))
        #if not os.path.isfile(cfg.modelu_path):
        #    print("=> no model found at '{}'".format(cfg.modelu_path))
        #if not os.path.isfile(cfg.modelv_path):
        #    print("=> no model found at '{}'".format(cfg.modelv_path))

    net = [net_h, net_w]#, net_u, net_v]
    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            psnr_epoch_test, ssim_epoch_test = inference(test_loader, test_name, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            pass
        pass


def inference(test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data_in, label) in (enumerate(test_loader)):
        data_in = data_in.squeeze().to(cfg.device)  # numU, numV, h*angin, w*angin
        ah,aw,h,w = data_in.shape
        #print(data.shape)
        data = data_in.contiguous().permute(0, 2, 1, 3).contiguous().view(ah*h, aw*w)
        label = label.squeeze()
        #print(label.shape)

        uh, vw = data.shape
        h0, w0 = uh // cfg.angin, vw // cfg.angin
        subLFin = LFdivide(data, cfg.angin, cfg.patchsize, cfg.stride)  # numU, numV, h*angin, w*angin
        numU, numV, H, W = subLFin.shape
        
        s = time.time()
        minibatch = 4
        num_inference = numU*numV//minibatch
        tmp_in = subLFin.contiguous().view(numU*numV, subLFin.shape[2], subLFin.shape[3])
        
        with torch.no_grad():
            out_lf = []
            for idx_inference in range(num_inference):
                tmp = tmp_in[idx_inference*minibatch:(idx_inference+1)*minibatch,:,:].unsqueeze(1)
                out_lf.append(test_model(tmp.to(cfg.device), net))
            if (numU*numV)%minibatch:
                tmp = tmp_in[(idx_inference+1)*minibatch:,:,:].unsqueeze(1)
                out_lf.append(test_model(tmp.to(cfg.device), net))#
        out_lf = torch.cat(out_lf, 0)
        #print(out_lf.shape)
        subLFout = out_lf.view(numU, numV, cfg.angout*cfg.angout-cfg.angin*cfg.angin,  cfg.patchsize, cfg.patchsize)
        #curr_time = time.time()-s

        outLF = LFintegrate_onedim(subLFout, cfg.angout, cfg.patchsize, cfg.stride, h0, w0)

        psnr, ssim = cal_metrics(label, outLF)#, cfg.angin, cfg.angout)
        #print(pred_y.shape)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        isExists = os.path.exists(cfg.save_path + test_name)
        if not (isExists ):
            os.makedirs(cfg.save_path + test_name)

        sio.savemat(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                        {'LF': outLF.cpu().numpy()})
        pass


    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test
def test_model(data, net):
    data = LFsplit(data, cfg.angin)
    #print(data.shape)
    b,n,c,h,w = data.shape
    net_h, net_w, net_u, net_v = net[0], net[1], net[2], net[3]
    h_1_list, h_2_list, h_3_list, w_1_list, w_2_list, w_3_list, u_list, v_list = SplitInput(data.contiguous().view(b,cfg.angin,cfg.angin,h,w))
    buffer = []
    h_1_1 = net_h(h_1_list[0])
    h_1_2 = net_h(h_1_list[1])
    h_2_1 = net_h(h_2_list[0])
    h_2_2 = net_h(h_2_list[1])
    h_3_1 = net_h(h_3_list[0])
    h_3_2 = net_h(h_3_list[1])

    w_1_1 = net_w(w_1_list[0])
    w_1_2 = net_w(w_1_list[1])
    w_2_1 = net_w(w_2_list[0])
    w_2_2 = net_w(w_2_list[1])
    w_3_1 = net_w(w_3_list[0])
    w_3_2 = net_w(w_3_list[1])

    torch.stack((data[:,1,0,...], data[:,0,1,...]), 1)

    u_0 = net_w(torch.stack((h_1_1, h_2_1), 1))
    u_1 = net_w(torch.stack((h_2_1, h_3_1), 1))
    u_2 = net_w(torch.stack((h_1_2, h_2_2), 1))
    u_4 = net_w(torch.stack((h_2_2, h_3_2), 1))

    d_0 = u_0 
    d_1 = u_1 
    d_2 = u_2 
    d_4 = u_4 

    buffer.extend([h_1_1, h_1_2, w_1_1, d_0, w_2_1, d_2, w_3_1, h_2_1, h_2_2, w_1_2, d_1, w_2_2, d_4, w_3_2, h_3_1, h_3_2])

    out = torch.stack(buffer, 1)
    return out
'''
def inference(test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angin, w*angin
        ah,aw,h,w = data.shape
        data = data.contiguous().view(1,ah*aw,-1,h,w)
        label = label.squeeze()

        length = 64
        crop = 8
        input_l, input_m, input_r = CropPatches_w(data, length, crop)

        pred_l = test_model(input_l, net)
        #print(pred_l.shape)
        ################### middles ###################
        pred_m = torch.Tensor(input_m.shape[0], cfg.angout*cfg.angout-cfg.angin*cfg.angin, input_m.shape[3], input_m.shape[4])
        for i in range(input_m.shape[0]):
            cur_input_m = input_m[i:i+1]
            pred_m[i:i+1] = test_model(cur_input_m, net)
        #print(pred_m.shape)
            
        ################### right ###################
        pred_r = test_model(input_r, net)
        #print(pred_r.shape)

        pred_y = MergePatches_w(pred_l, pred_m, pred_r, data.shape[3], data.shape[4], length, crop)  #[N,an2,hs,ws]
        pred_y = pred_y.squeeze()
        
        psnr, ssim = cal_metrics(label, pred_y)
        #print(pred_y.shape)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        isExists = os.path.exists(cfg.save_path + test_name)
        if not (isExists ):
            os.makedirs(cfg.save_path + test_name)

        sio.savemat(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                        {'LF': pred_y.cpu().numpy()})
        pass


    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test

def test_model(data, net):
    b,n,c,h,w = data.shape
    net_h, net_w, net_u, net_v = net[0], net[1], net[2], net[3]
    h_1_list, h_2_list, h_3_list, w_1_list, w_2_list, w_3_list, u_list, v_list = SplitInput(data.contiguous().view(b,cfg.angin,cfg.angin,h,w))
    buffer = []
    h_1_1 = net_h(h_1_list[0])
    h_1_2 = net_h(h_1_list[1])
    h_2_1 = net_h(h_2_list[0])
    h_2_2 = net_h(h_2_list[1])
    h_3_1 = net_h(h_3_list[0])
    h_3_2 = net_h(h_3_list[1])

    w_1_1 = net_w(w_1_list[0])
    w_1_2 = net_w(w_1_list[1])
    w_2_1 = net_w(w_2_list[0])
    w_2_2 = net_w(w_2_list[1])
    w_3_1 = net_w(w_3_list[0])
    w_3_2 = net_w(w_3_list[1])

    u_1 = net_u(u_list[0])
    u_2 = net_u(u_list[4])

    v_1 = net_v(v_list[0])
    v_2 = net_v(v_list[4])
    buffer.extend([h_1_1, h_1_2, w_1_1, u_1, w_2_1, v_2, w_3_1, h_2_1, h_2_2, w_1_2, v_1, w_2_2, u_2, w_3_2, h_3_1, h_3_2])

    out = torch.cat(buffer, 1)
    #print(out.shape)
    return out
'''
def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    test(cfg, test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
