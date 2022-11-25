import time
import argparse
from torch.autograd import Variable
import itertools
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import *
import math
from model_light import Net
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# Settings
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angin", type=int, default=3, help="angular resolution")
    parser.add_argument("--angout", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument('--model_name', type=str, default='ASRNet')
    parser.add_argument('--trainset_dir', type=str, default='../Data/TrainingData_SIG_3x3_ASR_5x5')
    parser.add_argument('--testset_dir', type=str, default='../Data/TestData_SIG_3x3_ASR_5x5/')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=5, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument("--smooth", type=float, default=0.001, help="smooth loss")
    parser.add_argument("--epi", type=float, default=1.0, help="epi loss")

    parser.add_argument("--patchsize", type=int, default=64, help="crop into patches for validation")
    parser.add_argument("--stride", type=int, default=32, help="stride for patch cropping")

    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='../checkpoint/DFnet_2xSR_5x5_epoch_35.pth.tar')

    return parser.parse_args()

if not os.path.exists('../checkpoint'):
        os.mkdir('../checkpoint')

def train(cfg, train_loader, test_Names, test_loaders):

    net_h = Net(2)#cfg.angin, cfg.angout)#, cfg.upscale_factor)
    net_w = Net(2)#cfg.angin, cfg.angout)#, cfg.upscale_factor)
    #net_u = Net(2)#cfg.angin, cfg.angout)#, cfg.upscale_factor)
    #net_v = Net(2)#cfg.angin, cfg.angout)#, cfg.upscale_factor)
    net_h.to(cfg.device)
    net_w.to(cfg.device)
    #net_u.to(cfg.device)
    #net_v.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0
    ##### get input index ######         
    ind_all = np.arange(cfg.angout*cfg.angout).reshape(cfg.angout, cfg.angout)        
    delt = (cfg.angout-1) // (cfg.angin-1)
    ind_source = ind_all[0:cfg.angout:delt, 0:cfg.angout:delt]
    ind_source = torch.from_numpy(ind_source.reshape(-1))

    # if cfg.load_pretrain:
    #     if os.path.isfile(cfg.model_path):
    #         model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
    #         net.load_state_dict(model['state_dict'])
    #         epoch_state = model["epoch"]
    #         print("load pre-train at epoch {}".format(epoch_state))
    #     else:
    #         print("=> no model found at '{}'".format(cfg.load_model))

    #net = torch.nn.DataParallel(net, device_ids=[0, 1])

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer_h = torch.optim.Adam([paras for paras in net_h.parameters() if paras.requires_grad == True], lr=cfg.lr)
    optimizer_w = torch.optim.Adam([paras for paras in net_w.parameters() if paras.requires_grad == True], lr=cfg.lr)
    #optimizer_u = torch.optim.Adam(itertools.chain(net_u.parameters(),
                                                   #net_v.parameters()), lr=cfg.lr)
    #optimizer_v = torch.optim.Adam([paras for paras in net_v.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler_h = torch.optim.lr_scheduler.StepLR(optimizer_h, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, step_size=cfg.n_steps, gamma=cfg.gamma)
    #scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=cfg.n_steps, gamma=cfg.gamma)
    #scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler_h._step_count = epoch_state
    scheduler_w._step_count = epoch_state
    #scheduler_u._step_count = epoch_state
    #scheduler_v._step_count = epoch_state
    loss_epoch = []
    loss_list = []
    

    for idx_epoch in range(epoch_state, cfg.n_epochs):  
        for idx_iter, (data) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data = Variable(data).to(cfg.device)
            #print(data.shape)
            h_1_list, h_2_list, h_3_list, w_1_list, w_2_list, w_3_list, u_list, v_list = SplitInput(data)
            syn_h_1_1_view = net_h(h_1_list[0])
            syn_h_1_2_view = net_h(h_1_list[1])
            central_h1_view_label = net_h(h_1_list[2])
            central_h1_view_syn = net_h(torch.stack((syn_h_1_1_view, syn_h_1_2_view), 1))
            loss_h = criterion_Loss(central_h1_view_label, data[:,0,1,...].unsqueeze(1)) + criterion_Loss(central_h1_view_syn, data[:,0,1,...].unsqueeze(1))

            syn_h_2_1_view = net_h(h_2_list[0])
            syn_h_2_2_view = net_h(h_2_list[1])
            central_h2_view_label = net_h(h_2_list[2])
            central_h2_view_syn = net_h(torch.stack((syn_h_2_1_view, syn_h_2_2_view), 1))
            loss_h = loss_h + criterion_Loss(central_h2_view_label, data[:,1,1,...].unsqueeze(1)) + criterion_Loss(central_h2_view_syn, data[:,1,1,...].unsqueeze(1))

            syn_h_3_1_view = net_h(h_3_list[0])
            syn_h_3_2_view = net_h(h_3_list[1])
            central_h3_view_label = net_h(h_3_list[2])
            central_h3_view_syn = net_h(torch.stack((syn_h_3_1_view, syn_h_3_2_view), 1))
            loss_h = loss_h + criterion_Loss(central_h3_view_label, data[:,2,1,...].unsqueeze(1)) + criterion_Loss(central_h3_view_syn, data[:,2,1,...].unsqueeze(1))
            
            syn_w_1_1_view = net_w(w_1_list[0])
            syn_w_1_2_view = net_w(w_1_list[1])
            central_w1_view_label = net_w(w_1_list[2])
            central_w1_view_syn = net_w(torch.stack((syn_w_1_1_view, syn_w_1_2_view), 1))
            loss_w = criterion_Loss(central_w1_view_label, data[:,1,0,...].unsqueeze(1)) + criterion_Loss(central_w1_view_syn, data[:,1,0,...].unsqueeze(1))

            syn_w_2_1_view = net_w(w_2_list[0])
            syn_w_2_2_view = net_w(w_2_list[1])
            central_w2_view_label = net_w(w_2_list[2])
            central_w2_view_syn = net_w(torch.stack((syn_w_2_1_view, syn_w_2_2_view), 1))
            loss_w = loss_w + criterion_Loss(central_w2_view_label, data[:,1,1,...].unsqueeze(1)) + criterion_Loss(central_w2_view_syn, data[:,1,1,...].unsqueeze(1))

            syn_w_3_1_view = net_w(w_3_list[0])
            syn_w_3_2_view = net_w(w_3_list[1])
            central_w3_view_label = net_w(w_3_list[2])
            central_w3_view_syn = net_w(torch.stack((syn_w_3_1_view, syn_w_3_2_view), 1))
            loss_w = loss_w + criterion_Loss(central_w3_view_label, data[:,2,1,...].unsqueeze(1)) + criterion_Loss(central_w3_view_syn, data[:,2,1,...].unsqueeze(1))

            optimizer_h.zero_grad()
            loss_h.backward()
            optimizer_h.step()

            optimizer_w.zero_grad()
            loss_w.backward()
            optimizer_w.step()

            loss_epoch.append((loss_h+loss_w).data.cpu())        
        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net_h.state_dict()},
                save_path='../checkpoint/', filename=cfg.model_name + '_h' + str(cfg.angin)+ 'xSR_' + 'x' + str(cfg.angout) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net_w.state_dict()},
                save_path='../checkpoint/', filename=cfg.model_name + '_w' + str(cfg.angin)+ 'xSR_' + 'x' + str(cfg.angout) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []

        ''' evaluation '''
        net = [net_h, net_w]
        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []
            for index, test_name in enumerate(test_Names):
                test_loader = test_loaders[index]
                psnr_epoch_test, ssim_epoch_test =  valid(test_loader, test_name, net)
                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
                pass
            pass

        del net
        scheduler_h.step()
        scheduler_w.step()
        #scheduler_u.step()
        pass


def valid(test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angin, w*angin
        ah,aw,h,w = data.shape
        #print(data.shape)
        data = data.contiguous().permute(0, 2, 1, 3).contiguous().view(ah*h, aw*w)
        label = label.squeeze()

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
        pass


    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test
def test_model(data, net):
    data = LFsplit(data, cfg.angin)
    #print(data.shape)
    b,n,c,h,w = data.shape
    net_h, net_w = net[0], net[1]
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
    #print(out.shape)
    #out = LFassemble(data, out, cfg.angin, cfg.angout)
    #print(out.shape)
    return out
def save_ckpt(state, save_path='../checkpoint', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))


def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=cfg.batch_size, shuffle=True)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    train(cfg, train_loader, test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
