import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
import cv2
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description="Choose mode")
parser.add_argument('-mode', required=True, choices=['train', 'test'])
parser.add_argument('-dim', type=int, default=64)
parser.add_argument('-num_epochs', type=int, default=2000)
parser.add_argument('-image_scale_h', type=int, default=256)
parser.add_argument('-image_scale_w', type=int, default=256)
parser.add_argument('-batch', type=int, default=8)
parser.add_argument('-img_cut', type=int, default=4)
parser.add_argument('-lr', type=float, default=5e-5)
parser.add_argument('-data_path', type=str, default='../munich/train/img3')
parser.add_argument('-label_path', type=str, default='../munich/train/lab3')
parser.add_argument('-load_model', required=True, choices=['True', 'False'], help='choose True or False')
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
print("use_cuda:", use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
IMG_CUT = opt.img_cut
writer = SummaryWriter('./runs2/fcrn')
class ResidualBlockClass(nn.Module):
    def __init__(self, name, input_dim, output_dim, resample=None, activate='relu'):
        super(ResidualBlockClass, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample 
        self.batchnormlize_1 = nn.BatchNorm2d(input_dim)
        self.activate = activate
        if resample == 'down':
            self.conv_0        = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.conv_shortcut = nn.AvgPool2d(3, stride=2, padding=1)
            self.conv_1        = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
            self.conv_2        = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=2, padding=1)
            self.batchnormlize_2 = nn.BatchNorm2d(input_dim)
        elif resample == 'up':
            self.conv_0        = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.conv_shortcut = nn.Upsample(scale_factor=2)
            self.conv_1        = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.conv_2        = nn.ConvTranspose2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=2, padding=2, 
                                           output_padding=1, dilation=2)
            self.batchnormlize_2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            self.conv_shortcut = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.conv_1        = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
            self.conv_2        = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.batchnormlize_2 = nn.BatchNorm2d(input_dim)
        else:
            raise Exception('invalid resample value')
        
    def forward(self, inputs):
        if self.output_dim == self.input_dim and self.resample == None:
            shortcut = inputs 
        elif self.resample == 'down':
            x = self.conv_0(inputs)
            shortcut = self.conv_shortcut(x)
        elif self.resample == None:
            x = inputs
            shortcut = self.conv_shortcut(x)   
        else:
            x = self.conv_0(inputs)
            shortcut = self.conv_shortcut(x)
        if self.activate == 'relu':
            x = inputs
            x = self.batchnormlize_1(x)
            x = F.relu(x)
            x = self.conv_1(x)
            x = self.batchnormlize_2(x)
            x = F.relu(x)
            x = self.conv_2(x) 
            return shortcut + x
        else:   
            x = inputs
            x = self.batchnormlize_1(x)
            x = F.leaky_relu(x)
            x = self.conv_1(x)
            x = self.batchnormlize_2(x)
            x = F.leaky_relu(x)
            x = self.conv_2(x)
            return shortcut + x 
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation=None):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
 
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X (W*H) X C
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
 
        out = self.gamma*out + x
        return out
class Fcrn(nn.Module):
    def __init__(self, dim=opt.dim):
        super(Fcrn, self).__init__()
        self.dim = dim
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.residual_block_1_down_1     = ResidualBlockClass('Detector.Res1', 1*dim, 2*dim, resample='down', activate='leaky_relu')
		#128x128
        self.residual_block_2_down_1     = ResidualBlockClass('Detector.Res2', 2*dim, 4*dim, resample='down', activate='leaky_relu')
		#64x64
        self.residual_block_3_down_1     = ResidualBlockClass('Detector.Res3', 4*dim, 4*dim, resample='down', activate='leaky_relu')
		#32x32
        self.residual_block_4_down_1     = ResidualBlockClass('Detector.Res4', 4*dim, 6*dim, resample='down', activate='leaky_relu')
		#16x16
        # self.residual_block_5_down_1     = ResidualBlockClass('Detector.Res5', 6*dim, 6*dim, resample='down', activate='leaky_relu')
        self.residual_block_6_none_1     = ResidualBlockClass('Detector.Res6', 6*dim, 6*dim, resample=None, activate='leaky_relu')
        self.residual_block_7_none_1       = ResidualBlockClass('Detector.Res7', 6*dim, 6*dim, resample=None, activate='leaky_relu')
        #32x32
        # self.sa_0                        = Self_Attn(in_dim=6*dim)

        self.residual_block_8_up_1       = ResidualBlockClass('Detector.Res8', 6*dim, 4*dim, resample='up', activate='leaky_relu')
        #64x64
        # self.sa_1                        = Self_Attn(in_dim=4*dim)

        self.residual_block_9_up_1       = ResidualBlockClass('Detector.Res9', 4*dim, 2*dim, resample='up', activate='leaky_relu')
        #128x128
        # self.sa_2                        = Self_Attn(in_dim=2*dim)

        self.residual_block_10_up_1      = ResidualBlockClass('Detector.Res10', 2*dim, 2*dim, resample='up', activate='leaky_relu')
        #256x256
        # self.sa_3                        = Self_Attn(in_dim=2*dim)

        self.residual_block_11_up_1      = ResidualBlockClass('Detector.Res11', 2*dim, 1*dim, resample='up', activate='leaky_relu')
    def forward(self, x):
        x = self.conv_1(x)
        x = self.residual_block_1_down_1(x)
        # x1 = 0.8*x + 0.2*torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3],device=device)
        x = self.residual_block_2_down_1(x)
        # x1 = 0.8*x + 0.2*torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3],device=device)
        x = self.residual_block_3_down_1(x)
        # x1 = 0.8*x + 0.2*torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3],device=device)
        x = self.residual_block_4_down_1(x)
        # x1 = 0.8*x + 0.2*torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3],device=device)
        # x = self.residual_block_5_down_1(x)
        x = self.residual_block_6_none_1(x)
        x = self.residual_block_7_none_1(x)
        # x = self.sa_0(x)
        x = self.residual_block_8_up_1(x)
        # x = self.sa_1(x)
        x = self.residual_block_9_up_1(x)
        # x = self.sa_2(x)
        x = self.residual_block_10_up_1(x)
        # x = self.sa_3(x)
        x = self.residual_block_11_up_1(x)
        x = F.normalize(x, dim=[0, 2, 3])
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.conv_2(x)
        x = F.sigmoid(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
class GAN_Dataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(opt.data_path))
    
    def __getitem__(self, idx):
        img_name = os.listdir(opt.data_path)[idx]
        imgA = cv2.imread(opt.data_path + '/' + img_name)
        imgA = cv2.resize(imgA, (opt.image_scale_w, opt.image_scale_h))
        imgB = cv2.imread(opt.label_path + '/' + img_name[:-4] + '.png', 0)
        imgB = cv2.resize(imgB, (opt.image_scale_w, opt.image_scale_h))
        imgB[imgB>30] = 255
#         imgB[imgB>100] = 1
        imgB = imgB/255 
        #imgB = imgB.astype('uint8')
        imgB = torch.FloatTensor(imgB)
        imgB = torch.unsqueeze(imgB, 0)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)
            
        return imgA, imgB

img_road = GAN_Dataset(transform)
train_dataloader = DataLoader(img_road, batch_size=opt.batch, shuffle=True)
print(len(train_dataloader.dataset), train_dataloader.dataset[7][1].shape)

class test_Dataset(Dataset):
    # DATA_PATH = './test/img'
    # LABEL_PATH = './test/lab'
    def __init__(self, transform=None):
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir('../munich/test/img'))
    
    def __getitem__(self, idx):
        img_name = os.listdir('../munich/test/img')
        img_name.sort(key=lambda x:int(x[:-4]))
        img_name = img_name[idx]
        imgA = cv2.imread('../munich/test/img' + '/' + img_name)
        imgA = cv2.resize(imgA, (opt.image_scale_w, opt.image_scale_h))
        imgB = cv2.imread('../munich/test/lab' + '/' + img_name[:-4] + '.png', 0)
        imgB = cv2.resize(imgB, (opt.image_scale_w, opt.image_scale_h))
        imgB[imgB>0] = 255
        imgB = imgB/255
#         imgB[imgB>100] = 1 
        #imgB = imgB.astype('uint8')
        imgB = torch.FloatTensor(imgB)
        imgB = torch.unsqueeze(imgB, 0)
        #print(imgB.shape)
        if self.transform:
            #imgA = imgA/255
            #imgA = np.transpose(imgA, (2, 0, 1))
            #imgA = torch.FloatTensor(imgA)
            imgA = self.transform(imgA)           
        return imgA, imgB

img_road_test = test_Dataset(transform)

test_dataloader = DataLoader(img_road_test, batch_size=1, shuffle=False)
print(len(test_dataloader.dataset), test_dataloader.dataset[7][1].shape)

loss = nn.BCELoss()

fcrn = Fcrn()
fcrn = nn.DataParallel(fcrn)
fcrn = fcrn.to(device)
if opt.load_model == 'True':
    fcrn.load_state_dict(torch.load('./model_fcrn/fcrn_munich.pkl'))

fcrn_optimizer = optim.Adam(fcrn.parameters(), lr=opt.lr)
fcrn_scheduler = optim.lr_scheduler.StepLR(fcrn_optimizer,step_size=500,gamma = 0.5)

def train(device, train_dataloader, epoch):
    fcrn.train()
    for batch_idx, (road, road_label)in enumerate(train_dataloader):
        road, road_label = road.to(device), road_label.to(device)
        detect = fcrn(road)
        detect_np = detect.detach().cpu()
        detect_np = np.transpose(np.array(utils.make_grid(detect_np, nrow=IMG_CUT, padding=0)), (1, 2, 0))
#         blur = cv2.GaussianBlur(detect_np*255, (3, 3), 0)
        # _, thresh = cv2.threshold(detect_np*255,50,255,cv2.THRESH_BINARY)
        road_np = road.detach().cpu()
        road_np = np.transpose(np.array(utils.make_grid(road_np, nrow=IMG_CUT, padding=0)), (1, 2, 0))
        road_label_np = road_label.detach().cpu()
        road_label_np = np.transpose(np.array(utils.make_grid(road_label_np, nrow=IMG_CUT, padding=0)), (1, 2, 0))
        fcrn_loss = loss(detect, road_label)
        fcrn_loss += torch.mean(torch.abs(detect-road_label))/(torch.mean(torch.abs(detect+road_label))+0.001)
        fcrn_optimizer.zero_grad()
        fcrn_loss.backward()
        fcrn_optimizer.step()

        writer.add_scalar('fcrn_loss', fcrn_loss.data.item(), global_step = batch_idx)
        if batch_idx % 20 == 0:
            tqdm.write('[{}/{}] [{}/{}] Loss_Fcrn: {:.6f}'
                .format(epoch, num_epochs, batch_idx, len(train_dataloader), fcrn_loss.data.item()))
        if batch_idx % 300 == 0:
            mix = np.concatenate(((road_np+1)*255/2, road_label_np*255, detect_np*255), axis=0)
            cv2.imwrite("./results_fcrn/dete{}_{}.png".format(epoch, batch_idx), mix)
def test(device, test_dataloader):
    # fcrn.eval()

    for batch_idx, (road, road_label)in enumerate(test_dataloader):
        road, lab = road.to(device), road_label.to(device)
        # z = torch.randn(road.shape[0], 1, IMAGE_SCALE, IMAGE_SCALE, device=device)
        # img_noise = torch.cat((road, z), dim=1)
        # fake_feature = Gen(img_noise)
        det_road = fcrn(road)
        label = det_road.detach().cpu()
        label = np.transpose(np.array(utils.make_grid(label, padding=0, nrow=1)), (1, 2, 0))
        road_np = road.detach().cpu()
        road_np = np.transpose(np.array(utils.make_grid(road_np, padding=0, nrow=1)), (1, 2, 0))
        lab_np = lab.detach().cpu()
        lab_np = np.transpose(np.array(utils.make_grid(lab_np, padding=0, nrow=1)), (1, 2, 0))
        # blur = cv2.GaussianBlur(label*255, (3, 3), 0)
        _, thresh = cv2.threshold(label*255, 50, 255, cv2.THRESH_BINARY)
        # mix = np.concatenate(((road_np+1)*255/2, lab_np*255, label*255), axis=0)
        # cv2.imwrite('../test1/lab_detefcrn/{}.png'.format(batch_idx), mix)
        cv2.imwrite('../test1/lab_detefcrn/{}.png'.format(batch_idx), thresh)
        print('testing...')
        print('{}/{}'.format(batch_idx, len(test_dataloader)))
    print('Done!')

def iou(path_img, path_lab):
    img_name = os.listdir(path_img)
    print(img_name)
    iou_list = []
    for i in range(len(img_name)):
        det = img_name[i]
        det = cv2.imread(path_img + '/' + det, 0)
        lab = img_name[i]
        lab = cv2.imread(path_lab + '/' + lab[:-4] + '.png', 0)

        count0, count1, a, count2 = 0, 0, 0, 0
        for j in range(det.shape[0]):
            for k in range(det.shape[1]):
                #TP
                if det[j][k] != 0 and lab[j][k] != 0:
                    count0 += 1
                #FN
                elif det[j][k] == 0 and lab[j][k] != 0:
                    count1 += 1
                #FP
                elif det[j][k] != 0 and lab[j][k] == 0:
                    count2 += 1
                #iou = (count0)/(count0 + count1 + count2)
                iou = count0/(count1 + count0 + count2 + 0.00001)
        iou_list.append(iou)
        print(img_name[i], ':', iou)
    print('mean_iou:', sum(iou_list)/len(iou_list))

if __name__ == '__main__':
    if opt.mode == 'train':
        num_epochs = opt.num_epochs
        for epoch in tqdm(range(num_epochs)):
            train(device, train_dataloader, epoch)
            fcrn_scheduler.step()
            if (epoch + 1) % 100 == 0:
                now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
                torch.save(fcrn.state_dict(), './model_fcrn/fcrn_munich.pkl')
                print('testing...')
                test(device, test_dataloader)
                iou('../test1/lab_detefcrn', '../munich/test/lab')
                
            
    if opt.mode == 'test':
        test(device, test_dataloader)
        iou('../test1/lab_detefcrn', '../munich/test/lab')