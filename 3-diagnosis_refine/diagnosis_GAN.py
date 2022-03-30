#!/usr/bin/python3
#import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
#from PIL import Image
import torch
import torch.nn as nn

from models import Generator
from model.pytorch_resnet import wide_Resnet50_2
from models import Discriminator
from SSIM_loss import SSIM
from utils import ReplayBuffer
from utils import LambdaLR
from utils import GAN_Logger
from datasets import ImageDataset


epoch = 0
n_epochs = 20
batchSize = 1
#w_size = 8
#h_size = 6
dataroot = './dataset/fpfa_dataset/'
lr = 0.0002
decay_epoch = 5
size = 512
input_nc = 3
output_nc = 3
n_class = 2
cuda = True
n_cpu = 0
output_path = './output/diagnosis_gan/'


output_img_path = output_path+'png/'
generator_load_path = './24_netG_A2B.pth'
discriminator_load_path = './24_netD_B.pth'
diagnosis_load_path = './wide_resnet50_2_epoch19.pth'

if not os.path.exists(output_path):
    os.makedirs(output_path)
    if not os.path.exists(output_path+'png/'):
        os.makedirs(output_path+'png/')

if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(input_nc, output_nc)
netD_B = Discriminator(output_nc)
netDiagnosis = wide_Resnet50_2(n_class)

if cuda:
    netG_A2B.cuda()
    netD_B.cuda()
    netDiagnosis.cuda()
    netG_A2B = nn.DataParallel(netG_A2B)
    netD_B = nn.DataParallel(netD_B)
    netDiagnosis = nn.DataParallel(netDiagnosis)

# load weight matrix
netG_A2B.load_state_dict(torch.load(generator_load_path))
netD_B.load_state_dict(torch.load(discriminator_load_path))
netDiagnosis.load_state_dict(torch.load(diagnosis_load_path))

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_diagnosis = torch.nn.MSELoss()
criterion_structure = SSIM()
criterion_identity = torch.nn.SmoothL1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(netG_A2B.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netDiagnosis.parameters(), lr=lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                     lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_A = Tensor(batchSize, input_nc, size, size)
input_B = Tensor(batchSize, output_nc, size, size)
target_diagnosis_A = Variable(Tensor(batchSize, 2))
target_diagnosis_B = Variable(Tensor(batchSize, 2))
#a = torch.zeros(batchSize, 1, 2)
#b = torch.zeros(batchSize, 1, 2)
#target_real = Variable(a.index_fill_(2, torch.tensor([1]), 1.0), requires_grad=False)
#target_fake = Variable(b.index_fill_(2, torch.tensor([0]), 1.0), requires_grad=False)
target_real = Variable(Tensor(batchSize, 1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize, 1).fill_(0.0), requires_grad=False)

fake_B_buffer = ReplayBuffer()
same_B_buffer = ReplayBuffer()

# Dataset loader
tr_dataloader = DataLoader(ImageDataset(dataroot, unaligned=True, size=size, mode='train'), batch_size=batchSize, shuffle=True, num_workers=n_cpu)
te_dataloader = DataLoader(ImageDataset(dataroot, unaligned=True, size=size, mode='test'), batch_size=batchSize, shuffle=True, num_workers=n_cpu)

# Loss plot
logger = GAN_Logger(n_epochs, len(tr_dataloader), output_img_path)
###################################

def patches2raw_img(output):
    raw_img = output.view(w_size, h_size, output.size(1), output.size(2), output.size(3))
    raw_img = raw_img.transpose(1, 2)
    raw_img = raw_img.transpose(2, 3)
    raw_img = raw_img.transpose(0, 1)
    print(raw_img.size(), raw_img.size(0), raw_img.size(1) * raw_img.size(2), raw_img.size(3) * raw_img.size(4))
    channel = raw_img.size(0)
    h_img_size = int(raw_img.size(1) * raw_img.size(2))
    w_img_size = int(raw_img.size(3) * raw_img.size(4))
    out_img = raw_img.reshape(1, channel, h_img_size, w_img_size)
    return out_img

def evaluate(generator, diagnosor, loader):
    for i, batch in enumerate(loader):
        real_A = Variable(input_A.copy_(batch['FAF'][0]))
        real_B = Variable(input_B.copy_(batch['FP'][0]))
        diagnosis_label_A = Variable(target_diagnosis_A.copy_(batch['FAF_label'][0]))
        diagnosis_label_B = Variable(target_diagnosis_B.copy_(batch['FP_label'][0]))
    return

###### Training ######
for e in range(epoch, n_epochs):
    for i, batch in enumerate(tr_dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['FAF'][0]))
        real_B = Variable(input_B.copy_(batch['FP'][0]))
        diagnosis_label_A = Variable(target_diagnosis_A.copy_(batch['FAF_label'][0]))
        diagnosis_label_B = Variable(target_diagnosis_B.copy_(batch['FP_label'][0]))

        if cuda:
            real_A.cuda()
            real_B.cuda()
            diagnosis_label_A.cuda()
            diagnosis_label_B.cuda()

        ###### Generators FAF to FP ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_FAF2FP(FP) should equal FP if real FP is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0

        #GAN loss
        # GAN loss for discriminator
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN = criterion_GAN(pred_fake, target_real)
        #GAN loss for diagnosor
        pred_diag = netDiagnosis(fake_B)
        loss_diag_GAN = criterion_GAN(pred_diag, diagnosis_label_A)

        # structure loss
        #loss_structure_B = criterion_structure(fake_B, real_A)

        # Total loss
        loss_G = 5.0 * loss_GAN + loss_diag_GAN + loss_identity_B #+ loss_structure_B
        loss_G.backward()#retain_graph=True)

        optimizer_G.step()
        ###################################

        ###### Discriminator  ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_B.step()
        ###################################

        ###### Discriminator: Diagnosis Network ######
        optimizer_D.zero_grad()
        # diagnosis loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        same_B = same_B_buffer.push_and_pop(same_B)

        pred_fake = netDiagnosis(fake_B)
        pred_same = netDiagnosis(same_B)
        pred_real = netDiagnosis(real_B)
        #print(pred_fake.size(), diagnosis_label_A.size())

        loss_diagnosis_fake = criterion_diagnosis(pred_fake, diagnosis_label_A)
        loss_diagnosis_same = criterion_diagnosis(pred_same, diagnosis_label_B)
        loss_diagnosis_real = criterion_diagnosis(pred_real, diagnosis_label_B)

        # Total loss
        loss_D = loss_diagnosis_fake + (loss_diagnosis_same + loss_diagnosis_real) * 0.5
        loss_D.backward()

        optimizer_D.step()
        ###################################

        # Progress report (http://localhost:8097)
        if i % 10 == 0:
            logger.log({'loss_G': loss_G, 'loss_G_identity': loss_identity_B,
                        #'loss_G_sturcture': loss_structure_B,
                        'loss_D': loss_D_A},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B, 'same_B': same_B})

   #evaluate(netG_A2B, netDiagnosis, te_dataloader)
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_B.step()
    lr_scheduler_D.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), output_path+str(e)+'_finetune_netG_A2B.pth')
    torch.save(netD_B.state_dict(), output_path + str(e) + '_finetune_netD_A.pth')
    torch.save(netDiagnosis.state_dict(), output_path+str(e)+'_finetune_netDiagnosis.pth')
###################################
