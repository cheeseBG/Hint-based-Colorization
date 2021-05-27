import torch
import torch.utils.data as data
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch.optim as optim

from PIL import Image
from math import log10, sqrt
from torchvision.transforms import ToPILImage
from IQA_pytorch import SSIM, utils

from model import ColorizationModel
from dataloader import ColorHintDataset


# 이어서 학습################

# save_path = './ColorizationNetwork'
# model_path = os.path.join(save_path, 'validation_model.tar')
# state_dict = torch.load(model_path)
#
# print(state_dict['memo'])
# print(state_dict.keys())
# print(state_dict['PSNR'])

##########################


def image_show(img):
  if isinstance(img, torch.Tensor):
    img = ToPILImage(img)()
  plt.imshow(img)
  plt.show()


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0))), 0, 1) * 255.0
    return image_numpy.astype(imtype)


def tensor2npy(input_image, imtype=np.uint8):
  if isinstance(input_image, torch.Tensor):
      image_tensor = input_image.data
  else:
      return input_image
  image_numpy = image_tensor[0].cpu().float().numpy()
  if image_numpy.shape[0] == 1:
      image_numpy = np.tile(image_numpy, (3, 1, 1))
  image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) ),0, 1) * 255.0
  return image_numpy


'''
 PSNR
'''
def PSNR(mse):

    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


'''
 SSIM
'''
def ssim(img_gt, img_output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_gt = utils.prepare_image(Image.fromarray(img_gt).convert("RGB")).to(device)
    img_output = utils.prepare_image(Image.fromarray(img_output).convert("RGB")).to(device)
    model = SSIM(channels=3)
    score = model(img_gt, img_output, as_loss=False)

    return score


# Change to your data root directory
root_path = "./"
# Depend on runtime setting
use_cuda = True

train_dataset = ColorHintDataset(root_path, 128)
train_dataset.set_mode("training")

val_dataset = ColorHintDataset(root_path, 128)
val_dataset.set_mode("validation")

train_dataloader = data.DataLoader(train_dataset, batch_size=25, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=25, shuffle=True)

batch = iter(train_dataloader)
sample = batch.next()

'''
    Image show
'''
# for i, data in enumerate(tqdm.tqdm(train_dataloader)):
#     if use_cuda:
#         l = data["l"].to('cuda')
#         ab = data["ab"].to('cuda')
#         hint = data["hint"].to('cuda')
#
#     gt_image = torch.cat((l, ab), dim=1)
#     hint_image = torch.cat((l, hint), dim=1)
#
#     gt_np = tensor2im(gt_image)
#     hint_np = tensor2im(hint_image)
#
#     gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_LAB2BGR)
#     hint_bgr = cv2.cvtColor(hint_np, cv2.COLOR_LAB2BGR)
#
#     image_show(gt_bgr)
#     image_show(hint_bgr)
#
#     input()


'''
 Network Training
'''
print('train dataset length: ', len(train_dataloader))

# Network setting
net = ColorizationModel().cuda()

# Learning  pre-trained model
#net.load_state_dict(state_dict['model_weight'], strict=True)

# Loss and Optimizer setting
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

object_epoch = 50

save_path = './ColorizationNetwork'
os.makedirs(save_path, exist_ok=True)
output_path = os.path.join(save_path, 'basic_model.tar')


def train_1epoch(net, dataloader, object_epoch):
    total_loss = 0
    iteration = 0
    psnr_total = 0
    ssim_total = 0

    net.train()

    for sample in tqdm.auto.tqdm(dataloader):
        img_l = sample['l']
        img_ab = sample['ab']
        img_hint = sample['hint']

        img_l = img_l.float().cuda()
        img_ab = img_ab.float().cuda()
        img_hint = img_hint.float().cuda()

        optimizer.zero_grad()
        output = net(img_l, img_hint)

        loss = criterion(output, img_ab)
        loss.backward()
        optimizer.step()

        gt_image = torch.cat((img_l, img_ab), dim=1)
        output_image = torch.cat((img_l, output), dim=1)

        # PSNR
        gt_np = tensor2npy(gt_image)
        output_np = tensor2npy(output_image)
        psnr_mse = np.mean((output_np - gt_np) ** 2)

        psnr = PSNR(psnr_mse)
        psnr_total += psnr

        # SSIM
        gt_np_ssim = tensor2im(gt_image)
        output_np_ssim = tensor2im(output_image)

        gt_bgr = cv2.cvtColor(gt_np_ssim, cv2.COLOR_LAB2BGR)
        output_bgr = cv2.cvtColor(output_np_ssim, cv2.COLOR_LAB2BGR)

        ssim_score = ssim(gt_bgr, output_bgr)
        ssim_total += ssim_score

        total_loss += loss.detach()
        iteration += 1

        # Image presentation

        if object_epoch == 99 and iteration == 179:
            gt_image = torch.cat((img_l, img_ab), dim=1)
            hint_image = torch.cat((img_l, img_hint), dim=1)
            output_image = torch.cat((img_l, output), dim=1)

            gt_np = tensor2im(gt_image)
            hint_np = tensor2im(hint_image)
            output_np = tensor2im(output_image)

            gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_LAB2BGR)
            hint_bgr = cv2.cvtColor(hint_np, cv2.COLOR_LAB2BGR)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_LAB2BGR)

            image_show(gt_bgr)
            image_show(hint_bgr)
            image_show(output_bgr)

    total_loss /= iteration
    psnr_total /= iteration
    ssim_total /= iteration

    return total_loss, psnr_total, ssim_total


def validation_1epoch(net, dataloader, object_epoch):
    total_loss = 0
    iteration = 0
    psnr_total = 0
    ssim_total = 0

    net.eval()

    for sample in tqdm.auto.tqdm(dataloader):
        img_l = sample['l']
        img_ab = sample['ab']
        img_hint = sample['hint']

        img_l = img_l.float().cuda()
        img_ab = img_ab.float().cuda()
        img_hint = img_hint.float().cuda()

        output = net(img_l, img_hint)

        loss = criterion(output, img_ab)

        gt_image = torch.cat((img_l, img_ab), dim=1)
        output_image = torch.cat((img_l, output), dim=1)

        # PSNR

        gt_np = tensor2npy(gt_image)
        output_np = tensor2npy(output_image)
        psnr_mse = np.mean((output_np - gt_np) ** 2)

        psnr = PSNR(psnr_mse)
        psnr_total += psnr

        # SSIM
        gt_np_ssim = tensor2im(gt_image)
        output_np_ssim = tensor2im(output_image)

        gt_bgr = cv2.cvtColor(gt_np_ssim, cv2.COLOR_LAB2BGR)
        output_bgr = cv2.cvtColor(output_np_ssim, cv2.COLOR_LAB2BGR)

        ssim_score = ssim(gt_bgr, output_bgr)
        ssim_total += ssim_score

        total_loss += loss.detach()
        iteration += 1

        # Image presentation

        if object_epoch == 99 and iteration == 179:
            gt_image = torch.cat((img_l, img_ab), dim=1)
            hint_image = torch.cat((img_l, img_hint), dim=1)
            output_image = torch.cat((img_l, output), dim=1)

            gt_np = tensor2im(gt_image)
            hint_np = tensor2im(hint_image)
            output_np = tensor2im(output_image)

            gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_LAB2BGR)
            hint_bgr = cv2.cvtColor(hint_np, cv2.COLOR_LAB2BGR)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_LAB2BGR)

            image_show(gt_bgr)
            image_show(hint_bgr)
            image_show(output_bgr)

    total_loss /= iteration
    psnr_total /= iteration
    ssim_total /= iteration

    return total_loss, psnr_total, ssim_total


for epoch in range(object_epoch):
    train_loss, train_psnr, train_ssim = train_1epoch(net, train_dataloader, epoch)
    print('[TRAINING] Epoch {} loss: {}, psnr: {}, ssim: {}'.format(epoch, train_loss, train_psnr, train_ssim.item()))

    with torch.no_grad():
        val_loss, val_psnr, val_ssim = validation_1epoch(net, val_dataloader, epoch)
        print('[VALIDATION] Epoch {} loss: {}, psnr: {}, ssim: {}'.format(epoch, val_loss, val_psnr, val_ssim.item()))

    torch.save({
        'memo': 'Colorization Model',
        'loss': val_loss,
        'PSNR': val_psnr,
        'epoch': epoch,
        'model_weight': net.state_dict()
    }, save_path + "/validation_model.tar")
