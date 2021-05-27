import torch
import os
import tqdm
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToPILImage
import tqdm
from PIL import Image
import numpy as np

from dataloader import ColorHintDataset
from model import ColorizationModel

save_path = './ColorizationNetwork'
model_path = os.path.join(save_path, 'validation_model.tar')
state_dict = torch.load(model_path)

print(state_dict['memo'])
print(state_dict.keys())
print(state_dict['PSNR'])

# Change to your data root directory
root_path = "./"

# Depend on runtime setting
use_cuda = True

test_dataset = ColorHintDataset(root_path, 128)
test_dataset.set_mode("testing")

test_dataloader = data.DataLoader(test_dataset)


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


net = ColorizationModel().cuda()

net.load_state_dict(state_dict['model_weight'], strict=True)


def test_1epoch(net, dataloader):
    net.eval()

    for sample in tqdm.auto.tqdm(dataloader):
        img_l = sample['l']
        img_hint = sample['hint']
        file_name = sample['file_name']

        img_l = img_l.float().cuda()
        img_hint = img_hint.float().cuda()

        output = net(img_l, img_hint)

        output_image = torch.cat((img_l, output), dim=1)

        output_np = tensor2im(output_image)

        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_LAB2BGR)

        # Image presentation
        #image_show(output_bgr)

        cv2.imwrite('./result/' + file_name[0], output_bgr)

    return 0


res = test_1epoch(net, test_dataloader)
