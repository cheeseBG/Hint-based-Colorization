import torch
import os
import tqdm
save_path = './ColorizationNetwork'
model_path = os.path.join(save_path, 'validation_model.tar')
state_dict = torch.load(model_path)

print(state_dict['memo'])
print(state_dict.keys())
print(state_dict['PSNR'])

''' Data '''
import torch
from torch.autograd import Variable
from torchvision import transforms

import cv2
import random
import numpy as np

import torch.utils.data as data
import os
import cv2


class ColorHintTransform(object):
    def __init__(self, size=256, mode="training"):
        super(ColorHintTransform, self).__init__()
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])

    def bgr_to_lab(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, ab = lab[:, :, 0], lab[:, :, 1:]
        return l, ab

    def hint_mask(self, bgr, threshold=[0.95, 0.97, 0.99]):
        h, w, c = bgr.shape
        mask_threshold = random.choice(threshold)
        mask = np.random.random([h, w, 1]) > mask_threshold
        return mask

    def img_to_mask(self, mask_img):
        mask = mask_img[:, :, 0, np.newaxis] >= 255
        return mask

    def __call__(self, img, mask_img=None):
        threshold = [0.95, 0.97, 0.99]
        if (self.mode == "training") | (self.mode == "validation"):
            image = cv2.resize(img, (self.size, self.size))
            mask = self.hint_mask(image, threshold)

            hint_image = image * mask

            l, ab = self.bgr_to_lab(image)
            l_hint, ab_hint = self.bgr_to_lab(hint_image)

            return self.transform(l), self.transform(ab), self.transform(ab_hint)

        elif self.mode == "testing":
            image = cv2.resize(img, (self.size, self.size))
            hint_image = image * self.img_to_mask(mask_img)

            l, _ = self.bgr_to_lab(image)
            _, ab_hint = self.bgr_to_lab(hint_image)

            return self.transform(l), self.transform(ab_hint)

        else:
            return NotImplementedError

class ColorHintDataset(data.Dataset):
    def __init__(self, root_path, size):
        super(ColorHintDataset, self).__init__()

        self.root_path = root_path
        self.size = size
        self.transforms = None
        self.examples = None
        self.hint = None
        self.mask = None

    def set_mode(self, mode):
        self.mode = mode
        self.transforms = ColorHintTransform(self.size, mode)

        if mode == "training":
            train_dir = os.path.join(self.root_path, "train")

            # File name
            self.examples = [os.path.join(self.root_path, "train", dirs) for dirs in os.listdir(train_dir)]

        elif mode == "validation":
            val_dir = os.path.join(self.root_path, "validation")

            # File name
            self.examples = [os.path.join(self.root_path, "validation", dirs) for dirs in os.listdir(val_dir)]

        elif mode == "testing":
            hint_dir = os.path.join(self.root_path, "hint")
            mask_dir = os.path.join(self.root_path, "mask")
            self.hint = [os.path.join(self.root_path, "hint", dirs) for dirs in os.listdir(hint_dir)]
            self.mask = [os.path.join(self.root_path, "mask", dirs) for dirs in os.listdir(mask_dir)]

        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode != "testing":
            return len(self.examples)
        else:
            return len(self.hint)

    def __getitem__(self, idx):
        if self.mode == "testing":
            hint_file_name = self.hint[idx]
            mask_file_name = self.mask[idx]
            hint_img = cv2.imread(hint_file_name)
            mask_img = cv2.imread(mask_file_name)

            input_l, input_hint = self.transforms(hint_img, mask_img)
            sample = {"l": input_l, "hint": input_hint,
                      "file_name": "image_%06d.png" % int(os.path.basename(hint_file_name).split('.')[0])}
        else:
            file_name = self.examples[idx]
            img = cv2.imread(file_name)
            l, ab, hint = self.transforms(img)
            sample = {"l": l, "ab": ab, "hint": hint}
        return sample

# Change to your data root directory
root_path = "./"
# Depend on runtime setting
use_cuda = True

test_dataset = ColorHintDataset(root_path, 128)
test_dataset.set_mode("testing")

test_dataloader = data.DataLoader(test_dataset)

'''
    img
'''
import torch
import torch.utils.data as data
import os
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToPILImage
import tqdm
from PIL import Image
import numpy as np

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

'''
 Network Model
'''
import torch
import torch.nn as nn

class ColorizationModel(nn.Module):
    def __init__(self, dist=False):
        super(ColorizationModel, self).__init__()
        self.dist = dist
        use_bias = True
        norm_layer = nn.BatchNorm2d

        ''' Hint conv layer '''
        # Hint conv1
        h_model1 = [nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        h_model1 += [nn.ReLU(True), ]
        h_model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        h_model1 += [nn.ReLU(True), ]
        h_model1 += [norm_layer(64), ]

        # Hint conv2
        h_model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        h_model2 += [nn.ReLU(True), ]
        h_model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        h_model2 += [nn.ReLU(True), ]
        h_model2 += [norm_layer(128), ]

        # Hint conv3
        h_model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        h_model3 += [nn.ReLU(True), ]
        h_model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        h_model3 += [nn.ReLU(True), ]
        h_model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        h_model3 += [nn.ReLU(True), ]
        h_model3 += [norm_layer(256), ]

        # Hint conv4
        h_model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        h_model4 += [nn.ReLU(True), ]
        h_model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        h_model4 += [nn.ReLU(True), ]
        h_model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        h_model4 += [nn.ReLU(True), ]
        h_model4 += [norm_layer(512), ]

        ''' Main '''
        # Conv1
        model1 = [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model1 += [nn.ReLU(True), ]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]
        # add a subsampling operation

        # Conv2
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model2 += [nn.ReLU(True), ]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]
        # add a subsampling layer operation

        # Conv3
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]
        # add a subsampling layer operation

        # Conv4
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model4 += [nn.ReLU(True), ]
        model4 += [norm_layer(512), ]

        # Conv5
        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model5 += [nn.ReLU(True), ]
        model5 += [norm_layer(512), ]

        # Conv5-1
        model5_1_add = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model5_1 = [nn.ReLU(True), ]
        model5_1 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model5_1 += [nn.ReLU(True), ]
        model5_1 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model5_1 += [nn.ReLU(True), ]
        model5_1 += [norm_layer(512), ]

        # Conv6
        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model6 += [nn.ReLU(True), ]
        model6 += [norm_layer(512), ]

        # Conv6-1
        model6_1 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model6_1 += [nn.ReLU(True), ]
        model6_1 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model6_1 += [nn.ReLU(True), ]
        model6_1 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model6_1 += [nn.ReLU(True), ]
        model6_1 += [norm_layer(512), ]

        # Conv7
        model7add = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model7 = [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7 += [nn.ReLU(True), ]
        model7 += [norm_layer(512), ]

        # Conv7
        model8up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model3short8 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model8 = [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [norm_layer(256), ]

        # Conv9
        model9up = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model2short9 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # add the two feature maps above

        model9 = [nn.ReLU(True), ]
        model9 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model9 += [nn.ReLU(True), ]
        model9 += [norm_layer(128), ]

        # Conv10
        model10up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model1short10 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # add the two feature maps above

        model10 = [nn.ReLU(True), ]
        model10 += [nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias), ]
        model10 += [nn.LeakyReLU(negative_slope=.2), ]

        # classification output
        model_class = [nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), ] #WHY 529?

        # regression output
        model_out = [nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), ]
        model_out += [nn.Tanh()]

        ''' Hint '''
        self.h_model1 = nn.Sequential(*h_model1)
        self.h_model2 = nn.Sequential(*h_model2)
        self.h_model3 = nn.Sequential(*h_model3)
        self.h_model4 = nn.Sequential(*h_model4)

        ''' Main '''
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)

        # Dilation
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)

        # Original
        self.model5_1_add = nn.Sequential(*model5_1_add)
        self.model5_1 = nn.Sequential(*model5_1)
        self.model6_1 = nn.Sequential(*model6_1)

        self.model7add = nn.Sequential(*model7add)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='nearest'), ])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1), ])

    def forward(self, input_l, mask_B, maskcent=0):
        mask_B = mask_B - maskcent

        ''' Hint '''
        h_conv1_2 = self.h_model1(mask_B)
        h_conv2_2 = self.h_model2(h_conv1_2[:, :, ::2, ::2])
        h_conv3_3 = self.h_model3(h_conv2_2[:, :, ::2, ::2])
        h_conv4_3 = self.h_model4(h_conv3_3[:, :, ::2, ::2])

        ''' Main '''
        conv1_2 = self.model1(torch.cat((input_l, mask_B), dim=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])

        # Dilation
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)

        #Original
        conv5_1_3_add = self.model5_1_add(h_conv4_3) + self.model5_1_add(conv4_3)
        conv5_1_3 = self.model5_1(conv5_1_3_add)
        conv6_1_3 = self.model6_1(conv5_1_3)

        conv7_add = self.model7add(conv6_3) + self.model7add(conv6_1_3)
        conv7_3 = self.model7(conv7_add)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        if(self.dist):
            out_cl = self.upsample4(self.softmax(self.model_class(conv8_3) * .2))

            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)

            return (out_reg, out_cl)
        else:
            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
            return out_reg


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
