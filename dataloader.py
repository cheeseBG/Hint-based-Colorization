import cv2
import torch.utils.data as data
import random
import numpy as np
import os
from torch.autograd import Variable
from torchvision import transforms


'''
 Original transform
'''
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

    def __call__(self, img):
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
            hint_image = image * self.img_to_mask(image)

            l, _ = self.bgr_to_lab(image)
            _, ab_hint = self.bgr_to_lab(hint_image)

            return self.transform(l), self.transform(ab_hint)

        else:
            return NotImplementedError


'''
  Augmentation1  
'''
class ColorHintTransform2(object):
    def __init__(self, size=256, mode="training"):
        super(ColorHintTransform2, self).__init__()
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor()])

    def bgr_to_lab(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, ab = lab[:, :, 0], lab[:, :, 1:]
        return l, ab

    def hint_mask(self, bgr, threshold=[0.95, 0.97, 0.99]):
        h, w, c = bgr.shape
        mask_threshold = random.choice(threshold)
        mask = np.random.random([h, w, 1]) > threshold
        return mask

    def img_to_mask(self, mask_img):
        mask = mask_img[:, :, 0, np.newaxis] >= 255
        return mask

    def __call__(self, img):
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
            hint_image = image * self.img_to_mask(image)

            l, _ = self.bgr_to_lab(image)
            _, ab_hint = self.bgr_to_lab(hint_image)

            return self.transform(l), self.transform(ab_hint)

        else:
            return NotImplementedError


'''
  Augmentation2  
'''
class ColorHintTransform3(object):
    def __init__(self, size=256, mode="training"):
        super(ColorHintTransform3, self).__init__()
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor()])

    def bgr_to_lab(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, ab = lab[:, :, 0], lab[:, :, 1:]
        return l, ab

    def hint_mask(self, bgr, threshold=[0.95, 0.97, 0.99]):
        h, w, c = bgr.shape
        mask_threshold = random.choice(threshold)
        mask = np.random.random([h, w, 1]) > threshold
        return mask

    def img_to_mask(self, mask_img):
        mask = mask_img[:, :, 0, np.newaxis] >= 255
        return mask

    def __call__(self, img):
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
            hint_image = image * self.img_to_mask(image)

            l, _ = self.bgr_to_lab(image)
            _, ab_hint = self.bgr_to_lab(hint_image)

            return self.transform(l), self.transform(ab_hint)

        else:
            return NotImplementedError


'''
  Augmentation3  
'''
class ColorHintTransform4(object):
    def __init__(self, size=256, mode="training"):
        super(ColorHintTransform4, self).__init__()
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor()])

    def bgr_to_lab(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, ab = lab[:, :, 0], lab[:, :, 1:]
        return l, ab

    def hint_mask(self, bgr, threshold=[0.95, 0.97, 0.99]):
        h, w, c = bgr.shape
        mask_threshold = random.choice(threshold)
        mask = np.random.random([h, w, 1]) > threshold
        return mask

    def img_to_mask(self, mask_img):
        mask = mask_img[:, :, 0, np.newaxis] >= 255
        return mask

    def __call__(self, img):
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
            hint_image = image * self.img_to_mask(image)

            l, _ = self.bgr_to_lab(image)
            _, ab_hint = self.bgr_to_lab(hint_image)

            return self.transform(l), self.transform(ab_hint)

        else:
            return NotImplementedError


'''
    original
'''
# class ColorHintDataset(data.Dataset):
#     def __init__(self, root_path, size):
#         super(ColorHintDataset, self).__init__()
#
#         self.root_path = root_path
#         self.size = size
#         self.transforms = None
#         self.examples = None
#         self.hint = None
#         self.mask = None
#
#     def set_mode(self, mode):
#         self.mode = mode
#         self.transforms = ColorHintTransform(self.size, mode)
#
#         if mode == "training":
#             train_dir = os.path.join(self.root_path, "train")
#
#             # File name
#             self.examples = [os.path.join(self.root_path, "train", dirs) for dirs in os.listdir(train_dir)]
#
#         elif mode == "validation":
#             val_dir = os.path.join(self.root_path, "validation")
#
#             # File name
#             self.examples = [os.path.join(self.root_path, "validation", dirs) for dirs in os.listdir(val_dir)]
#
#         elif mode == "testing":
#             hint_dir = os.path.join(self.root_path, "hint")
#             mask_dir = os.path.join(self.root_path, "mask")
#             self.hint = [os.path.join(self.root_path, "hint", dirs) for dirs in os.listdir(hint_dir)]
#             self.mask = [os.path.join(self.root_path, "mask", dirs) for dirs in os.listdir(mask_dir)]
#
#         else:
#             raise NotImplementedError
#
#     def __len__(self):
#         if self.mode != "testing":
#             return len(self.examples)
#         else:
#             return len(self.hint)
#
#     def __getitem__(self, idx):
#         if self.mode == "testing":
#             hint_file_name = self.hint[idx]
#             mask_file_name = self.mask[idx]
#             hint_img = cv2.imread(hint_file_name)
#             mask_img = cv2.imread(mask_file_name)
#
#             input_l, input_hint = self.transforms(hint_img, mask_img)
#             sample = {"l": input_l, "hint": input_hint,
#                       "file_name": "image_%06d.png" % int(os.path.basename(hint_file_name).split('.')[0])}
#         else:
#             file_name = self.examples[idx]
#             img = cv2.imread(file_name)
#             l, ab, hint = self.transforms(img)
#             sample = {"l": l, "ab": ab, "hint": hint}
#         return sample


'''
    Augmentation
'''
class ColorHintDataset(data.Dataset):
    def __init__(self, root_path, size):
        super(ColorHintDataset, self).__init__()

        self.root_path = root_path
        self.size = size
        self.transforms = None
        self.transforms2 = None
        self.transforms3 = None
        self.transforms4 = None
        self.examples = None
        self.hint = None
        self.mask = None

    def set_mode(self, mode):
        self.mode = mode
        self.transforms = ColorHintTransform(self.size, mode)
        self.transforms2 = ColorHintTransform2(self.size, mode)
        self.transforms3 = ColorHintTransform3(self.size, mode)
        self.transforms4 = ColorHintTransform4(self.size, mode)

        if mode == "training":
            train_dir = os.path.join(self.root_path, "train")

            # File name
            x1 = [os.path.join(self.root_path, "train", dirs) for dirs in os.listdir(train_dir)]
            x2 = [os.path.join(self.root_path, "train", dirs) for dirs in os.listdir(train_dir)]
            x3 = [os.path.join(self.root_path, "train", dirs) for dirs in os.listdir(train_dir)]
            x4 = [os.path.join(self.root_path, "train", dirs) for dirs in os.listdir(train_dir)]

            self.examples = x1 + x2 + x3 + x4
            print(len(self.examples))

        elif mode == "validation":
            val_dir = os.path.join(self.root_path, "validation")

            # File name

            x1 = [os.path.join(self.root_path, "validation", dirs) for dirs in os.listdir(val_dir)]
            x2 = [os.path.join(self.root_path, "validation", dirs) for dirs in os.listdir(val_dir)]
            x3 = [os.path.join(self.root_path, "validation", dirs) for dirs in os.listdir(val_dir)]
            x4 = [os.path.join(self.root_path, "validation", dirs) for dirs in os.listdir(val_dir)]

            self.examples = x1 + x2 + x3 + x4

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
        file_name = self.examples[idx]
        img = cv2.imread(file_name)

        if self.mode == "testing":
            hint_file_name = self.hint[idx]
            mask_file_name = self.mask[idx]
            hint_img = cv2.imread(hint_file_name)
            mask_img = cv2.imread(mask_file_name)

            input_l, input_hint = self.transforms(hint_img, mask_img)
            sample = {"l": input_l, "hint": input_hint,
                      "file_name": "image_%06d.png" % int(os.path.basename(hint_file_name).split('.')[0])}

        elif idx < 4500 and (self.mode == "training" or self.mode == "validation"):
            l, ab, hint = self.transforms(img)

            sample = {"l": l, "ab": ab, "hint": hint}

        elif 4500 <= idx < 9000 and (self.mode == "training" or self.mode == "validation"):
            l, ab, hint = self.transforms2(img)

            sample = {"l": l, "ab": ab, "hint": hint}

        elif 9000 <= idx < 13500 and (self.mode == "training" or self.mode == "validation"):
            l, ab, hint = self.transforms3(img)

            sample = {"l": l, "ab": ab, "hint": hint}

        elif 13500 <= idx < 18000 and (self.mode == "training" or self.mode == "validation"):
            l, ab, hint = self.transforms4(img)

            sample = {"l": l, "ab": ab, "hint": hint}

        return sample
