'''
 Network Model
'''
import torch
import torch.nn as nn


class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()
        use_bias = True
        norm_layer = nn.BatchNorm2d

        # Conv1
        model1 = [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model1 += [nn.ReLU(True), ]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]

        # Conv2
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model2 += [nn.ReLU(True), ]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]

        # Conv3
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]

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
        model5_1 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model5_1 += [nn.ReLU(True), ]
        model5_1 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model5_1 += [nn.ReLU(True), ]
        model5_1 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model5_1 += [nn.ReLU(True), ]
        model5_1 += [norm_layer(512), ]

        # Conv5-2
        model5_2 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model5_2 += [nn.ReLU(True), ]
        model5_2 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model5_2 += [nn.ReLU(True), ]
        model5_2 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model5_2 += [nn.ReLU(True), ]
        model5_2 += [norm_layer(512), ]

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

        # Conv6-2
        model6_2 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model6_2 += [nn.ReLU(True), ]
        model6_2 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model6_2 += [nn.ReLU(True), ]
        model6_2 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model6_2 += [nn.ReLU(True), ]
        model6_2 += [norm_layer(512), ]


        # Conv7
        model7 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model7 += [nn.ReLU(True), ]
        model7 += [norm_layer(512), ]

        # Conv7-1
        model7_1 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7_1 += [nn.ReLU(True), ]
        model7_1 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7_1 += [nn.ReLU(True), ]
        model7_1 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7_1 += [nn.ReLU(True), ]
        model7_1 += [norm_layer(512), ]

        # Conv7-2
        model7_2 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7_2 += [nn.ReLU(True), ]
        model7_2 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7_2 += [nn.ReLU(True), ]
        model7_2 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7_2 += [nn.ReLU(True), ]
        model7_2 += [norm_layer(512), ]

        # Conv8
        model8add = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model8 = [nn.ReLU(True), ]
        model8 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [norm_layer(512), ]

        # Conv9
        model9up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model3short9 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model9 = [nn.ReLU(True), ]
        model9 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model9 += [nn.ReLU(True), ]
        model9 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model9 += [nn.ReLU(True), ]
        model9 += [norm_layer(256), ]

        # Conv10
        model10up = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model2short10 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model10 = [nn.ReLU(True), ]
        model10 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model10 += [nn.ReLU(True), ]
        model10 += [norm_layer(128), ]

        # Conv11
        model11up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model1short11 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model11 = [nn.ReLU(True), ]
        model11 += [nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias), ]
        model11 += [nn.LeakyReLU(negative_slope=.2), ]

        # classification output
        model_class = [nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), ]

        # regression output
        model_out = [nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), ]
        model_out += [nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)

        # Dilation
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)

        # Non-Dilation 1
        self.model5_1 = nn.Sequential(*model5_1)
        self.model6_1 = nn.Sequential(*model6_1)
        self.model7_1 = nn.Sequential(*model7_1)

        # Non-Dilation 2
        self.model5_2 = nn.Sequential(*model5_2)
        self.model6_2 = nn.Sequential(*model6_2)
        self.model7_2 = nn.Sequential(*model7_2)

        self.model8add = nn.Sequential(*model8add)
        self.model8 = nn.Sequential(*model8)

        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model11up = nn.Sequential(*model11up)
        self.model11 = nn.Sequential(*model11)
        self.model3short9 = nn.Sequential(*model3short9)
        self.model2short10 = nn.Sequential(*model2short10)
        self.model1short11 = nn.Sequential(*model1short11)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

    def forward(self, input_l, input_hint):
        conv1_2 = self.model1(torch.cat((input_l, input_hint), dim=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])

        # Dilation
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        # Original1
        conv5_1_3 = self.model5_1(conv4_3)
        conv6_1_3 = self.model6_1(conv5_1_3)
        conv7_1_3 = self.model7_1(conv6_1_3)

        # Original2
        conv5_2_3 = self.model5_2(conv4_3)
        conv6_2_3 = self.model6_2(conv5_2_3)
        conv7_2_3 = self.model7_2(conv6_2_3)

        conv8_add = self.model8add(conv7_3) + self.model8add(conv7_1_3) + self.model8add(conv7_2_3)
        conv8_3 = self.model8(conv8_add)

        conv9_up = self.model9up(conv8_3) + self.model3short9(conv3_3)
        conv9_3 = self.model9(conv9_up)

        conv10_up = self.model10up(conv9_3) + self.model2short10(conv2_2)
        conv10_3 = self.model10(conv10_up)
        conv11_up = self.model11up(conv10_3) + self.model1short11(conv1_2)
        conv11_2 = self.model11(conv11_up)
        out_reg = self.model_out(conv11_2)
        return out_reg
