import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import *


class ScoreNet(nn.Module):

    def __init__(self, args):
        super(ScoreNet, self).__init__()
        self.args = args

        self.in_channels = args.in_channels
        self.ngf = args.ngf  # 64
        self.num_classes = args.num_classes  # 10
        self.act = nn.ELU()

        self.begin_conv = nn.Conv2d(self.in_channels, self.ngf, kernel_size=3, stride=1, padding=1)
        self.norm = ConditionalInstanceNorm2dPlus(self.ngf, self.num_classes)

        self.end_conv = nn.Conv2d(self.ngf, self.in_channels, kernel_size=3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=self.act),
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=self.act)]
        )

        self.res2 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=self.act),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=self.act)]
        )

        self.res3 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=self.act,
                                     dilation=2),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=self.act,
                                     dilation=2)]
        )

        if args.image_size == 28:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=self.act,
                                         adjust_padding=True, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=self.act,
                                         dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=self.act,
                                         adjust_padding=False, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=self.act,
                                         dilation=4)]
            )

        self.refine1 = CondRefineBlock([2 * self.ngf], 2 * self.ngf, self.num_classes, act=self.act, start=True)
        self.refine2 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.num_classes, act=self.act)
        self.refine3 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, self.num_classes, act=self.act)
        self.refine4 = CondRefineBlock([self.ngf, self.ngf], self.ngf, self.num_classes, act=self.act, end=True)

    def _compute_cond_module(self, module, x, y):
        for m in module:
            x = m(x, y)
        return x

    def forward(self, x, y):
        # todo x? y?
        if not self.args.logit_transform:
            x = 2 * x - 1.

        output = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, output, y)
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        layer4 = self._compute_cond_module(self.res4, layer3, y)

        ref1 = self.refine1([layer4], y, layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], y, layer2.shape[2:])
        output = self.refine4([layer1, ref3], y, layer1.shape[2:])

        output = self.norm(output, y)
        output = self.act(output)
        output = self.end_conv(output)
        return output












