import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


class ConditionalResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, resample=None, act=nn.ELU(),
                 adjust_padding=False, dilation=None):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = ConditionalInstanceNorm2dPlus(input_dim, num_classes)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
            else:
                self.conv1 = nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1)
                self.normalize2 = ConditionalInstanceNorm2dPlus(input_dim, num_classes)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = ConditionalInstanceNorm2dPlus(output_dim, num_classes)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
                self.normalize2 = ConditionalInstanceNorm2dPlus(output_dim, num_classes)
                self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = ConditionalInstanceNorm2dPlus(input_dim, num_classes)

    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class CondRefineBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, act=nn.ELU(), start=False, end=False):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(CondRCUBlock(in_planes[i], 2, 2, num_classes, act))

        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, act)

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes)

        self.crp = CondCRPBlock(features, 2, num_classes, act)

    def forward(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]
        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h


class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        else:
            self.conv = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)),
                                      nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases))

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        # TODO
        return output


class CondRCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, num_classes, act=nn.ELU()):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_norm'.format(i + 1, j + 1), ConditionalInstanceNorm2dPlus(features, num_classes, bias=True))
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1),
                        nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)
            x += residual
        return x


class CondMSFBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes):
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(nn.Conv2d(in_planes[i], features, kernel_size=3, stride=1, padding=1, bias=True))
            self.norms.append(ConditionalInstanceNorm2dPlus(in_planes[i], num_classes, bias=True))

    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class CondCRPBlock(nn.Module):
    def __init__(self, features, n_stages, num_classes, act=nn.ELU()):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_stages):
            self.norms.append(ConditionalInstanceNorm2dPlus(features, num_classes, bias=True))
            self.convs.append(nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False))
        self.n_stages = n_stages
        self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.maxpool(path)
            path = self.convs[i](path)
            x = path + x
        return x


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)



