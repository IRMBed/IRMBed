import torch
import pdb
from collections import defaultdict, OrderedDict
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import model_zoo
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = [ 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetSepFcOFC(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetSepFcOFC, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_classes = 1
        self.num_classes = 1
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, num_classes)
        self.class_classifier_sep1 = nn.Linear(512 * block.expansion, num_classes)
        self.class_classifier_sep2 = nn.Linear(512 * block.expansion, num_classes)
        self.class_classifier_sep3 = nn.Linear(512 * block.expansion, num_classes)
        self.class_classifier_sep4 = nn.Linear(512 * block.expansion, num_classes)
        self.sep=False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def set_sep(self, sep, key=None):
        assert isinstance(sep, bool)
        self.sep = sep

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def encoder(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    # def forward(self, x):
    #     return self._forward_impl(x)

    # def encoder(self, x):
    #     pass

    def init_sep_by_share(self):
        self.class_classifier_sep1.weight.data = self.class_classifier.weight.data.clone()
        self.class_classifier_sep2.weight.data = self.class_classifier.weight.data.clone()
        self.class_classifier_sep3.weight.data = self.class_classifier.weight.data.clone()
        self.class_classifier_sep4.weight.data = self.class_classifier.weight.data.clone()
        self.class_classifier_sep1.bias.data = self.class_classifier.bias.data.clone()
        self.class_classifier_sep2.bias.data = self.class_classifier.bias.data.clone()
        self.class_classifier_sep3.bias.data = self.class_classifier.bias.data.clone()
        self.class_classifier_sep4.bias.data = self.class_classifier.bias.data.clone()


    def forward(self, x, group_idx=None, only_backward_fc=False):
        #if self.training or self.norm_layer != batchnorm_dson.BatchNorm2d:

        if only_backward_fc:
            # assert self.sep
            assert group_idx is not None
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        self.fp = x

        if self.sep is False:
            return self.class_classifier(x)
        else:
            assert group_idx is not None
            output = torch.zeros([x.shape[0], self.num_classes]).cuda()
            output[group_idx == 0]= self.class_classifier_sep1(x[group_idx == 0])
            output[group_idx == 1]= self.class_classifier_sep2(x[group_idx == 1])
            output[group_idx == 2]= self.class_classifier_sep3(x[group_idx == 2])
            output[group_idx == 3]= self.class_classifier_sep4(x[group_idx == 3])
            return output

    def share_param_id(self):
        share_params = [
            p[1] for p
            in self.named_parameters()
            if 'classifier' in p[0] and
                'sep' not in p[0]]
        share_param_id = [id(i) for i in share_params]
        return share_param_id

    def sep_param_id(self):
        sep_params = [
            p[1] for p
            in self.named_parameters()
            if 'classifier' in p[0] and
                'sep' in p[0]]
        sep_param_id = [id(i) for i in sep_params]
        return sep_param_id

    def rep_param_id(self):
        rep_param_id = [
            id(p) for p in self.parameters()
            if id(p) not in  self.sep_param_id()
                and id(p) not in self.share_param_id()]
        return rep_param_id

    def get_optimizer_schedule(self, args):
        if args.irm_penalty_weight > 0:
            if args.opt == "Adam":
                opt_fun = optim.Adam
                optimizer_rep = opt_fun(
                    filter(lambda p:id(p) in self.rep_param_id(), self.parameters()),
                    lr=args.lr)
                optimizer_share = opt_fun(
                    filter(lambda p:id(p) in self.share_param_id(), self.parameters()),
                    lr=args.lr* args.penalty_wlr)
                optimizer_sep = optim.SGD(
                    filter(lambda p:id(p) in self.sep_param_id(), self.parameters()),
                    lr=args.lr* args.penalty_welr)
            elif args.opt == "SGD":
                opt_fun = optim.SGD
                optimizer_rep = opt_fun(
                    filter(lambda p:id(p) in self.rep_param_id(), self.parameters()),
                    momentum=0.9,
                    lr=args.lr)
                optimizer_share = opt_fun(
                    filter(lambda p:id(p) in self.share_param_id(), self.parameters()),
                    momentum=args.w_momentum,
                    lr=args.lr * args.penalty_wlr)
                optimizer_sep = opt_fun(
                    filter(lambda p:id(p) in self.sep_param_id(), self.parameters()),
                    momentum=args.w_momentum,
                    lr=args.lr * args.penalty_welr)
            else:
                raise Exception
            if args.lr_schedule_type == "step":
                print("step_gamma=%s" % args.step_gamma)
                scheduler_rep = lr_scheduler.StepLR(optimizer_rep, step_size=int(args.n_epochs/3.), gamma=args.step_gamma)
                scheduler_sep = lr_scheduler.StepLR(optimizer_sep, step_size=int(args.n_epochs), gamma=args.step_gamma)
                scheduler_share = lr_scheduler.StepLR(optimizer_share, step_size=int(args.n_epochs/3.), gamma=args.step_gamma)

            return [optimizer_rep, optimizer_share, optimizer_sep], [scheduler_rep, scheduler_share, scheduler_sep]
        else:
            if args.opt == "Adam":
                opt_fun = optim.Adam
                optimizer= opt_fun(
                    filter(lambda p:id(p) in self.rep_param_id(), self.parameters()),
                    lr=args.lr)
            elif args.opt == "SGD":
                opt_fun = optim.SGD
                optimizer= opt_fun(
                    filter(lambda p:id(p) in self.rep_param_id(), self.parameters()),
                    momentum=0.9,
                    lr=args.lr)
            else:
                raise Exception
            scheduler= lr_scheduler.StepLR(
                optimizer,
                step_size=int(args.n_epochs/3.),
                gamma=args.step_gamma)
            return [optimizer], [scheduler]

class ResNetSep2FcInvRat(ResNetSepFcInvRat):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        nn.Module.__init__(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_classes = 1
        self.num_classes = 1
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.class_classifier = nn.Sequential(
                                nn.Linear(512 * block.expansion, 512 * block.expansion),
                                nn.ReLU(inplace=True),
                                nn.Linear(512 * block.expansion, num_classes))
        # TODO fix the hard code of embeding dim
        # 4 denotes num of environment, 128 denotes the embedding dimension
        self.env_embedding = torch.nn.Embedding(4, 128)
        hidden_dim = 512 * block.expansion
        self.class_classifier_sep = nn.Sequential(
                                nn.Linear(512 * block.expansion+128, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, num_classes))
        self.sep=False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


class ResNetSepFcGame(ResNetSepFcOFC):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetSepFcGame, self).__init__(block, layers, num_classes, zero_init_residual,groups, width_per_group, replace_stride_with_dilation,norm_layer)

    def forward(self, x, group_idx=None, only_backward_fc=False):
        #if self.training or self.norm_layer != batchnorm_dson.BatchNorm2d:

        if only_backward_fc:
            # assert self.sep
            assert group_idx is not None
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        self.fp = x
        if self.args.env_nums==2:
            return 0.5 * (self.class_classifier_sep1(x) + self.class_classifier_sep2(x))
        elif self.args.env_nums==4:
            return 0.25 * (self.class_classifier_sep1(x) + self.class_classifier_sep2(x) + self.class_classifier_sep3(x) + self.class_classifier_sep4(x))
        else:
            raise RuntimeError

    def rep_param_id(self):
        rep_param_id = [
            id(p) for p in self.parameters()
            if id(p) not in  self.sep_param_id()
                and id(p) not in self.share_param_id()]
        return rep_param_id


    def sep1_param_id(self):
        sep_params = [
            p[1] for p
            in self.named_parameters()
            if 'classifier_sep1' in p[0]]
        sep_param_id = [id(i) for i in sep_params]
        return sep_param_id

    def sep2_param_id(self):
        sep_params = [
            p[1] for p
            in self.named_parameters()
            if 'classifier_sep2' in p[0]]
        sep_param_id = [id(i) for i in sep_params]
        return sep_param_id


    def sep3_param_id(self):
        sep_params = [
            p[1] for p
            in self.named_parameters()
            if 'classifier_sep3' in p[0]]
        sep_param_id = [id(i) for i in sep_params]
        return sep_param_id

    def sep4_param_id(self):
        sep_params = [
            p[1] for p
            in self.named_parameters()
            if 'classifier_sep4' in p[0]]
        sep_param_id = [id(i) for i in sep_params]
        return sep_param_id


    def get_optimizer_schedule(self, args):
        self.args = args
        assert args.irm_penalty_weight > 0
        optimizer_rep = optim.SGD(
            filter(lambda p:id(p) in self.rep_param_id(), self.parameters()),
            momentum=0.9,
            lr=args.lr,
        )
        optimizer_sep1 = optim.SGD(
            filter(lambda p:id(p) in self.sep1_param_id(), self.parameters()),
            momentum=0.9,
            lr=args.lr * args.penalty_lrml
            )
        optimizer_sep2 = optim.SGD(
            filter(lambda p:id(p) in self.sep2_param_id(), self.parameters()),
            momentum=0.9,
            lr=args.lr * args.penalty_lrml
            )
        optimizer_sep3 = optim.SGD(
            filter(lambda p:id(p) in self.sep3_param_id(), self.parameters()),
            momentum=0.9,
            lr=args.lr * args.penalty_lrml
            )
        optimizer_sep4 = optim.SGD(
            filter(lambda p:id(p) in self.sep4_param_id(), self.parameters()),
            momentum=0.9,
            lr=args.lr * args.penalty_lrml
            )
        if args.lr_schedule_type == "step":
            scheduler_rep = lr_scheduler.StepLR(optimizer_rep, step_size=int(args.n_epochs/3.), gamma=0.2)
            scheduler_sep1 = lr_scheduler.StepLR(optimizer_sep1, step_size=int(args.n_epochs/3.), gamma=0.2)
            scheduler_sep2 = lr_scheduler.StepLR(optimizer_sep2, step_size=int(args.n_epochs/3.), gamma=0.2)
            scheduler_sep3 = lr_scheduler.StepLR(optimizer_sep3, step_size=int(args.n_epochs/3.), gamma=0.2)
            scheduler_sep4 = lr_scheduler.StepLR(optimizer_sep4, step_size=int(args.n_epochs/3.), gamma=0.2)
        else:
            raise RuntimeError

        return [optimizer_rep, optimizer_sep1, optimizer_sep2, optimizer_sep3, optimizer_sep4], [scheduler_rep, scheduler_sep1, scheduler_sep2, scheduler_sep3, scheduler_sep4]


def _resnet_sepfc_ofc(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetSepFcOFC(block, layers, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_state_dict)
        model.load_state_dict(model_dict)
    return model

def _resnet_sepfc_game(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetSepFcGame(block, layers, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_state_dict)
        model.load_state_dict(model_dict)
    return model


def resnet18_sepfc_ofc(pretrained=False, progress=True, **kwargs):
    return _resnet_sepfc_ofc('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def _resnet_sep2fc_invrat(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetSep2FcInvRat(block, layers, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_state_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50_sepfc_ofc(pretrained=False, progress=True, **kwargs):
    return _resnet_sepfc_ofc(
        'resnet50', Bottleneck,
        [3, 4, 6, 3], pretrained, progress,
        **kwargs)

def resnet18_sep2fc_invrat(pretrained=False, progress=True, **kwargs):
    return _resnet_sep2fc_invrat('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet50_sep2fc_invrat(pretrained=False, progress=True, **kwargs):
    return _resnet_sep2fc_invrat(
        'resnet50', Bottleneck,
        [3, 4, 6, 3], pretrained, progress,
        **kwargs)


def resnet50_sepfc_game(pretrained=False, progress=True, **kwargs):
    return _resnet_sepfc_game(
        'resnet50', Bottleneck,
        [3, 4, 6, 3], pretrained, progress,
        **kwargs)

def resnet18_sepfc_game(pretrained=False, progress=True, **kwargs):
    return _resnet_sepfc_game('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

