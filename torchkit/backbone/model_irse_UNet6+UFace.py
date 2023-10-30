# based on:
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/backbone/model_irse.py
from collections import namedtuple
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import PReLU
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Module
from torch.nn import Sigmoid
from torch import cat
from torchkit.backbone.common import initialize_weights, Flatten, SEModule


class BasicBlockIR(Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

class BasicBlockIR_s(Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR_s, self).__init__()
        if in_channel == depth:
            self.shortcut_layer1 = MaxPool2d(1, stride)
            self.shortcut_layer2 = MaxPool2d(1, stride)
            self.shortcut_layer3 = MaxPool2d(1, stride)
        else:
            self.shortcut_layer1 = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
            self.shortcut_layer2 = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
            self.shortcut_layer3 = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer1 = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))
        self.res_layer2 = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))
        self.res_layer3 = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        # shortcut1 = self.shortcut_layer(x0)
        # shortcut2 = self.shortcut_layer(x1)
        # shortcut = cat([shortcut1, shortcut2], dim=2)
        height = x.shape[2]
        x0 = x[:, :, :int(3 * height / 7), :]
        x1 = x[:, :, int(3 * height / 7):int(4 * height / 7), :]
        x2 = x[:, :, int(4 * height / 7):, :]
        shortcut1 = self.shortcut_layer1(x0)
        shortcut2 = self.shortcut_layer2(x1)
        shortcut3 = self.shortcut_layer3(x2)
        # print(shortcut1.shape,shortcut2.shape,shortcut3.shape)
        shortcut = cat([shortcut1,shortcut2,shortcut3],dim=2)
        res0 = self.res_layer1(x0)
        res1 = self.res_layer2(x1)
        res2 = self.res_layer3(x2)
        res = cat([res0,res1,res2],dim=2)
        # print(shortcut.shape,shortcut3.shape)
        return  res + shortcut

class BasicBlockIR_hc(Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR_hc, self).__init__()
        spl = 3
        spl_b = 4-3
        if in_channel == depth:
            self.shortcut_layer1 = MaxPool2d(1, stride)
            self.shortcut_layer2 = MaxPool2d(1, stride)
            self.shortcut_layer3 = MaxPool2d(1, stride)
        else:
            self.shortcut_layer1 = Sequential(
                Conv2d(spl*in_channel//4, spl*depth//4, (1, 1), stride, bias=False),
                BatchNorm2d(spl*depth//4))
            self.shortcut_layer2 = Sequential(
                Conv2d(spl*in_channel//4, spl*depth//4, (1, 1), stride, bias=False),
                BatchNorm2d(spl*depth//4))
            self.shortcut_layer3 = Sequential(
                Conv2d(spl_b*in_channel//4, spl_b*depth//4, (1, 1), stride, bias=False),
                BatchNorm2d(spl_b*depth//4))
     
        self.res_layer1 = Sequential(
            BatchNorm2d(spl*in_channel//4),
            Conv2d(spl*in_channel//4, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, spl*depth//4, (3, 3), stride, 1, bias=False),
            BatchNorm2d(spl*depth//4))
        self.res_layer2 = Sequential(
            BatchNorm2d(spl*in_channel//4),
            Conv2d(spl*in_channel//4, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, spl*depth//4, (3, 3), stride, 1, bias=False),
            BatchNorm2d(spl*depth//4))
        self.res_layer3 = Sequential(
            BatchNorm2d(spl_b*in_channel//4),
            Conv2d(spl_b*in_channel//4, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, spl_b*depth//4, (3, 3), stride, 1, bias=False),
            BatchNorm2d(spl_b*depth//4))

    def forward(self, x):
        # shortcut1 = self.shortcut_layer(x0)
        # shortcut2 = self.shortcut_layer(x1)
        # shortcut = cat([shortcut1, shortcut2], dim=2)
        height = x.shape[2]
        ch = x.shape[1]
        xc0 = x[:, :int(ch / 4), :, :]
        xc1 = x[:, int(ch/4):,:, :]
        x0 = xc1[:, :, :int(3 * height / 7), :]
        x1 = xc1[:, :, int(3 * height / 7):, :]
        shortcut1 = self.shortcut_layer1(x0)
        shortcut2 = self.shortcut_layer2(x1)
        shortcut3 = self.shortcut_layer3(xc0)
        shortcut = cat([shortcut1,shortcut2],dim=2)
        res_c0 = self.res_layer3(xc0) + shortcut3
        res0 = self.res_layer1(x0)
        res1 = self.res_layer2(x1)
        res_c1 = cat([res0,res1],dim=2) + shortcut
        res = cat([res_c0,res_c1],dim=1)
        return res


class BottleneckIR(Module):
    """ BasicBlock with bottleneck for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] +\
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        """ Args:
            input_size: input_size of backbone
            num_layers: num_layers of backbone
            mode: support ir or irse
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], \
            "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], \
            "mode should be ir or ir_se"
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == 'ir':
                unit_module = BasicBlockIR
            elif mode == 'ir_se':
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == 'ir':
                unit_module = BottleneckIR
            elif mode == 'ir_se':
                unit_module = BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer1 = Sequential(BatchNorm2d(output_channel//4),
                                           Dropout(0.4), Flatten(),
                                           Linear(128 * 7 * 7, 128),
                                           BatchNorm1d(128, affine=False))
            self.output_layer2 = Sequential(BatchNorm2d(3*output_channel//4),
                                           Dropout(0.4), Flatten(),
                                           Linear(384 * 7 * 3, 256),
                                           BatchNorm1d(256, affine=False))
            self.output_layer3 = Sequential(BatchNorm2d(3*output_channel//4),
                                           Dropout(0.4), Flatten(),
                                           Linear(384 * 7 * 4, 128),
                                           BatchNorm1d(128, affine=False))
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel), Dropout(0.4), Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False))

        modules1 = []
        for ii in range(len(blocks)):
            block = blocks[ii]
            if ii < 8 :
               unit_module = BasicBlockIR_hc
            else : unit_module = BasicBlockIR
            for bottleneck in block:
                modules1.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        # modules2 = []
        # for block in blocks:
        #     for bottleneck in block:
        #         modules2.append(
        #             unit_module(bottleneck.in_channel, bottleneck.depth,
        #                         bottleneck.stride))
        self.body1 = Sequential(*modules1)
        # self.body2 = Sequential(*modules2)

        initialize_weights(self.modules())

    def forward(self, x):
        # height = x.shape[2]
        # x = x[:, :, :int(4 * height / 7), :]
        x = self.input_layer(x)
        # height = x.shape[2]
        # x0 = x[:, :, :int(4 * height / 7), :]
        # x1 = x[:, :, int(4 * height / 7):, :]
        # x0 = self.body1(x0)
        x = self.body1(x)
        # x = cat([x0,x1],dim=2)
        height = x.shape[2]
        ch = x.shape[1]
        xc0 = x[:, :int(ch / 4), :, :]
        xc1 = x[:, int(ch/4):,:, :]
        x0 = xc1[:, :, :int(3 * height / 7), :]
        x1 = xc1[:, :, int(3 * height / 7):, :]
        xc0 = self.output_layer1(xc0)
        x0 = self.output_layer2(x0)
        x1 = self.output_layer3(x1)
        x = cat([xc0,x0,x1],dim=1)
        # x = self.output_layer1(x)
        return x


def IR_18(input_size):
    """ Constructs a ir-18 model.
    """
    model = Backbone(input_size, 18, 'ir')

    return model


def IR_34(input_size):
    """ Constructs a ir-34 model.
    """
    model = Backbone(input_size, 34, 'ir')

    return model


def IR_50(input_size):
    """ Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def IR_101(input_size):
    """ Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def IR_152(input_size):
    """ Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_200(input_size):
    """ Constructs a ir-200 model.
    """
    model = Backbone(input_size, 200, 'ir')

    return model


def IR_SE_50(input_size):
    """ Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """ Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """ Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model


def IR_SE_200(input_size):
    """ Constructs a ir_se-200 model.
    """
    model = Backbone(input_size, 200, 'ir_se')

    return model
