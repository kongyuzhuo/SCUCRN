import torch.nn as nn
import torch
import cv2

cnt = 0


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)  # nIn 相当于神经元的个数
        # 相当于全连接层，*2因为使用双向LSTM，两个方向隐层单元拼在一起
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)  # 得到一个feature map
        T, b, h = recurrent.size()    # T就是时间步长，b是batch_size，h是hidden unit
        t_rec = recurrent.view(T * b, h)  # RNN输出的结果进行一个尺度变换，把T和b结合在一起

        output = self.embedding(t_rec)  # 输出的结果维度：[T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        # imgH：图片高度
        # nc：输入图片通道数  ----灰度图是1
        # nclass：分类数目   ----比如从a到z 是27，多一个空白符
        # nh：rnn隐藏层神经元节点数  ----通常设置为256
        # n_rnn：rnn的层数
        # LeakyReLu：是否使用LeakyReLu
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]    # 卷积层卷积尺寸3表示3×3，2表示2×2
        ps = [1, 1, 1, 1, 1, 1, 0]    # padding 大小 （填充）
        ss = [1, 1, 1, 1, 1, 1, 1]    # stride 大小   （滑动的列数）
        nm = [64, 128, 256, 256, 512, 512, 512]   # 卷积核个数
        # ----竖着看，是一个卷积的配置

        cnn = nn.Sequential()   # 依次被传入构造器当中

        # ---convRelu是创建一个卷积层

        def convRelu(i, batchNormalization=False):
            # 确定输入 channel 维度  ----i=0,表示是第一个卷积层
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]      # 确定输入 channel 维度
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))  # 添加卷积层
            # BN 层
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            # ReLu 激活层
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # 拿输入图像为 1×32×100 举例（灰度图）

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        # ---nn.MaxPool2d(kernel_size(2x2),stride)
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        # ---nn.MaxPool2d(kernel_size ,stride, padding) 其中kernel_size 可以是数值，可以是元组
        # ---元组：高是2，宽是2(kernel_size); 高度移动2，宽度移动1(stride); 高度上没有padding，宽度上padding为1
        # ---高度上（2，2，0）和之前没区别，相当于nn.MaxPool2d(2, 2)
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),       # nh 是hidden unit
            BidirectionalLSTM(nh, nh, nclass))    # nclass 是类别的个数

    def forward(self, input):
        conv = self.cnn(input)    # conv features 得到feature map
        b, c, h, w = conv.size()  # b是batch_size c是channel h是高度 w是宽度
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)    # squeeze(a)就是将a中所有为1的维度删掉,即h
        conv = conv.permute(2, 0, 1)  # [w, b, c] [26,b,512]

        # rnn features
        output = self.rnn(conv)

        return output


if __name__ == '__main__':
    crnn = CRNN(32, 3, 37, 256)
    input = torch.Tensor(32, 3, 16, 64)
    output = crnn(input)
    print(output.shape)
