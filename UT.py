import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class UT_HAR_MLP(nn.Module):
    def __init__(self):
        super(UT_HAR_MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(250 * 90, 1024),  # 一个数据的张量为250*9
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = x.view(-1, 250 * 90)
        x = self.fc(x)
        return x


class UT_HAR_LeNet(nn.Module):
    def __init__(self):
        super(UT_HAR_LeNet, self).__init__()
        self.encoder = nn.Sequential(
            # input size: (1,250,90)
            nn.Conv2d(1, 32, 7, stride=(3, 1)),  # 输入，输出，卷积核，步长水平步长3，垂直滑动1  32， 82，84
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),  # 20*20
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 64*10*10
            nn.Conv2d(64, 96, (3, 3), stride=1),  # 96*8*8
            nn.ReLU(True),
            nn.MaxPool2d(2)  # 96*4*4
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 96 * 4 * 4)
        out = self.fc(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class UT_HAR_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=7):  # 7训练集的分类个数
        super(UT_HAR_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(1, 3, 7, stride=(3, 1)),  # 输入1*250*90 输出 3*82*84
            nn.ReLU(),
            nn.MaxPool2d(2),  # 3*41*42
            nn.Conv2d(3, 3, kernel_size=(10, 11), stride=1),  # 3*32*32
            nn.ReLU()
        )
        self.in_channels = 64  #

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入3*250*90 输出64*125*45
        self.batch_norm1 = nn.BatchNorm2d(64)  # 数据归一化
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 输出64*65*23

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def UT_HAR_ResNet18():
    return UT_HAR_ResNet(Block, [2, 2, 2, 2])


def UT_HAR_ResNet50():
    return UT_HAR_ResNet(Bottleneck, [3, 4, 6, 3])


def UT_HAR_ResNet101():
    return UT_HAR_ResNet(Bottleneck, [3, 4, 23, 3])


class UT_HAR_RNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(UT_HAR_RNN, self).__init__()
        self.rnn = nn.RNN(90, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        x = x.view(-1, 250, 90)
        x = x.permute(1, 0, 2)  # 250 1 90
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_GRU(nn.Module):
    def __init__(self, hidden_dim=64):
        super(UT_HAR_GRU, self).__init__()
        self.gru = nn.GRU(90, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        x = x.view(-1, 250, 90)
        x = x.permute(1, 0, 2)
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_LSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super(UT_HAR_LSTM, self).__init__()
        self.lstm = nn.LSTM(90, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        x = x.view(-1, 250, 90)
        x = x.permute(1, 0, 2)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_BiLSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super(UT_HAR_BiLSTM, self).__init__()
        self.lstm = nn.LSTM(90, hidden_dim, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        x = x.view(-1, 250, 90)
        x = x.permute(1, 0, 2)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_CNN_GRU(nn.Module):
    def __init__(self):
        super(UT_HAR_CNN_GRU, self).__init__()
        self.encoder = nn.Sequential(
            # input size: (250,90)
            nn.Conv1d(250, 250, 12, 3),
            nn.ReLU(True),
            nn.Conv1d(250, 250, 5, 2),
            nn.ReLU(True),
            nn.Conv1d(250, 250, 5, 1)
            # 250 x 8
        )
        self.gru = nn.GRU(8, 128, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # batch x 1 x 250 x 90
        x = x.view(-1, 250, 90)
        x = self.encoder(x)
        # batch x 250 x 8
        x = x.permute(1, 0, 2)
        # 250 x batch x 8
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return outputs


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size_w=25, patch_size_h=18, emb_size=25 * 18, img_size=250 * 90):
        # 输入通道数为1,每个patch宽=50，高=18,图片的大小为250*90，相当于分了5*5=25个patch
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size_w, patch_size_h),
                      stride=(patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),  # 例如 5*6*7*8 -> 5*56*6
            # rearrange：用于对张量的维度进行重新变换排序，可用于替换pytorch中的reshape，view，transpose和permute等操作
            # repeat：用于对张量的某一个维度进行复制，可用于替换pytorch中的repeat
            # reduce：类似于tensorflow中的reduce操作，可以用于求平均值，最大最小值的同时压缩张量维度。分别有max，min，sum，mean，prod。

        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))  # 多输入一个向量，开始预测
        self.position = nn.Parameter(torch.randn(int(img_size / emb_size) + 1, emb_size))  # 26，50*18

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x


class MultiHeadAttention(nn.Module): # 一种注意力机制，主要用于NLP
    def __init__(self, emb_size=450, num_heads=3, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    # def __init__(self, emb_size, expansion=4, drop_p=0.):
    #     super().__init__(
    #         nn.Linear(emb_size, expansion * emb_size),
    #         nn.GELU(),
    #         nn.Dropout(drop_p),
    #         nn.Linear(expansion * emb_size, emb_size),
    #     )
    #*************************************************************************
    def __init__(self, emb_size, expansion=2, drop_p=0.1 ):
        super().__init__()
        # 使用不同大小的卷积核
        self.conv1 = nn.Conv1d(emb_size, expansion * emb_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(emb_size, expansion * emb_size, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(drop_p)
        self.projection = nn.Conv1d(expansion * emb_size * 2, emb_size, kernel_size=1)

    def forward(self, x):
        x = rearrange(x, 'b n e -> b e n')
        x1 = self.gelu(self.conv1(x))
        x2 = self.gelu(self.conv2(x))
        out = torch.cat([x1, x2], dim=1)  # 将不同卷积核的输出拼接
        out = self.projection(out)
        out = rearrange(out, 'b e n -> b n e')
        return self.drop(out)
    #*******************************卷积核大小*********************************
    # def __init__(self, emb_size, expansion=2, drop_p=0.1):
    #     super().__init__()
    #     # 使用不同大小的卷积核
    #     self.conv1 = nn.Conv1d(emb_size, expansion * emb_size, kernel_size=7, padding=3)  # 卷积核大小为 3
    #     self.conv2 = nn.Conv1d(emb_size, expansion * emb_size, kernel_size=7, padding=3)  # 卷积核大小为 5
    #     self.conv3 = nn.Conv1d(emb_size, expansion * emb_size, kernel_size=7, padding=3)  # 卷积核大小为 7
    #     self.gelu = nn.GELU()
    #     self.drop = nn.Dropout(drop_p)
    #
    #     # 投影层的输入维度需要根据拼接后的通道数调整
    #     self.projection = nn.Conv1d(expansion * emb_size * 3, emb_size, kernel_size=1)
    #
    # def forward(self, x):
    #     x = rearrange(x, 'b n e -> b e n')  # 调整输入维度
    #     x1 = self.gelu(self.conv1(x))  # 卷积层 1（核大小 3）
    #     x2 = self.gelu(self.conv2(x))  # 卷积层 2（核大小 5）
    #     x3 = self.gelu(self.conv3(x))  # 卷积层 3（核大小 7）
    #
    #     # 将三个卷积层的输出在通道维度上拼接
    #     out = torch.cat([x1, x2,x3], dim=1)
    #     out = self.projection(out)  # 通过投影层减少通道数
    #     out = rearrange(out, 'b e n -> b n e')  # 调整回输出维度
    #     return self.drop(out)


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=450,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.1,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=1, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=900, n_classes=7):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class UT_HAR_ViT(nn.Sequential): # vision Transformer
    def __init__(self,
                 in_channels=1,
                 patch_size_w=50,
                 patch_size_h=18,
                 emb_size=900,
                 img_size=250 * 90,
                 depth=1,
                 n_classes=7,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

class TransformerBlockViTBiLSTM(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, dropout, lstm_hidden):
        super(TransformerBlockViTBiLSTM, self).__init__()
        self.attention = MultiHeadAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        # 改为使用双向LSTM
        self.lstm = nn.LSTM(emb_size, lstm_hidden, batch_first=True, bidirectional=True)
        # LSTM输出维度现在是 2 * lstm_hidden
        self.lstm_fc = nn.Linear(2 * lstm_hidden, emb_size)  # 用于调整LSTM输出维度回到emb_size
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = FeedForwardBlock(emb_size, forward_expansion, dropout)
        self.norm3 = nn.LayerNorm(emb_size)

    def forward(self, value):
        attention = self.attention(value)
        x = value + attention
        x = self.norm1(x)

        # LSTM层处理，现在是双向的
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_fc(lstm_out)  # 通过线性层调整维度

        x = x + lstm_out  # 应用残差连接
        x = self.norm2(x)  # 再次归一化

        forward = self.feed_forward(x)
        x = x + forward  # 应用残差连接
        x = self.norm3(x)  # 最终归一化
        return x

class TransformerBlockLV(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, dropout, lstm_hidden):
        super(TransformerBlockLV, self).__init__()
        self.attention = MultiHeadAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        # 单向LSTM
        self.lstm = nn.LSTM(emb_size, lstm_hidden, batch_first=True, bidirectional=False)
        # LSTM输出维度现在是lstm_hidden
        self.lstm_fc = nn.Linear(lstm_hidden, emb_size)  # 用于调整LSTM输出维度回到emb_size
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = FeedForwardBlock(emb_size, forward_expansion, dropout)
        self.norm3 = nn.LayerNorm(emb_size)

    def forward(self, value):
        attention = self.attention(value)
        x = value + attention
        x = self.norm1(x)

        # LSTM层处理，现在是单向的
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_fc(lstm_out)  # 通过线性层调整维度

        x = x + lstm_out  # 应用残差连接
        x = self.norm2(x)  # 再次归一化

        forward = self.feed_forward(x)
        x = x + forward  # 应用残差连接
        x = self.norm3(x)  # 最终归一化
        return x

class TransformerEncoderLV(nn.Module):
    def __init__(self, depth, emb_size, heads, forward_expansion, dropout, lstm_hidden):
        super(TransformerEncoderLV, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlockLV(emb_size, heads, forward_expansion, dropout, lstm_hidden) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerEncoderViTBiLSTM(nn.Module):
    def __init__(self, depth, emb_size, heads, forward_expansion, dropout, lstm_hidden):
        super(TransformerEncoderViTBiLSTM, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlockViTBiLSTM(emb_size, heads, forward_expansion, dropout, lstm_hidden) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UT_HAR_ViTLSTM(nn.Module):
    def __init__(self, in_channels=1, patch_size_w=25, patch_size_h=18, emb_size=450, img_size=250*90, depth=1, heads=3 , forward_expansion=4, dropout=0.0, lstm_hidden=64, n_classes=7,**kwargs):
        super(UT_HAR_ViTLSTM, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size)
        self.encoder = TransformerEncoderLV(depth, emb_size, heads, forward_expansion, dropout, lstm_hidden)
        # 使用线性层而不是分类头，因为在TransformerBlock中已经有了归一化层
        self.classifier = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        # 假设全局平均池化发生在序列长度维度上
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
#*************************************

# ************************************
class UT_HAR_ViTBiLSTM(nn.Module):
    def __init__(self, in_channels=1, patch_size_w=25, patch_size_h=18, emb_size=450, img_size=250*90, depth=1, heads=3, forward_expansion=4, dropout=0.1, lstm_hidden=64, n_classes=7,**kwargs):
        super(UT_HAR_ViTBiLSTM, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size)
        self.encoder = TransformerEncoderViTBiLSTM(depth, emb_size, heads, forward_expansion, dropout, lstm_hidden)
        # 使用线性层而不是分类头，因为在TransformerBlock中已经有了归一化层
        self.classifier = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        # 假设全局平均池化发生在序列长度维度上
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

# TCN + LSTM

# Vit BiLSTM MCF反馈
class PatchEmbeddingMCF(nn.Module):
    def __init__(self, in_channels=1, patch_size_w=50, patch_size_h=18, emb_size=50 * 18, img_size=250 * 90):
        # 初始化PatchEmbedding类，设置输入通道数、补丁大小、嵌入维度和图像大小
        self.patch_size_w = patch_size_w  # 设置每个补丁的宽度
        self.patch_size_h = patch_size_h  # 设置每个补丁的高度
        super().__init__()
        self.projection = nn.Sequential(
            # 使用卷积层作为投影方法，将每个patch映射到embedding空间
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size_w, patch_size_h),
                      stride=(patch_size_w, patch_size_h)),
            # Rearrange用于将卷积层的输出重排成Transformer所需的格式
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # 引入分类标记（cls_token），这是Transformer中的一个额外的可学习向量
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # 添加位置编码，为每个patch和cls_token提供位置信息
        self.position = nn.Parameter(torch.randn(int(img_size / emb_size) + 1, emb_size))

    def forward(self, x):
        # 定义前向传播方法
        b, _, _, _ = x.shape  # 获取输入的批量大小
        x = self.projection(x)  # 应用投影层
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)  # 复制cls_token到每个样本
        x = torch.cat([cls_tokens, x], dim=1)  # 将cls_token拼接到patch embeddings上
        x += self.position  # 添加位置编码
        return x


class MultiHeadAttentionMCF(nn.Module):
    # 实现多头注意力机制
    def __init__(self, emb_size=450, num_heads=3, dropout=0.2):
        super().__init__()
        self.emb_size = emb_size  # 嵌入维度
        self.num_heads = num_heads  # 多头数量
        self.qkv = nn.Linear(emb_size, emb_size * 3)  # 创建一个线性层，用于生成查询、键、值
        self.att_drop = nn.Dropout(dropout)  # 定义注意力dropout
        self.projection = nn.Linear(emb_size, emb_size)  # 最后的投影层

    def forward(self, x, mask=None):
        # 定义前向传播方法
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        # 重排查询、键、值
        queries, keys, values = qkv[0], qkv[1], qkv[2]  # 分离查询、键、值
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # 计算注意力分数
        if mask is not None:
            # 应用掩码（如果有）
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)  # 计算缩放因子
        att = F.softmax(energy, dim=-1) / scaling  # 应用softmax并缩放
        att = self.att_drop(att)  # 应用注意力dropout
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)  # 聚合值
        out = rearrange(out, "b h n d -> b n (h d)")  # 重新排列输出
        out = self.projection(out)  # 应用最后的投影层
        return out


class MixedConvFeedForward(nn.Module):
    def __init__(self, emb_size, expansion=3, drop_p=0.2):
        super().__init__()
        # 使用不同大小的卷积核
        self.conv1 = nn.Conv1d(emb_size, expansion * emb_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(emb_size, expansion * emb_size, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(drop_p)
        self.projection = nn.Conv1d(expansion * emb_size * 2, emb_size, kernel_size=1)

    def forward(self, x):
        x = rearrange(x, 'b n e -> b e n')
        x1 = self.gelu(self.conv1(x))
        x2 = self.gelu(self.conv2(x))
        out = torch.cat([x1, x2], dim=1)  # 将不同卷积核的输出拼接
        out = self.projection(out)
        out = rearrange(out, 'b e n -> b n e')
        return self.drop(out)


class TransformerBlockMCF(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, dropout, lstm_hidden):
        super(TransformerBlockMCF, self).__init__()
        self.attention = MultiHeadAttentionMCF(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        # 2向LSTM
        self.lstm = nn.LSTM(emb_size, lstm_hidden, batch_first=True, bidirectional=True)
        # LSTM输出维度现在是lstm_hidden
        self.lstm_fc = nn.Linear(2*lstm_hidden, emb_size)  # 用于调整LSTM输出维度回到emb_size
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = MixedConvFeedForward(emb_size, forward_expansion, dropout)
        self.norm3 = nn.LayerNorm(emb_size)

    def forward(self, value):
        attention = self.attention(value)
        x = value + attention
        x = self.norm1(x)

        # LSTM层处理，现在是双向的
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_fc(lstm_out)  # 通过线性层调整维度

        x = x + lstm_out  # 应用残差连接
        x = self.norm2(x)  # 再次归一化

        forward = self.feed_forward(x)
        x = x + forward  # 应用残差连接
        x = self.norm3(x)  # 最终归一化
        return x

class TransformerEncoderMCF(nn.Module):
    def __init__(self, depth, emb_size, heads, forward_expansion, dropout, lstm_hidden):
        super(TransformerEncoderMCF, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlockMCF(emb_size, heads, forward_expansion, dropout, lstm_hidden) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ViTBiLSTMMCF(nn.Module):
    def __init__(self, in_channels=1, patch_size_w=25, patch_size_h=18, emb_size=900, img_size=250*90, depth=1, heads=3, forward_expansion=4, dropout=0.2, lstm_hidden=64, n_classes=7):
        super(ViTBiLSTMMCF, self).__init__()
        self.patch_embedding = PatchEmbeddingMCF(in_channels, patch_size_w, patch_size_h, emb_size, img_size)
        self.encoder = TransformerEncoderMCF(depth, emb_size, heads, forward_expansion, dropout, lstm_hidden)
        # 使用线性层而不是分类头，因为在TransformerBlock中已经有了归一化层
        self.classifier = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        # 假设全局平均池化发生在序列长度维度上
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
#*************************************************************************************************************

class UT_HAR_SAE(nn.Module):
    def __init__(self, num_classes=7):
        super(UT_HAR_SAE, self).__init__()

        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(16)

        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(32)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 动态全连接层
        self.fc1 = None  # 初始化为None
        self.fc2 = nn.Linear(128, num_classes)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入形状: (64, 1, 250, 90)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 经过卷积、BN和池化
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 经过卷积、BN和池化

        # 展平
        x = torch.flatten(x, start_dim=1)  # 将特征展平为一维

        # 初始化全连接层
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device) # 根据展平后的特征维度动态创建fc1层

        x = self.relu(self.fc1(x))  # 全连接层
        x = self.fc2(x)  # 输出层
        return x
#*************************************************************************************************************
# class UT_HAR_ABLSTM(nn.Module):
#     def __init__(self, input_size=90, output_size=7, seq_lens=250, num_hiddens=128, num_layers=2, num_heads=1):
#         super(UT_HAR_ABLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=num_hiddens,
#                             bidirectional=True, num_layers=num_layers,
#                             batch_first=True)  # 这里的LSTM是双向的
#         self.attention = nn.MultiheadAttention(embed_dim=num_hiddens * 2, num_heads=num_heads, batch_first=True)
#         self.fc1 = nn.Linear(num_hiddens * 2, num_hiddens * 2)  # LSTM输出的隐藏状态大小为num_hiddens * 2
#         self.fc2 = nn.Linear(num_hiddens * 2, 128)
#         self.fc3 = nn.Linear(128, output_size)
#
#     def forward(self, x):
#         # x的形状应该是 (batch_size, seq_len, input_size)
#         x = x.view(x.size(0), x.size(2), x.size(3))  # 调整为 (batch_size, seq_len, input_size)
#         out, _ = self.lstm(x)  # out的形状为 (batch_size, seq_len, 2 * num_hiddens)
#
#         # 使用最后一个时间步的输出
#         out = out[:, -1, :]  # 取出最后一个时间步的输出 (batch_size, 2 * num_hiddens)
#         x = self.fc1(out)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = torch.relu(x)
#         x = self.fc3(x)
#         return x
class Attention_ABLSTM(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention_ABLSTM, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)  # 计算注意力权重，输出为1维
        self.softmax = nn.Softmax(dim=1)  # 对时间步进行 softmax 操作

    def forward(self, lstm_output):
        # lstm_output: (batch_size, time_steps, hidden_dim)
        attn_weights = self.softmax(self.attn(lstm_output))  # (batch_size, time_steps, 1)
        attn_output = torch.bmm(attn_weights.transpose(1, 2), lstm_output)  # (batch_size, 1, hidden_dim)
        return attn_output.squeeze(1), attn_weights  # 去掉第1维，返回 (batch_size, hidden_dim)


# ABLSTM模型
class UT_HAR_ABLSTM(nn.Module):
    def __init__(self, input_channels=1, input_dim=16*44, hidden_dim=128, num_layers=2, num_classes=7):
        super(UT_HAR_ABLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=(3, 3), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 双向LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # 注意力层
        self.attention = Attention_ABLSTM(hidden_dim * 2)  # 双向LSTM输出维度是hidden_dim的两倍
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 卷积层
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))  # 输出形状: (batch_size, 16, 124, 44)

        # 调整形状以输入到LSTM中
        x = x.view(x.size(0), x.size(2), -1)  # 输出形状: (batch_size, 124, 16*44)

        # LSTM层
        lstm_out, _ = self.lstm(x)  # 输出形状: (batch_size, 124, 2*hidden_dim)

        # 注意力机制
        attn_output, _ = self.attention(lstm_out)  # 输出形状: (batch_size, 2*hidden_dim)

        # 全连接层
        fc_out = torch.relu(self.fc1(attn_output))  # 输出形状: (batch_size, hidden_dim)
        output = self.fc2(fc_out)  # 输出形状: (batch_size, num_classes)

        return output

#*************************************************************************************************************
#THAT
class ConvStream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvStream, self).__init__()
        # 定义两个卷积层，使用3x3的卷积核，且padding为1以保持特征图尺寸
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        # 定义最大池化层，池化窗口为2x2
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        # 卷积-激活-池化
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

class UT_HAR_THAT_TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(UT_HAR_THAT_TransformerBlock, self).__init__()
        # 定义多头自注意力层
        self.attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads)
        # 定义层归一化
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        # 定义前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.ReLU(),
            nn.Linear(emb_size * 2, emb_size)
        )

    def forward(self, x):
        # 通过自注意力层
        attn_out, _ = self.attention(x, x, x)
        # 残差连接和归一化
        x = self.norm1(attn_out + x)
        ff_out = self.feed_forward(x)
        return self.norm2(ff_out + x)  # 另一个残差连接

class UT_HAR_THAT(nn.Module):
    def __init__(self, num_classes=7):
        super(UT_HAR_THAT, self).__init__()
        # 定义空间流和时间流
        self.spatial_stream = ConvStream(in_channels=1, out_channels=64)  # 输入通道为1
        self.temporal_stream = ConvStream(in_channels=1, out_channels=64)  # 输入通道为1
        # 定义变换器块
        self.transformer = UT_HAR_THAT_TransformerBlock(emb_size=64, num_heads=8)
        # 定义全连接层
        self.fc = nn.Linear(64, num_classes)

    def forward(self, input_data):
        # 通过空间流和时间流提取特征
        spatial_features = self.spatial_stream(input_data)
        temporal_features = self.temporal_stream(input_data)

        # Flatten特征图并为变换器准备输入
        spatial_features = spatial_features.view(spatial_features.size(0), -1, 64)  # 这里的64是通道数
        temporal_features = temporal_features.view(temporal_features.size(0), -1, 64)

        # 合并两个流的特征
        combined_features = torch.cat((spatial_features, temporal_features), dim=1)
        # 通过变换器块
        transformer_output = self.transformer(combined_features)
        # 进行全局平均池化并通过全连接层得到输出
        output = self.fc(transformer_output.mean(dim=1))
        return output
#*************************************************************************************************************
#Sanet
class UT_HAR_Sanet(nn.Module):
    def __init__(self, num_classes=7):
        super(UT_HAR_Sanet, self).__init__()

        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)

        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Dropout层
        self.dropout = nn.Dropout(p=0.4)  # 50% 的 dropout

        # 初始化全连接层
        self.fc1 = None  # 初始化为None
        self.fc2 = nn.Linear(128, num_classes)  # 输出层

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 经过卷积、BN和池化
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 经过卷积、BN和池化

        # 展平
        x = x.view(x.size(0), -1)  # 将特征展平为一维

        # 动态创建全连接层
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)  # 在当前设备上创建fc1层

        x = self.dropout(self.relu(self.fc1(x)))  # 添加 Dropout
        x = self.fc2(x)  # 输出层
        return x
#***************************************************************************************************ACT

# CBAM 模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=32):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa

        return x

# CNN + CBAM 提取语义特征
class CNN_CBAM(nn.Module):
    def __init__(self, input_channels):
        super(CNN_CBAM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((125, 45)),  # 使用AdaptiveMaxPool代替MaxPool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((64, 22))
        )
        self.cbam = CBAM(128)

    def forward(self, x):
        x = self.cnn(x)
        x = self.cbam(x)
        return x

# BGRU + Self-Attention 提取时序特征
class BGRU_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout=0.3):
        super(BGRU_Attention, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4)
        self.dropout = nn.Dropout(dropout)  # 加入dropout层

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attention_out, _ = self.self_attention(gru_out, gru_out, gru_out)
        attention_out = self.dropout(attention_out)
        return attention_out

# 特征融合层
class FeatureFusion(nn.Module):
    def __init__(self, input_size):
        super(FeatureFusion, self).__init__()
        self.fc = nn.Linear(input_size, 256)
        self.dropout = nn.Dropout(p=0.3)


    def forward(self, cnn_features, gru_features):
        # # 打印特征维度
        # print(f"CNN Features shape: {cnn_features.shape}")
        # print(f"GRU Features shape: {gru_features.shape}")

        # 展平特征
        cnn_flatten = cnn_features.view(cnn_features.size(0), -1)
        gru_flatten = gru_features.view(gru_features.size(0), -1)

        # 合并特征
        combined_features = torch.cat([cnn_flatten, gru_flatten], dim=1)
        # print(f"Combined Features shape: {combined_features.shape}")

        # 动态调整全连接层输入尺寸
        if self.fc.in_features != combined_features.size(1):
            self.fc = nn.Linear(combined_features.size(1), 256).to(combined_features.device)

        # 全连接层处理
        fused_features = self.fc(combined_features)
        fused_features = self.dropout(fused_features)
        return fused_features


# 完整模型
class UT_HAR_ACT(nn.Module):
    def __init__(self, num_classes=7):
        super(UT_HAR_ACT, self).__init__()
        self.cnn_cbam = CNN_CBAM(input_channels=1)  # 输入1通道
        self.bgru_attention = BGRU_Attention(input_size=90, hidden_size=64, num_layers=2)  # 假设90为时序特征
        self.feature_fusion = FeatureFusion(input_size=256 + 128)  # 合并特征
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # 语义特征提取
        cnn_features = self.cnn_cbam(x)

        # 时序特征提取，将250作为序列长度
        gru_features = self.bgru_attention(x.view(x.size(0), x.size(2), -1))  # x调整为 [batch_size, seq_len, feature_dim]

        # 特征融合
        fused_features = self.feature_fusion(cnn_features.view(cnn_features.size(0), -1), gru_features.view(gru_features.size(0), -1))

        # 分类预测
        out = self.fc(fused_features)
        return out
##*****************************************CNN+BiLSTM
class CNN_Module(nn.Module):
    def __init__(self):
        super(CNN_Module, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入通道为1，输出通道为64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 空间维度减半
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 空间维度再减半
        )

    def forward(self, x):
        return self.cnn(x)  # 输出 [batch_size, 128, 62, 22] 的特征

# 双向 LSTM 模块
class BiLSTM_Module(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM_Module, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        output, _ = self.bilstm(x)
        return output  # 返回 [batch_size, seq_len, hidden_size*2]

# 整体模型
class UT_HAR_CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=7):
        super(UT_HAR_CNN_BiLSTM, self).__init__()
        self.cnn_module = CNN_Module()
        self.bilstm_module1 = BiLSTM_Module(input_size=128*22, hidden_size=128, num_layers=1)  # 注意 input_size 变为 128*22
        self.bilstm_module2 = BiLSTM_Module(input_size=128*2, hidden_size=128, num_layers=1)
        self.fc = nn.Linear(128*2, num_classes)  # 双向LSTM输出为 hidden_size*2

    def forward(self, x):
        batch_size = x.size(0)

        # CNN 提取空间特征
        cnn_features = self.cnn_module(x)  # 输出 [batch_size, 128, 62, 22]
        cnn_features = cnn_features.view(batch_size, 62, -1)  # 展平为 [batch_size, 62, 128*22]

        # Bi-LSTM 层1
        lstm_out1 = self.bilstm_module1(cnn_features)  # 输出 [batch_size, 62, 256]

        # Bi-LSTM 层2
        lstm_out2 = self.bilstm_module2(lstm_out1)  # 输出 [batch_size, 62, 256]

        # 最后时刻的输出用于分类
        final_output = lstm_out2[:, -1, :]  # 取最后一个时间步的输出 [batch_size, 256]

        # 全连接层输出
        output = self.fc(final_output)  # 输出 [batch_size, num_classes]
        return F.log_softmax(output, dim=1)
