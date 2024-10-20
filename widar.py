import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class Widar_MLP(nn.Module):
    def __init__(self, num_classes):
        super(Widar_MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(22 * 20 * 20, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, 22 * 20 * 20)
        x = self.fc(x)
        return x


class Widar_LeNet(nn.Module):
    def __init__(self, num_classes):
        super(Widar_LeNet, self).__init__()
        self.encoder = nn.Sequential(
            # input size: (22,20,20)
            nn.Conv2d(22, 32, 6, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.ReLU(True),
            nn.Conv2d(64, 96, 3, stride=1),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
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

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
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


class Widar_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes):
        super(Widar_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.ConvTranspose2d(22, 3, 7, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=7, stride=1),
            nn.ReLU()
        )
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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


def Widar_ResNet18(num_classes):
    return Widar_ResNet(Block, [2, 2, 2, 2], num_classes=num_classes)


def Widar_ResNet50(num_classes):
    return Widar_ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def Widar_ResNet101(num_classes):
    return Widar_ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


class Widar_RNN(nn.Module):
    def __init__(self, num_classes):
        super(Widar_RNN, self).__init__()
        self.rnn = nn.RNN(400, 64, num_layers=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 22, 400)
        x = x.permute(1, 0, 2)
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs


class Widar_GRU(nn.Module):
    def __init__(self, num_classes):
        super(Widar_GRU, self).__init__()
        self.gru = nn.GRU(400, 64, num_layers=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 22, 400)
        x = x.permute(1, 0, 2)
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return outputs


class Widar_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(Widar_LSTM, self).__init__()
        self.lstm = nn.LSTM(400, 64, num_layers=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 22, 400)
        x = x.permute(1, 0, 2)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class Widar_BiLSTM(nn.Module):
    def __init__(self, num_classes):
        super(Widar_BiLSTM, self).__init__()
        self.lstm = nn.LSTM(400, 64, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 22, 400)
        x = x.permute(1, 0, 2)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


# class Widar_CNN_GRU(nn.Module):
#     def __init__(self, num_classes):
#         super(Widar_CNN_GRU, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 8, 6, 2),
#             nn.ReLU(),
#             nn.Conv2d(8, 16, 3, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(16 * 3 * 3, 64),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#         )
#         self.gru = nn.GRU(64, 128, num_layers=1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         batch_size = len(x)
#         # batch x 22 x 20 x 20
#         x = x.view(batch_size * 22, 1, 20, 20)
#         # 22*batch x 1 x 20 x 20
#         x = self.encoder(x)
#         # 22*batch x 16 x 3 x 3
#         x = x.view(-1, 16 * 3 * 3)
#         x = self.fc(x)
#         # 22*batch x 64
#         x = x.view(-1, 22, 64)
#         x = x.permute(1, 0, 2)
#         # 22 x batch x 64
#         _, ht = self.gru(x)
#         outputs = self.classifier(ht[-1])
#         return outputs
class Widar_CNN_GRU(nn.Module):
    def __init__(self, num_classes):
        super(Widar_CNN_GRU, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 6, 2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 3 * 3, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.gru = nn.GRU(64, 128, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size = len(x)
        # batch x 22 x 20 x 20
        x = x.view(batch_size * 22, 1, 20, 20)
        # 22*batch x 1 x 20 x 20
        x = self.encoder(x)
        # 22*batch x 16 x 3 x 3
        x = x.view(-1, 16 * 3 * 3)
        x = self.fc(x)
        # 22*batch x 64
        x = x.view(-1, 22, 64)
        x = x.permute(1, 0, 2)
        # 22 x batch x 64
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return outputs

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size_w=2, patch_size_h=40, emb_size=2 * 40, img_size=22 * 400):
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size_w, patch_size_h),
                      stride=(patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position = nn.Parameter(torch.randn(int(img_size / emb_size) + 1, emb_size))

    def forward(self, x):
        x = x.view(-1, 1, 22, 400)
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=80, num_heads=4, dropout=0.2):
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
    # def __init__(self, emb_size, expansion=4, drop_p=0.2):
    #     super().__init__(
    #         nn.Linear(emb_size, expansion * emb_size),
    #         nn.GELU(),
    #         nn.Dropout(drop_p),
    #         nn.Linear(expansion * emb_size, emb_size),
    #         nn.Dropout(drop_p)
    #     )
    def __init__(self, emb_size, expansion=2, drop_p=0.2):
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


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=80,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.2,
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
    def __init__(self, depth=2, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes))


class Widar_ViT(nn.Sequential):
    def __init__(self,
                 in_channels=1,
                 patch_size_w=2,
                 patch_size_h=40,
                 emb_size=80,
                 img_size=22 * 400,
                 depth=2,
                 *,
                 num_classes,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )

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

class  Widar_ViTLSTM(nn.Module):
    def __init__(self, in_channels=1, patch_size_w=2, patch_size_h=40, emb_size=80,
                 img_size=22*400, depth=2, heads=10, forward_expansion=4, dropout=0.5, lstm_hidden=48, n_classes=22,**kwargs):
        super( Widar_ViTLSTM, self).__init__()
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


class  Widar_ViTBiLSTM(nn.Module):

    def __init__(self, in_channels=1, patch_size_w=2, patch_size_h=40, emb_size=80, img_size=22 * 400, depth=1,
                     heads=4, forward_expansion=4, dropout=0.5, lstm_hidden=32, n_classes=22, **kwargs):
        super( Widar_ViTBiLSTM, self).__init__()
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

# ***********************

# **********************
class PatchEmbeddingMCF(nn.Module):
    def __init__(self, in_channels=1, patch_size_w=2, patch_size_h=40, emb_size=2 * 40, img_size=22 * 400):
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
        x = x.view(-1, 1, 22, 400)
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x


class MultiHeadAttentionMCF(nn.Module): # 一种注意力机制，主要用于NLP
    def __init__(self, emb_size=80, num_heads=4, dropout=0.5):
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

class FeedForwardBlockMCF(nn.Sequential):
    # def __init__(self, emb_size, expansion=4, drop_p=0.):
    #     super().__init__(
    #         nn.Linear(emb_size, expansion * emb_size),
    #         nn.GELU(),
    #         nn.Dropout(drop_p),
    #         nn.Linear(expansion * emb_size, emb_size),
    #     )
    def __init__(self, emb_size, expansion=2, drop_p=0.5):
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



class TransformerBlockViTBiLSTMMCF(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, dropout, lstm_hidden):
        super(TransformerBlockViTBiLSTMMCF, self).__init__()
        self.attention = MultiHeadAttentionMCF(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        # 改为使用双向LSTM
        self.lstm = nn.LSTM(emb_size, lstm_hidden, batch_first=True, bidirectional=True)
        # LSTM输出维度现在是 2 * lstm_hidden
        self.lstm_fc = nn.Linear(2 * lstm_hidden, emb_size)  # 用于调整LSTM输出维度回到emb_size
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = FeedForwardBlockMCF(emb_size, forward_expansion, dropout)
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

class TransformerEncoderViTBiLSTMMCF(nn.Module):
    def __init__(self, depth, emb_size, heads, forward_expansion, dropout, lstm_hidden):
        super(TransformerEncoderViTBiLSTMMCF, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlockViTBiLSTMMCF(emb_size, heads, forward_expansion, dropout, lstm_hidden) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Widar_ViTBiLSTMMCF(nn.Module):
    def __init__(self, in_channels=1, patch_size_w=2, patch_size_h=40, emb_size=80, img_size=22*400, depth=2, heads=4, forward_expansion=4, dropout=0.2, lstm_hidden=48 , n_classes=22,**kwargs):
        super(Widar_ViTBiLSTMMCF, self).__init__()
        self.patch_embedding = PatchEmbeddingMCF(in_channels, patch_size_w, patch_size_h, emb_size, img_size)
        self.encoder = TransformerEncoderViTBiLSTMMCF(depth, emb_size, heads, forward_expansion, dropout, lstm_hidden)
        # 使用线性层而不是分类头，因为在TransformerBlock中已经有了归一化层
        self.classifier = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        # 假设全局平均池化发生在序列长度维度上
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


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
class Widar_ABLSTM(nn.Module):
    def __init__(self, input_channels=1, input_dim=16*199, hidden_dim=128, num_layers=2, num_classes=22):
        super(Widar_ABLSTM, self).__init__()
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
        x = x.view(-1, 1, 22, 400)
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

class Widar_SAE(nn.Module):
    def __init__(self, num_classes=22):
        super(Widar_SAE, self).__init__()

        # 修改卷积层1：输入通道改为22
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)

        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(48)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 动态全连接层
        self.fc1 = None  # 初始化为None
        self.fc2 = nn.Linear(128, num_classes)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 22, 20, 20)
        # 输入形状: (64, 22, 20, 20)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 经过卷积、BN和池化
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 经过卷积、BN和池化

        # 计算展平后的特征图形状，动态创建全连接层
        x = torch.flatten(x, start_dim=1)  # 将特征展平为一维

        # 初始化全连接层：动态调整 fc1 输入尺寸
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)  # 根据展平后的特征维度动态创建fc1层

        x = self.relu(self.fc1(x))  # 全连接层
        x = self.fc2(x)  # 输出层
        return x
##*****************************************CNN+BiLSTM***********************************
class CNN_Module(nn.Module):
    def __init__(self):
        super(CNN_Module, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(22, 64, kernel_size=3, padding=1),  # 输入通道为1，输出通道为64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 空间维度减半
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 空间维度再减半
        )

    def forward(self, x):
        return self.cnn(x)


# 双向 LSTM 模块
class BiLSTM_Module(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM_Module, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        output, _ = self.bilstm(x)
        return output  # 返回 [batch_size, seq_len, hidden_size*2]

# 整体模型
class Widar_CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=22):
        super(Widar_CNN_BiLSTM, self).__init__()
        self.cnn_module = CNN_Module()
        self.bilstm_module1 = BiLSTM_Module(input_size=128 * 5, hidden_size=128, num_layers=1)  # input_size 变为 128*5
        self.bilstm_module2 = BiLSTM_Module(input_size=128 * 2, hidden_size=128, num_layers=1)
        self.fc = nn.Linear(128 * 2, num_classes)  # 双向 LSTM 输出为 hidden_size*2


    def forward(self, x):
        batch_size = x.size(0)

        # CNN 提取空间特征
        cnn_features = self.cnn_module(x)  # 输出 [batch_size, 128, new_height, new_width]

        # 展平 CNN 输出
        cnn_features = cnn_features.view(batch_size, cnn_features.size(2),
                                         -1)  # [batch_size, new_height, 128*new_width]

        # Bi-LSTM 层1
        lstm_out1 = self.bilstm_module1(cnn_features)  # 输出 [batch_size, new_height, 256]

        # Bi-LSTM 层2
        lstm_out2 = self.bilstm_module2(lstm_out1)  # 输出 [batch_size, new_height, 256]

        # 最后时刻的输出用于分类
        final_output = lstm_out2[:, -1, :]  # 取最后一个时间步的输出 [batch_size, 256]

        # 全连接层输出
        output = self.fc(final_output)  # 输出 [batch_size, num_classes]
        return F.log_softmax(output, dim=1)
