import torch
from torch import nn

class ecgTransForm(nn.Module):
    def __init__(self, 
                 num_classes=1, 
                 class_names=['C'], 
                 sequence_len=4096, 
                 input_channels=12, 
                 kernel_size=8, 
                 stride=1, 
                 dropout=0.2, 
                 mid_channels=32, 
                 final_out_channels=128, 
                 trans_dim=25, 
                 num_heads=5, 
                 feature_dim=128):
        super(ecgTransForm, self).__init__()

        self.num_classes = num_classes
        self.class_names = class_names
        self.sequence_len = sequence_len

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout

        self.mid_channels = mid_channels
        self.final_out_channels = final_out_channels
        # NOTE: WE remove feature_dim, and instead assume that trans_dim (transformer dim) is the internal dimensional representation
        # self.feature_dim = feature_dim

        self.trans_dim = trans_dim
        self.num_heads = num_heads

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(self.input_channels, self.mid_channels, kernel_size=filter_sizes[0],
                               stride=self.stride, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(self.input_channels, self.mid_channels, kernel_size=filter_sizes[1],
                               stride=self.stride, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(self.input_channels, self.mid_channels, kernel_size=filter_sizes[2],
                               stride=self.stride, bias=False, padding=(filter_sizes[2] // 2))

        self.bn = nn.BatchNorm1d(self.mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(self.dropout)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(self.mid_channels, self.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(self.mid_channels * 2, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.inplanes = 128
        self.crm = self._make_layer(SEBasicBlock, 128, 3)

        # NOTE: LINE BELOW added by me
        self.projection = nn.Linear(self.final_out_channels, self.trans_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.trans_dim, nhead=self.num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(self.trans_dim, self.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x_in):
        # Multi-scale Convolutions
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)
        x_concat = torch.mean(torch.stack([x1, x2, x3], dim=2), dim=2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))
        
        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)
        
        # Channel Recalibration Module
        x = self.crm(x)
        
        # Store batch size and sequence length for later reshaping
        batch_size, channels, seq_len = x.shape
        
        # Permute to get sequence dimension first for transformer
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        
        # Reshape for projection
        x_reshaped = x.reshape(-1, self.final_out_channels)
        
        # Project to transformer dimension
        x_projected = self.projection(x_reshaped)
        
        # Reshape back for transformer
        x = x_projected.reshape(batch_size, seq_len, self.trans_dim)
        
        # Bi-directional Transformer - flip along the sequence dimension (dim=1)
        x1 = self.transformer_encoder(x)
        x2 = self.transformer_encoder(torch.flip(x, [1]))
        x = x1 + x2
        
        # Permute for adaptive average pooling
        x = x.permute(0, 2, 1)  # [batch_size, trans_dim, seq_len]
        
        # Apply adaptive average pooling over sequence length
        x = self.aap(x)  # [batch_size, trans_dim, 1]
        
        # Remove the last dimension
        x_flat = x.squeeze(-1)  # [batch_size, trans_dim]
        
        # # Optional projection from trans_dim to feature_dim
        # x_flat = self.projection2(x_flat)  # [batch_size, feature_dim]
        
        # Final classification
        x_out = self.clf(x_flat)
        return x_out
    # def forward(self, x_in):

    #     print("before convs", x_in.shape)
    #     # Multi-scale Convolutions
    #     x1 = self.conv1(x_in)
    #     print("x1", x1.shape)
    #     x2 = self.conv2(x_in)
    #     x3 = self.conv3(x_in)
    #     print("x3", x3.shape)
    #     x_concat = torch.mean(torch.stack([x1, x2, x3],2), 2)
    #     x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))
    #     print("x_concat", x_concat.shape)

    #     x = self.conv_block2(x_concat)
    #     print("x conv block 2", x.shape)
    #     x = self.conv_block3(x)
    #     print("x after block3", x.shape)

    #     # Channel Recalibration Module
    #     x = self.crm(x)

    #     print("after crm", x.shape)
    #     # seq_len = x.shape[2]
    #     # x = x.permute(0, 2, 1)
    #     # x = x.reshape(-1, self.final_out_channels)
    #     print("before projection", x.shape)
    #     # project to transformer dimension
    #     x = self.projection(x)
    #     print("after projection", x.shape)
    #     # x = x.reshape(-1, seq_len, self.trans_dim)
    #     # Bi-directional Transformer
    #     x1 = self.transformer_encoder(x)
    #     x2 = self.transformer_encoder(torch.flip(x,[2]))
    #     x = x1+x2
    #     print("before avg pool", x.shape)
    #     x = self.aap(x)
    #     print("after avg pool", x.shape)
    #     x_flat = x.reshape(x.shape[0], -1)
    #     x_out = self.clf(x_flat)
    #     return x_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out