import torch
import torch.nn as nn
# from Model.Model import MLP as Encoder
# from Model.Model import Predictor
from Model.efficient_kan import KAN
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class Encoder(nn.Module):
    def __init__(self,input_dim=17,output_dim=1,layers_num=4,hidden_dim=50,droupout=0.2):
        super(Encoder, self).__init__()

        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                self.layers.append(Sin())
                self.layers.append(nn.Dropout(p=droupout))
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)    # 使用 Xavier 正态分布初始化权重

    def forward(self,x):
        x = self.net(x)
        return x


class Predictor(nn.Module):
    def __init__(self,input_dim=40):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim,32),
            Sin(),
            nn.Linear(32,1)
        )
        self.input_dim = input_dim
    def forward(self,x):
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),

            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out

class kan(nn.Module):
    def __init__(self):
        super(kan, self).__init__()
        self.encoder = KAN([17,60,60,32]).to(device)
        self.predictor = KAN([32,32,1]).to(device)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x
    
class KAN_medium(nn.Module):
    def __init__(self):
        super(KAN_medium, self).__init__()
        self.backbone = KAN([17,34,1]).to(device)

    def forward(self,x):
        x = self.backbone(x)
        return x   

class KAN_small(nn.Module):
    def __init__(self):
        super(KAN_small, self).__init__()
        self.encoder = KAN([2,4]).to(device)
        self.predictor = KAN([4,1]).to(device)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x    

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.encoder = Encoder(input_dim=17, output_dim=32, layers_num=3, hidden_dim=60, droupout=0.2)
        self.predictor = Predictor(input_dim=32)
        self.to(device)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = ResBlock(input_channel=1, output_channel=8, stride=1).to(device)  # N,8,17
        self.layer2 = ResBlock(input_channel=8, output_channel=16, stride=2).to(device)  # N,16,9
        self.layer3 = ResBlock(input_channel=16, output_channel=24, stride=2).to(device)  # N,24,5
        self.layer4 = ResBlock(input_channel=24, output_channel=16, stride=1).to(device)  # N,16,5
        self.layer5 = ResBlock(input_channel=16, output_channel=8, stride=1).to(device)  # N,8,5
        self.layer6 = nn.Linear(8*5,1)

    def forward(self, x):
        N,L = x.shape[0],x.shape[1]
        x = x.view(N,1,L)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out.view(N,-1))
        return out.view(N,1)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=17, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(64 * 2, 1)
        self.to(device)

    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, input_size)
        x = x.unsqueeze(1)  # (N, 1, 17)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (N, 1, 64*2)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]  # (N, 64*2)
        
        # Fully connected layer
        out = self.fc(lstm_out)  # (N, 1)
        
        return out.view(-1, 1)


class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        # TCN layers with dilated convolutions
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 1)
        self.to(device)

    def forward(self, x):
        N, L = x.shape[0], x.shape[1]
        x = x.view(N, 1, L)  # (N, 1, 17)
        
        # TCN forward pass
        tcn_out = self.tcn(x)  # (N, 128, 1)
        
        # Flatten
        tcn_out = tcn_out.view(N, -1)  # (N, 128)
        
        # Fully connected layer
        out = self.fc(tcn_out)  # (N, 1)
        
        return out.view(-1, 1)

class Transformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=1): # d_model=32, nhead=4, num_layers=1,2;
        super(Transformer, self).__init__()
        # 输入映射：将每个变量的 1 维特征映射到 d_model
        self.input_proj = nn.Linear(1, d_model)
        # Transformer Encoder（无位置编码，因变量无顺序依赖）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,  # FFN 隐藏层维度
            batch_first=True,  # 输入格式 (B, seq_len, d_model)
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出头：聚合变量特征 → 预测 SOH (B, 1)
        self.output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化：(B, d_model, M) → (B, d_model, 1)
            nn.Flatten(),  # (B, d_model)
            nn.Linear(d_model, 1)  # 回归输出 SOH
        )
    
    def forward(self, x):
        # x: (B, M) → (B, M, 1) → (B, M, d_model)
        x = self.input_proj(x.unsqueeze(-1))
        # Transformer 编码：建模变量间依赖 → (B, M, d_model)
        x_encoded = self.transformer_encoder(x)
        # 维度转置：(B, M, d_model) → (B, d_model, M)（适配池化层）
        x_transposed = x_encoded.transpose(1, 2)
        # 输出 SOH：(B, 1)
        soh = self.output_head(x_transposed)
        return soh

class Transformer_v1(nn.Module):
    def __init__(self):
        super(Transformer_v1, self).__init__()
        self.input_dim = 17
        self.hidden_dim = 64
        self.num_heads = 4
        self.num_layers = 1
        
        # Embedding layer to project input to hidden dimension
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Positional encoding
        self.positional_encoding = self._get_positional_encoding(1, self.hidden_dim)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.to(device)

    def _get_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        N, L = x.shape[0], x.shape[1]
        
        # Reshape to (N, 1, 17) for sequence processing
        x = x.unsqueeze(1)  # (N, 1, 17)
        
        # Embedding
        x_emb = self.embedding(x)  # (N, 1, 64)
        
        # Add positional encoding
        x_emb = x_emb + self.positional_encoding[:, :x_emb.size(1), :].to(device)  # (N, 1, 64)
        
        # Transformer forward pass
        transformer_out = self.transformer_encoder(x_emb)  # (N, 1, 64)
        
        # Take the output from the only time step
        transformer_out = transformer_out[:, -1, :]  # (N, 64)
        
        # Fully connected layer
        out = self.fc(transformer_out)  # (N, 1)
        
        return out.view(-1, 1)


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))


if __name__ == '__main__':
    x = torch.randn(10,17)
    y1 = MLP()(x)
    y2 = CNN()(x)
    y3 = LSTM()(x)
    y4 = TCN()(x)
    y5 = Transformer()(x)
    
    print("MLP output shape:", y1.shape)
    print("CNN output shape:", y2.shape)
    print("LSTM output shape:", y3.shape)
    print("TCN output shape:", y4.shape)
    print("Transformer output shape:", y5.shape)
    
    count_parameters(MLP())
    count_parameters(CNN())
    count_parameters(LSTM())
    count_parameters(TCN())
    count_parameters(Transformer())