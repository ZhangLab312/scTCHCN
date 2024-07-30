'''
Let's start with the theme framework of our model, but don't write the code yet. I'll guide you through each part step by step.

首先，我们这个框架的主题是自监督蒸馏学习模型，名为scTCHCN。

Here, the teacher model is an autoencoder model with two input channels. The first input channel is an encoder for sequence data, and the second input channel is a graph encoder for graph-structured data. The features extracted by these two encoders are concatenated (sequence features concatenated with features from all nodes in the graph) and input into the Teacher model, which is a multi-layer Transformer encoder that performs global feature learning on the concatenated features. The output features are then processed through two output channels: the first output channel sends the features to a sequence decoder and a graph decoder based on the previously concatenated input channels, and the second output channel sends the Transformer encoder's output features to the Student model for distillation learning.

这里的老师模型是一个由两个通道输入的自编码器模型。第一个输入通道是一个序列数据组成的编码器，第二个输入通道是一个由图构成的图编码器。两个编码器输出的数据特征通过拼接（序列特征拼接所有图中节点的特征）后输入到Teacher模型中，这个模型由多层Transformer编码器构成，用于对拼接的特征进行全局特征学习。输出的特征数据通过两个输出通道输入到后续层：第一种输出通道将特征按照之前拼接的输入通道一和输入通道二分别送入一个序列解码器和一个图解码器；第二种输出通道将Transformer编码器的输出特征送入Student模型中进行教师蒸馏学习。

The student model is a smaller Transformer encoder layer that accepts sequence data from the input side and data transferred from the Teacher model. It performs self-supervised encoding to reconstruct data representations, thereby learning the distillation of the abilities of the two input channels in the Teacher model.

学生模型是一个规模较小的Transformer编码器层，它接受来自输入端的序列数据和从Teacher模型传入的数据。通过自监督编码器重构数据表达量，实现对Teacher模型中两个输入通道能力的蒸馏学习。

Both the Teacher model and the Student model are self-supervised models. The Teacher model is more complex as it uses both sequence and graph encoders to extract features, while the Student model only includes a sequence encoder and learns the features extracted by multiple encoders from the Teacher.

这里的Teacher模型和Student模型都是自监督模型。老师模型使用了序列编码器和图编码器提取特征，因此更为复杂；学生模型仅包含序列编码器，通过学习老师模型传授的数据特征，掌握了从多个通道不同编码器提取特征的能力。
'''
import csv
import os
import pickle

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import anndata
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv, TransformerConv
from linformer import Linformer
import torch.optim as optim
from tqdm import tqdm

# import cupy as cp
# from scipy.sparse import csr_matrix
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义参数量统计函数
def count_parameters(model):
    """
    计算整个模型的参数总量以及每个子模块的参数量。

    返回:
        - total_params (int): 模型的总参数量
        - module_params (dict): 每个模块的参数量，键为模块名称，值为参数量
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    module_params = {}

    # 遍历模型的所有子模块
    for name, module in model.named_modules():
        # 计算当前模块的参数量
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params > 0:
            module_params[name] = params

    return total_params, module_params


# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# 定义闵可夫斯基损失函数
class MinkowskiLoss(nn.Module):
    def __init__(self, p=3):
        super(MinkowskiLoss, self).__init__()
        self.p = p

    def forward(self, x, x_hat):
        # 对每个样本计算闵可夫斯基距离
        loss_per_sample = torch.sum(torch.abs(x - x_hat) ** self.p, dim=1) ** (1 / self.p)
        # 对所有样本的损失取平均值
        return torch.mean(loss_per_sample)


# 定义编码器类
class LinformerSeqEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, seq_length=10000, num_heads=4, num_layers=1):
        super(LinformerSeqEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=seq_length)

        self.linformer = Linformer(
            dim=embed_dim,
            seq_len=seq_length,
            depth=num_layers,
            heads=num_heads,
            k=256  # 投影长度，可以根据需要调整
        )

        self.fc = nn.Linear(seq_length, seq_length // 2)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.float()
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.linformer(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        x = torch.sum(x, dim=2, keepdim=True)
        x = x.view(x.size(0), -1)

        return x


# 定义解码器类
class LinformerSeqDecoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(LinformerSeqDecoder, self).__init__()
        self.fc = nn.Linear(10000 // 2, 10000)

        self.linformer = Linformer(
            dim=embed_dim,
            seq_len=5000,
            depth=1,
            heads=4,
            k=256
        )

        self.positional_encoding = PositionalEncoding(embed_dim, max_len=10000)
        self.embedding = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.linformer(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        x = torch.sum(x, dim=2, keepdim=True)
        x = x.view(x.size(0), -1)

        return x


# 定义自编码器类
class LinformerSeqAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(LinformerSeqAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define Linformer graph Transformer
class LinformerGraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length=60, num_heads=4, num_layers=1):
        super(LinformerGraphTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=seq_length)

        self.linformer = Linformer(
            dim=hidden_dim,
            seq_len=seq_length,
            depth=num_layers,
            heads=num_heads,
            k=256  # Projection length, adjust as needed
        )

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        # Embedding
        x = self.embedding(x)

        # Positional encoding
        x = self.positional_encoding(x)

        # Linformer encoding
        x = self.linformer(x)

        # Dimensionality reduction
        x = self.fc(x)
        return x


# Define Linformer graph Transformer encoder
class LinformerGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length=60, num_heads=4, num_layers=1):
        super(LinformerGraphEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=seq_length)

        self.linformer = Linformer(
            dim=hidden_dim,
            seq_len=seq_length,
            depth=num_layers,
            heads=num_heads,
            k=256  # Projection length, adjust as needed
        )

    def forward(self, x):
        # Embedding
        x = self.embedding(x)

        # Positional encoding
        x = self.positional_encoding(x)

        # Linformer encoding
        x = self.linformer(x)

        return x


# Define Linformer graph Transformer decoder
class LinformerGraphDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, seq_length=60, num_heads=4, num_layers=1):
        super(LinformerGraphDecoder, self).__init__()
        self.linformer = Linformer(
            dim=hidden_dim,
            seq_len=seq_length,
            depth=num_layers,
            heads=num_heads,
            k=256  # Projection length, adjust as needed
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linformer decoding
        x = self.linformer(x)
        # Dimensionality reduction
        x = self.fc(x)
        return x


# Define graph autoencoder model using Linformer
class LinformerGraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LinformerGraphAutoencoder, self).__init__()
        self.encoder = LinformerGraphEncoder(input_dim, hidden_dim)
        self.decoder = LinformerGraphDecoder(hidden_dim, input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Encoder
        encoded = self.encoder(x)

        # Decoder
        decoded = self.decoder(encoded)

        return decoded


# 定义图自编码器的编码器部分
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_type='GIN', num_heads=4):
        super(GraphEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        self.num_heads = num_heads

        if gnn_type == 'GAT':
            self.encoder = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        elif gnn_type == 'GCN':
            self.encoder = GCNConv(input_dim, hidden_dim)
        elif gnn_type == 'GraphSAGE':
            self.encoder = SAGEConv(input_dim, hidden_dim)
        elif gnn_type == 'GIN':
            nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.encoder = GINConv(nn1)
        elif gnn_type == 'Linformer':
            self.encoder = LinformerGraphTransformer(input_dim, hidden_dim)
        elif gnn_type == 'TransformerConv':
            self.encoder = TransformerConv(input_dim, hidden_dim, heads=num_heads)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

    def forward(self, x, edge_index):
        edge_index = edge_index.long().view(2, -1)
        # edge_index = edge_index.type(torch.int64)
        x = x.float()
        x = self.encoder(x, edge_index)
        x = x[:, 0:1, :]
        x = x.view(x.shape[0], -1)

        x = F.relu(x)  # 应用激活函数
        return x


# 定义图自编码器的解码器部分
class GraphDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, gnn_type='GIN', num_heads=4):
        super(GraphDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gnn_type = gnn_type
        self.num_heads = num_heads

        if gnn_type == 'GAT':
            self.decoder = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)
        elif gnn_type == 'GCN':
            self.decoder = GCNConv(hidden_dim, output_dim)
        elif gnn_type == 'GraphSAGE':
            self.decoder = SAGEConv(hidden_dim, output_dim)
        elif gnn_type == 'GIN':
            nn2 = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
            self.decoder = GINConv(nn2)
        elif gnn_type == 'Linformer':
            self.decoder = LinformerGraphTransformer(hidden_dim, output_dim)
        elif gnn_type == 'TransformerConv':
            self.decoder = TransformerConv(hidden_dim, output_dim, heads=num_heads)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

    def forward(self, x, edge_index):
        edge_index = edge_index.long().view(2, -1)
        x = self.decoder(x, edge_index)
        x = x[:, 0:1, :]
        x = x.view(x.shape[0], -1)
        x = F.relu(x)  # 应用激活函数
        return x


# 定义图自编码器模型
class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_type='GAT', num_heads=4):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, gnn_type, num_heads)
        self.decoder = GraphDecoder(hidden_dim, input_dim, gnn_type, num_heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        return x


# 全局学习层 - Linformer Model
# Teacher Model - Global Feature Learner
class TeacherModel(nn.Module):
    def __init__(self, input_dim, embed_dim, seq_length=10000, num_heads=4, num_layers=3):
        super(TeacherModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=seq_length)

        self.linformer = Linformer(
            dim=embed_dim,
            seq_len=seq_length,
            depth=num_layers,
            heads=num_heads,
            k=256  # Projection length, adjust as needed
        )

        self.fc = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.linformer(x)
        x = self.fc(x)
        x = torch.sum(x, dim=2, keepdim=True)
        x = x.view(x.size(0), -1)

        return x  # Return two identical outputs


# Student Model
class StudentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length=10000, num_heads=4, num_layers=1):
        super(StudentModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=seq_length)

        self.linformer = Linformer(
            dim=hidden_dim,
            seq_len=seq_length,
            depth=num_layers,
            heads=num_heads,
            k=256  # Projection length, adjust as needed
        )

        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.linformer(x)
        x = self.fc(x)
        x = torch.sum(x, dim=2, keepdim=True)
        x = x.view(x.size(0), -1)

        return x


class scTCHCN(nn.Module):
    def __init__(self, seq_input_dim, seq_embed_dim, seq_length=10000, seq_num_heads=4, seq_num_layers=2,
                 graph_input_dim=250, graph_embed_dim=10000, teacher_input_dim=10000, teacher_embed_dim=10000,
                 student_input_dim=10000, student_embed_dim=5000, graph_output_dim=10000):
        super(scTCHCN, self).__init__()

        self.linformerSeqEncoder = LinformerSeqEncoder(
            input_dim=seq_input_dim,
            embed_dim=seq_embed_dim,
            seq_length=seq_length,
            num_heads=seq_num_heads,
            num_layers=seq_num_layers
        )

        self.linformerSeqDecoder = LinformerSeqDecoder(
            input_dim=seq_input_dim,
            embed_dim=seq_embed_dim
        )
        self.GraphEncoder = GraphEncoder(graph_input_dim, graph_embed_dim)
        self.GraphDecoder = GraphDecoder(graph_input_dim, graph_output_dim)

        self.teacherModel = TeacherModel(teacher_input_dim, teacher_embed_dim)
        self.studentModel = StudentModel(student_input_dim, student_embed_dim)

    def forward(self, x_seq, x_graph, adj):
        seq_encoder = self.linformerSeqEncoder(x_seq)
        graph_encoder = self.GraphEncoder(x_graph, adj)
        concat_seq_graph = torch.cat((seq_encoder, graph_encoder), dim=1)
        teacher_feature = self.teacherModel(concat_seq_graph)
        split_seq = teacher_feature[:, :5000]
        split_graph = teacher_feature[:, 5000:]
        sum_xseq_teacher = teacher_feature * 0.5 + x_seq * 0.5
        student_feature = self.studentModel(sum_xseq_teacher)
        seq_decoder = self.linformerSeqDecoder(split_seq)
        split_graph = split_graph.view(split_graph.shape[0], 20, 250)
        graph_decoder = self.GraphDecoder(split_graph, adj)
        return teacher_feature, student_feature, seq_decoder, graph_decoder

    def save_models(self, teacher_path, student_path):
        torch.save(self.teacherModel.state_dict(), teacher_path)
        torch.save(self.studentModel.state_dict(), student_path)


import scipy.sparse as sp


class CustomDataset(Dataset):
    def __init__(self, x_seq_path, feature_adj_path):
        # 读取数据
        self.x_seq = anndata.read_h5ad(x_seq_path).X
        self.x_seq = torch.tensor(sp.coo_matrix(self.x_seq).toarray())
        self.x_seq = self.x_seq[:, :10000]
        with open(feature_adj_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # 现在 loaded_data 中包含了之前保存的所有数据
        loaded_feature = loaded_data['features']
        loaded_adj = loaded_data['adj']
        self.x_graph = loaded_feature
        self.adj = loaded_adj

    def __len__(self):
        # 返回数据集的长度
        return len(self.x_seq)  # 假设以 x_seq 的长度为准

    def __getitem__(self, index):
        # 根据索引返回数据样本
        sample = {
            'x_seq': self.x_seq[index],
            'x_graph': self.x_graph[index],
            'adj': self.adj[index]
        }
        return sample


def main():
    # Define your model parameters
    seq_input_dim = 1  # Example sequence input dimension
    seq_embed_dim = 200  # Example sequence embedding dimension
    graph_input_dim = 250
    graph_embed_dim = 5000
    teacher_input_dim = 1
    teacher_embed_dim = 200
    student_input_dim = 1
    student_embed_dim = 200
    batch_size = 12
    graph_output_dim = 5000
    # Training loop
    num_epochs = 500
    best_loss = float('inf')
    max_no_improvement = 30  # 最大连续未改善epoch的数量

    # 文件夹路径
    base_path = '/private/litianhao/scTCHCN/data/processed/'
    model_path = '/private/litianhao/scTCHCN/data/'

    folders = []
    file_path = 'no_folder_name.csv'
    # 打开文件
    with open(file_path, 'r', newline='') as file:
        # 创建 CSV 读取器
        csv_reader = csv.reader(file)
        # 跳过第一行（列名）
        next(csv_reader)
        # 读取剩余的每一行数据
        for row in csv_reader:
            folders.extend(row)

    model = scTCHCN(
        seq_input_dim, seq_embed_dim, graph_input_dim=graph_input_dim, graph_embed_dim=graph_embed_dim,
        teacher_input_dim=teacher_input_dim, teacher_embed_dim=teacher_embed_dim,
        student_input_dim=student_input_dim, student_embed_dim=student_embed_dim, graph_output_dim=graph_output_dim
    ).to(device)

    # 计算参数量
    total_params, module_params = count_parameters(model)
    print(f"Total parameters in the model: {total_params}")

    # 打印每个模块的参数量
    # for module_name, params in module_params.items():
    #     print(f"Parameters in module '{module_name}': {params}")

    read_model_path = os.path.join(model_path, 'model.pkl')
    model.load_state_dict(torch.load(read_model_path))

    # 遍历文件夹
    for folder in folders:
        print("begin_with_" + str(folder))
        no_improvement_count = 0
        x_seq_path = os.path.join(base_path, folder, f'{folder}_hvg_10k_1e4.h5ad')
        feature_adj_path = os.path.join(base_path, folder, 'feature_adj.pkl')

        # Create dataset and dataloader
        dataset = CustomDataset(x_seq_path, feature_adj_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

        # Define loss functions
        criterion1 = nn.MSELoss().to(device)
        criterion2 = nn.MSELoss().to(device)
        criterion3 = nn.MSELoss().to(device)
        criterion_minkowski = MinkowskiLoss(p=50).to(device)
        # criterion4 = nn.MSELoss().to(device)
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in tqdm(dataloader, desc='train_loader_epoch_{}'.format(epoch), leave=True):
                x_seq_batch = batch['x_seq'].to(device)
                x_graph_batch = batch['x_graph'].to(device)
                adj_batch = batch['adj'].to(device)

                x_seq_batch_view = x_seq_batch.view(x_seq_batch.size(0), -1)
                # 训练
                teacher_feature, student_feature, seq_decoder, graph_decoder = model(x_seq_batch, x_graph_batch,
                                                                                     adj_batch)

                x_graph_batch_view = x_graph_batch.view(x_graph_batch.size(0), -1)
                graph_decoder = graph_decoder.view(graph_decoder.size(0), -1)

                student_feature = student_feature.to(torch.float32)
                teacher_feature = teacher_feature.to(torch.float32)
                graph_decoder = graph_decoder.to(torch.float32)
                x_graph_batch_view = x_graph_batch_view.to(torch.float32)
                seq_decoder = seq_decoder.to(torch.float32)
                x_seq_batch_view = x_seq_batch_view.to(torch.float32)

                loss1 = criterion1(student_feature, teacher_feature)  # 学生输出与老师输出
                loss2 = criterion2(seq_decoder, x_seq_batch_view)  # 原始序列和经过序列Decoder后的输出
                loss3 = criterion3(graph_decoder, x_graph_batch_view)  # 原始图数据和经过图Decoder后的输出
                loss_minkowski = criterion_minkowski(seq_decoder, student_feature)

                loss = loss1 + loss2 + loss3 + loss_minkowski
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

            # 保存模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improvement_count = 0
                model.save_models(
                    os.path.join(model_path, f'teacher_model_best.pt'),
                    os.path.join(model_path, f'student_model_best.pt')
                )
            else:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    print("Early stopping")
                    break


if __name__ == '__main__':
    main()
