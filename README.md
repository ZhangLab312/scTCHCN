# Code Structure Overview

## 代码结构概述

This repository implements a self-supervised distillation learning model framework named scTCHCN. It mainly includes two models: TeacherModel and StudentModel, as well as an autoencoder architecture for sequence and graph data processing. The following is a detailed description of the structure and functionalities:

该代码实现了一个名为 scTCHCN 的自监督蒸馏学习模型框架。其主要包括两个模型：教师模型（TeacherModel）和学生模型（StudentModel），以及一个包含序列和图数据处理的自编码器架构。以下是详细的结构和功能说明：

## 1. Model Architecture
## 1. 模型架构

- **scTCHCN**: This is the main model class, which contains the initialization and forward propagation of all submodules.
  - **scTCHCN**: 这是主要的模型类，包含了所有子模块的初始化和前向传播过程。

- **LinformerSeqEncoder**: An encoder for sequence data, utilizing the Linformer model for sequence feature extraction.
  - **LinformerSeqEncoder**: 用于处理序列数据的编码器，利用Linformer模型进行序列特征提取。

- **LinformerSeqDecoder**: A decoder for sequence data.
  - **LinformerSeqDecoder**: 用于序列数据解码的解码器。

- **GraphEncoder**: An encoder for graph data, supporting various types of Graph Neural Networks (GNNs).
  - **GraphEncoder**: 处理图数据的编码器，支持多种图神经网络（GNN）类型。

- **GraphDecoder**: A decoder for graph data, supporting various types of GNNs.
  - **GraphDecoder**: 图数据的解码器，支持多种GNN类型。

- **TeacherModel**: The teacher model, which uses Linformer for global feature learning.
  - **TeacherModel**: 教师模型，利用Linformer进行全局特征学习。

- **StudentModel**: The student model, which learns the features transferred from the teacher model through self-supervision.
  - **StudentModel**: 学生模型，通过自监督机制学习教师模型传递的特征。

- **count_parameters**: Computes the total number of parameters in the model and the parameter count for each submodule.
  - **count_parameters**: 计算模型中所有参数的总量和每个子模块的参数量。

- **PositionalEncoding**: Defines positional encoding to enhance the performance of sequence models.
  - **PositionalEncoding**: 定义位置编码，用于提升序列模型的表现。

- **CustomDataset**: A custom dataset class for loading sequence data and graph adjacency matrices.
  - **CustomDataset**: 自定义数据集类，用于加载序列数据和图数据的邻接矩阵。

## 2. Model Functionality
## 2. 模型功能

- **scTCHCN Model Forward Propagation**:
  - Processes sequence and graph data.
  - Concatenates features from both data sources and performs global feature learning using the teacher model.
  - Splits the teacher model's output features into sequence features and graph features.
  - Performs self-supervised learning using the student model.
  - The decoder is used to reconstruct sequence data and graph data.
  - **scTCHCN 模型的前向传播**:
    - 处理序列数据和图数据。
    - 将两个数据源的特征进行拼接，并通过教师模型进行全局特征学习。
    - 将教师模型的输出特征分为序列特征和图特征。
    - 通过学生模型进行自监督学习。
    - 解码器用于重构序列数据和图数据。

- **TeacherModel**:
  - A teacher model for global feature learning, outputting two identical feature representations.
  - **TeacherModel**:
    - 用于全局特征学习的教师模型，输出两个相同的特征表示。

- **StudentModel**:
  - A student model that learns features from the teacher model.
  - **StudentModel**:
    - 用于从教师模型学习特征的学生模型。

## 3. Training Process
## 3. 训练流程

- **Data Preparation**:
  - The CustomDataset class is used to load and process sequence and graph data.
  - DataLoader is used for batching the data.
  - **数据准备**:
    - CustomDataset 类用于加载和处理序列数据和图数据。
    - 使用DataLoader对数据进行批处理。

- **Training Loop**:
  - Computes and prints model parameters.
  - Loads pre-trained models (if available).
  - Trains the model, computes loss, and uses Adam optimizer and learning rate scheduler for training.
  - **训练循环**:
    - 计算和打印模型参数。
    - 加载预训练模型（如果有的话）。
    - 训练模型并计算损失，使用Adam优化器和学习率调度器进行训练。

## 4. Utility Functions
## 4. 功能函数

- **save_models**: Saves the state dictionaries of the teacher and student models.
  - **save_models**: 保存教师模型和学生模型的状态字典。

- **count_parameters**: Computes and returns the total number of model parameters and the parameter count for each module.
  - **count_parameters**: 计算并返回模型参数的总量及每个模块的参数量。

## Summary
## 总结

The entire code implements a composite self-supervised learning model that integrates sequence and graph data processing. It uses a distillation mechanism with teacher and student models to enhance performance. The framework supports various graph neural network architectures and utilizes the Linformer model for sequence feature extraction and reconstruction.

整个代码实现了一个复合的自监督学习模型，结合了序列数据和图数据处理，通过教师模型和学生模型的蒸馏机制，旨在提高模型的性能。该框架支持多种图神经网络架构，并通过Linformer模型处理序列数据的特征提取和重构。
