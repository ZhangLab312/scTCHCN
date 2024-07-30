import numpy as np
from scipy.io import mmread
import scanpy as sc
import os
from scipy import sparse
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp


def process_mtx_file(file_path, output_root):
    # 读取mtx文件
    matrix = mmread(file_path)

    # 提取文件前缀
    file_prefix = os.path.splitext(os.path.basename(file_path))[0]

    # 将矩阵转换为稀疏格式
    matrix_sparse = sparse.csr_matrix(matrix.T)

    print(f'提取了{file_path}，转化为稀疏矩阵，shape：', matrix_sparse.shape)

    # 创建一个AnnData对象
    adata = sc.AnnData(X=matrix_sparse)

    # 筛选低表达基因，假设我们使用每个基因至少有一个细胞表达
    sc.pp.filter_genes(adata, min_cells=1)

    # 对数正则化
    sc.pp.log1p(adata)

    print('对数正则化完毕，开始筛选高变异基因')

    # 选择高变异基因（HVGs）
    sc.pp.highly_variable_genes(adata, n_top_genes=10000)
    adata_hvg_10k = adata[:, adata.var['highly_variable']].copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=250)
    adata_hvg_250 = adata[:, adata.var['highly_variable']].copy()

    # 归一化到1e4和1e3
    sc.pp.normalize_total(adata_hvg_10k, target_sum=1e4)
    sc.pp.normalize_total(adata_hvg_250, target_sum=1e3)

    print('高变异基因筛选完毕了')

    # 创建保存目录
    output_dir = os.path.join(output_root, file_prefix)
    os.makedirs(output_dir, exist_ok=True)

    # 保存为h5ad文件，使用稀疏格式
    adata_hvg_10k.write_h5ad(f'{output_dir}/{file_prefix}_hvg_10k_1e4.h5ad')
    adata_hvg_250.write_h5ad(f'{output_dir}/{file_prefix}_hvg_250_1e3.h5ad')

    print('保存h5ad文件完毕，开始计算图相关性数据')

    ######################################################################
    # 计算需要用到的图数据

    # 读取包含 PCA 特征的 .h5ad 文件
    adata_pca = sc.read_h5ad(f'{output_dir}/{file_prefix}_hvg_250_1e3.h5ad')

    # 提取 PCA 矩阵
    pca_matrix = adata_pca.X.A

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(pca_matrix)

    # 将相关系数转换为相似度
    similarity_matrix = (1 + similarity_matrix) / 2

    # 设置对角线元素为1
    np.fill_diagonal(similarity_matrix, 1)

    # 提取相似度矩阵的上三角部分（包括对角线）
    tri_upper_indices = np.triu_indices(similarity_matrix.shape[0])
    tri_upper_values = similarity_matrix[tri_upper_indices]

    # 将上三角部分转换为稀疏矩阵
    sparse_similarity_matrix = sp.coo_matrix((tri_upper_values, tri_upper_indices), shape=similarity_matrix.shape)

    # 加载稀疏矩阵
    sparse_similarity_matrix = sparse_similarity_matrix.tocsr()  # 转换为CSR格式以便快速行访问

    print("Upper triangular similarity matrix saved to NPZ file with sparse storage and compression.")

    # 初始化一个矩阵来存储每个细胞的前19个最相关细胞的索引
    n_cells = sparse_similarity_matrix.shape[0]
    top_k_indices = np.zeros((n_cells, 19), dtype=np.int)

    # 遍历每个细胞
    for i in tqdm(range(n_cells)):
        # 获取当前细胞的相关系数
        row = sparse_similarity_matrix.getrow(i)
        # 获取相关系数的非零元素索引和值
        nonzero_indices = row.indices
        nonzero_values = row.data

        # 排除自身的索引
        mask = nonzero_indices != i
        nonzero_indices = nonzero_indices[mask]
        nonzero_values = nonzero_values[mask]

        # 找出前19个相关性最高的细胞的索引
        if len(nonzero_values) >= 19:
            top_indices = nonzero_indices[np.argsort(-nonzero_values)[:19]]
        else:
            top_indices = nonzero_indices[np.argsort(-nonzero_values)]

        # 填充到结果矩阵
        top_k_indices[i, :len(top_indices)] = top_indices

    # 将top_k_indices矩阵保存为npz文件
    np.savez(f'{output_dir}/top_correlated_19_cells.npz', top_k_indices=top_k_indices)

    # 获取top_k_indices矩阵
    top_k_indices_data = np.load(f'{output_dir}/top_correlated_19_cells.npz')
    top_k_indices = top_k_indices_data['top_k_indices']

    # 初始化一个三维张量来存储所有20x20的相关性矩阵
    similarity_tensor = np.zeros((n_cells, 20, 20))

    # 为每个细胞构建新的20x20相关性矩阵
    for idx in tqdm(range(n_cells)):
        cell_index = idx

        # 获取指定细胞及其最相关的19个细胞的索引
        relevant_indices = top_k_indices[cell_index]
        relevant_indices = np.insert(relevant_indices, 0, cell_index)  # 插入自身索引

        # 初始化一个20x20的相关性矩阵
        similarity_matrix_20x20 = np.zeros((20, 20))

        # 填充相关性矩阵
        for i in range(20):
            for j in range(i, 20):  # 仅填充上三角矩阵
                idx_i = relevant_indices[i]
                idx_j = relevant_indices[j]
                similarity_matrix_20x20[i, j] = sparse_similarity_matrix[idx_i, idx_j]
                similarity_matrix_20x20[j, i] = similarity_matrix_20x20[i, j]  # 矩阵对称填充

        # 将生成的20x20矩阵存储到三维张量中
        similarity_tensor[idx, :, :] = similarity_matrix_20x20

    # 输出新的三维张量
    print(similarity_tensor.shape)

    # 将三维张量保存为npz文件
    np.savez(f'{output_dir}/cells_similarity_sub_matrix.npz', similarity_tensor=similarity_tensor)

    print('提取了子图数据完毕')


def process_all_mtx_files(input_dir, output_root):
    mtx_files = [f for f in os.listdir(input_dir) if f.endswith('.mtx')]
    for file_name in tqdm(mtx_files):
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing file: {file_path}")
        process_mtx_file(file_path, output_root)


# 示例调用
input_directory = '../../data/raw'
output_directory = '../../data/processed'

process_all_mtx_files(input_directory, output_directory)
