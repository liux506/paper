import argparse

from recbole_gnn.quick_start import run_recbole_gnn
from recbole_gnn.quick_start import load_data_and_model
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MHCN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='douban-book', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    '''
    config, model, dataset, train_data, valid_data, test_data=load_data_and_model(model_file='/root/xiao/RecBole-social 3 (1)/saved/MHCN-Jan-29-2024_15-52-12.pth')
    for param in model.state_dict():
        if param =='user_embedding.weight':
            user_embeddings=model.state_dict()[param].cpu().numpy()
        if param =='item_embedding.weight':
            item_embeddings=model.state_dict()[param].cpu().numpy()
    item_num = np.size(item_embeddings, 0)
    indices = torch.randint(0, item_num - 1, [15000])  # 随机选取6000个物品embedding进行降维可视化
    item_embeddings_sample = torch.tensor(item_embeddings)
    item_embeddings_sample = torch.index_select(item_embeddings_sample, 0, indices)

    # t-SNE拟合和转换
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_obj = tsne.fit_transform(item_embeddings_sample)

    # 创建DataFrame存储t-SNE结果以及标签
    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                            'Y': tsne_obj[:, 1],
                            'labels': 'Item'})  # 所有的标签都是'Item'

    # 生成散点图
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x="X", y="Y", hue="labels", palette="viridis", size="labels", alpha=1.0, data=tsne_df)
    plt.legend().set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=24, width=4)
    plt.grid(True, linewidth=0.5, color='grey')
    # plt.tight_layout()
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig("mhcn_book.svg", format='svg', bbox_inches='tight')
    plt.show()'''



