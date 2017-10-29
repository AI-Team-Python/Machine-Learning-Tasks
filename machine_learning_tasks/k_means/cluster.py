# @Author  : lzq
# @Date    : 2017/10/27 17:47
# @File    : cluster
"""
转载自博客 http://blog.csdn.net/eventqueue/article/details/73133617
"""
import numpy as np
import matplotlib.pyplot as plt
import random

# 设置随机数seed，确保数据重现
random.seed(1024)


# 获取训练数据 80个样本，4个类
def get_data():
    with open('dataset/data', mode='r', encoding='utf-8') as f_read:
        data = f_read.readlines()
        data = [item.split() for item in data]
        data = map(lambda x: [float(x[0].strip()), float(x[1].strip())], data)
        data = list(data)
        return np.array(data)


# 计算两个点之间的距离，这里采用欧式距离，度量指标具体根据业务来选择
def compute_distance(vec1, vec2):
    return np.sqrt(sum(pow(vec2 - vec1, 2)))


# 用随机样本初始化中心点
def init_centroids(data, k):
    num_samples, dim = data.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, num_samples))
        centroids[i, :] = data[index, :]

    return centroids


# 计算损失
def get_loss(cluset_assment):
    length = cluset_assment.shape[0]
    total_loss = 0
    for i in range(length):
        total_loss += cluset_assment[i, 1]

    return total_loss / length


# 主程序
def _kmeans(data, k):
    num_samples = data.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    cluster_assment = np.mat(np.zeros((num_samples, 2)))
    # 未收敛之前，不断迭代优化 收敛两种情况：要嘛 质心不变 要嘛 总损失小于某个阈值
    cluster_change = True

    # 初始化中心点
    centroids = init_centroids(data, k)

    while cluster_change:
        cluster_change = False
        for i in range(num_samples):
            min_dist = 10000000.0
            min_index = 0

            # 对每个样本计算相应的簇,选择最小的簇
            for j in range(k):
                distance = compute_distance(centroids[j, :], data[i, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j

            # step 3: 更新样本点与中心点的分配关系,所有样本点，只要有一个质心发生变化，就必须重新分布
            if int(cluster_assment[i, 0]) != int(min_index):
                cluster_change = True
                cluster_assment[i, :] = min_index, min_dist
            else:
                cluster_assment[i, 1] = min_dist

        # step 4: 更新样本中心
        for j in range(k):
            # 选出每一簇的所有点，进行质心重新计算  .A是对mat返回np.array
            pointsInCluster = data[np.nonzero(cluster_assment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

        print('Congratulations, cluster complete!')
        return centroids, cluster_assment


# 以2D形式可视化数据
def show_data_cluster(data, k, centroids, cluster_assment):
    num_samples, dim = data.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return

    # 颜色定义数量
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large!")
        return 1

    # 绘制所有非中心样本点
    for i in range(num_samples):
        mark_index = int(cluster_assment[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[mark_index])

    # 绘制中心点
    mark2 = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark2[i], markersize=12)

    plt.show()

# sklearn 画图
def show_data_cluster2(data, centroids):
    num_samples, dim = data.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return

    # 颜色定义数量
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large!")
        return 1

    # 绘制所有非中心样本点
    for i in range(num_samples):
        plt.plot(data[i, 0], data[i, 1], mark[0])

    # 绘制中心点
    mark2 = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    i = 0
    for item in centroids:
        plt.plot(item[0], item[1], mark2[i], markersize=12)
        i += 1

    plt.show()


# 用肘部法则来确定最佳的K值
def elbow_k(k, data):
    cluster_assments = []
    k_range = range(1, k)
    for i in k_range:
        centroid, cluster_assment = _kmeans(data, i)
        cluster_assments.append(get_loss(cluster_assment))

    plt.plot(k_range, cluster_assments, 'bx-')
    plt.xlabel('k')
    plt.ylabel('平均畸变程度')
    plt.title('用肘部法则来确定最佳的K值')
    plt.show()


# 主程
def main():
    # 获取数据
    data = get_data()

    # 开始聚类
    k = 4
    centroids, cluster_assment = _kmeans(data, k)

    # 获取损失
    print('loss: ')
    print(get_loss(cluster_assment))

    # 显示结果
    show_data_cluster(data, k, centroids, cluster_assment)

    elbow_k(10, data)


if __name__ == "__main__":
    main()

    # 调用sklearn包实现
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist

    data = get_data()
    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        meandistortions.append(sum(np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])


    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('平均畸变程度')
    plt.title('用肘部法则来确定最佳的K值')
    plt.show()

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    show_data_cluster2(data, centroids)