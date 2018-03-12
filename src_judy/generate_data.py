############

#   @File name: generate_data.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-03-01 09:33:43

# @Last modified by:   Heerye
# @Last modified time: 2018-03-05T21:57:45-05:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############
import torch
import matplotlib
import matplotlib.pyplot as plt


import numpy
from settings import data_index, torch_seed, numpy_seed, train_ratio

numpy.random.seed(numpy_seed)
torch.manual_seed(torch_seed)

def generate_data(n_points, n_classes, center_label_list):

    x_axis = torch.ones(n_points, 1)
    y_axis = torch.ones(n_points, 1)

    x, y = {}, {}
    labels = []

    for idx, center_label in enumerate(center_label_list):

        a, b, c, d = center_label
        labels.append(c)

        std = numpy.random.random() * 0.5 + 0.5
        x[idx] = torch.normal(torch.cat((x_axis * a, y_axis * b), 1), d * std)
        y[idx] = torch.ones(n_points) * c

    x = torch.cat(x.values(), 0).type(torch.FloatTensor)
    y = torch.cat(y.values(), ).type(torch.LongTensor)

    perm = torch.randperm(len(y))

    wall = int(train_ratio*len(y))

    train_x, train_y = x[perm[:wall]], y[perm[:wall]]
    test_x, test_y = x[perm[wall:]], y[perm[wall:]]

    assert len(set(labels)) == n_classes, "labels not enough!"

    F = 2

    return train_x, train_y, test_x, test_y, F, len(set(labels)), len(y)


if data_index == 0:
    # (x1, x2), label, std
    n_classes = 2
    std = 0.8
    center_label_list = [[1, 1, 0, std],
                         [2, 2, 1, std]]

elif data_index == 1:
    # (x1, x2), label, std
    n_classes = 4
    std = 0.5
    center_label_list = [[1, 1, 0, std],
                         [2, 2, 1, std],
                         [1, 2, 2, std],
                         [2, 1, 3, std]]
elif data_index == 2:
    n_clusters = 12
    n_classes = 5
    center_label_list = [[]] * n_clusters
    for i in range(n_clusters):
        std = 1.0/n_clusters
        label = numpy.random.randint(0, n_classes)
        x1 = numpy.random.randint(0, n_clusters) / n_clusters
        x2 = numpy.random.randint(0, n_clusters) / n_clusters
        center_label_list[i] = [x1, x2, label, std]

elif data_index == 3:
    n_clusters = 40
    n_classes = 10
    center_label_list = [[]] * n_clusters
    for i in range(n_clusters):
        std = 1.5 / n_clusters
        label = numpy.random.randint(0, n_classes)
        x1 = numpy.random.randint(0, n_clusters) / n_clusters
        x2 = numpy.random.randint(0, n_clusters) / n_clusters
        center_label_list[i] = [x1, x2, label, std]
else:
    print('invalid data!!')

if __name__ == '__main__':

    from settings import args, use_cuda

    n_points = 300
    x, y, t_x, t_y, F, c, n = generate_data(n_points, n_classes, center_label_list)

    fig = plt.figure(1)
    fig.subplots_adjust(bottom=0.13, left=0.16)
    plt.style.use('seaborn-whitegrid')

    if use_cuda:
        plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[
                    :, -1], c=y.cpu().numpy(), s=50, alpha=0.3, cmap='rainbow')
    else:
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, -1],
                    c=y.numpy(), s=50, alpha=0.3, cmap='rainbow')
    plt.title('Training, number of labels: %d' % c, fontsize=15)
    plt.savefig('images/data-' + str(data_index) +
                '.png', format='png', dpi=1000)
    # plt.show()
    # plt.close()

    fig = plt.figure(2)
    fig.subplots_adjust(bottom=0.13, left=0.16)
    plt.style.use('seaborn-whitegrid')

    if use_cuda:
        plt.scatter(t_x.cpu().numpy()[:, 0], t_x.cpu().numpy()[
                    :, -1], c=t_y.cpu().numpy(), s=50, alpha=0.3, cmap='rainbow')
    else:
        plt.scatter(t_x.numpy()[:, 0], t_x.numpy()[:, -1],
                    c=t_y.numpy(), s=50, alpha=0.3, cmap='rainbow')
    plt.title('Testing, number of labels: %d' % c, fontsize=15)
    plt.savefig('images/tdata-' + str(data_index) +
                '.png', format='png', dpi=1000)
    plt.show()
    plt.close()
