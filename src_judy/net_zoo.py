# @Author: Xi He <Heerye>
# @Date:   2018-03-01T11:44:45-05:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: net_zoo.py
# @Last modified by:   Heerye
# @Last modified time: 2018-03-03T09:07:04-05:00

import torch
import torch.nn.functional as F

# class Net(torch.nn.Module):
#
#     def __init__(self, n_features, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_features, n_hidden)
#         self.out = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         # x = F.sigmoid(self.hidden(x))
#         x = self.out(x)
#         return x

class Net(torch.nn.Module):

    def __init__(self, layer_list):
        super(Net, self).__init__()
        layers = []
        n = len(layer_list)
        for i in range(len(layer_list)-2):
            # layers += [torch.nn.Linear(layer_list[i], layer_list[i+1]), torch.nn.ReLU(inplace=True)]
            layers += [torch.nn.Linear(layer_list[i], layer_list[i+1]), torch.nn.ReLU()]
            # layers += [torch.nn.Linear(layer_list[i], layer_list[i+1]), torch.nn.Sigmoid()]
        self.hidden = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(layer_list[n-2], layer_list[n-1])

        # for m in self.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         m.weight.data.normal_(0, 0.1)
        #         m.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    # net = HardNet([2,4,6,5])
    net = Net(2, 3, 5)
    print(net)
