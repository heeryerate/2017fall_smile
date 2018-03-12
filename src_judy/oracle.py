# @Author: Xi He <Heerye>
# @Date:   2018-03-01T11:49:04-05:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: oracle.py
# @Last modified by:   Heerye
# @Last modified time: 2018-03-03T11:49:38-05:00

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, grad

import numpy
from numpy import linalg as LA

from scipy.stats import ortho_group

from settings import args, use_cuda


def get_params(net):
    return net.parameters()


def get_loss(net, x, y):
    fx = net(x)
    loss = torch.nn.CrossEntropyLoss()(fx, y)
    # loss = torch.nn.MSELoss()(fx, y)
    return loss


def get_attr(net):
    params = get_params(net)

    count = [0]
    for p in params:
        count.append(count[-1] + p.numel())

    orths = torch.from_numpy(ortho_group.rvs(count[-1]))
    eyes = torch.eye(count[-1])

    return count, orths, eyes


def get_flatness(net, x, y, eps, count, orths):

    params_snap = [para.data.clone() for para in net.parameters()]

    if use_cuda:
        tmp = 0.0
        for para in net.parameters():
            tmp += LA.norm(para.data.cpu().numpy())
    else:
        tmp = 0.0
        for para in net.parameters():
            tmp += LA.norm(para.data.numpy())

    w = numpy.empty((count[-1], ))
    for i, v in enumerate(orths):
        v = v * eps
        if use_cuda:
            tensor_v = [Variable(v[ind:ind + para.numel()].resize_(para.size())).cuda().type(
                torch.cuda.FloatTensor) for ind, para in zip(count[:-1], net.parameters())]
        else:
            tensor_v = [Variable(v[ind:ind + para.numel()].resize_(para.size())).type(
                torch.FloatTensor) for ind, para in zip(count[:-1], net.parameters())]

        for para, v_para in zip(net.parameters(), tensor_v):
            para.data.add_(v_para.data)

        if use_cuda:
            w[i] = get_loss(net, x, y).data.cpu().numpy()
        else:
            w[i] = get_loss(net, x, y).data.numpy()

        for para, para_snap in zip(net.parameters(), params_snap):
            para.data.copy_(para_snap)

    return w


def get_grad(net, x, y):
    params = get_params(net)
    loss = get_loss(net, x, y)
    g = torch.autograd.grad(loss, params, create_graph=True)
    return g


def get_Hv(net, grad, v):
    params = get_params(net)
    gv = 0.0
    for g_para,  v_para in zip(grad, v):
        gv += (g_para * v_para).sum()
    hv = torch.autograd.grad(gv, params, create_graph=True)
    return hv


def get_eig(net, grad, count, eyes):

    hess = numpy.empty((count[-1], count[-1]))
    for i, v in enumerate(eyes):
        if use_cuda:
            tensor_v = [Variable(v[ind:ind + para.numel()].resize_(para.size())).cuda()
                        for ind, para in zip(count[:-1], get_params(net))]
        else:
            tensor_v = [Variable(v[ind:ind + para.numel()].resize_(para.size()))
                        for ind, para in zip(count[:-1], get_params(net))]

        tensor_hv = get_Hv(net, grad, tensor_v)
        if use_cuda:
            hess[:, i] = torch.cat([para.contiguous().view(
                para.numel()) for para in tensor_hv]).data.cpu().numpy()
        else:
            hess[:, i] = torch.cat([para.contiguous().view(
                para.numel()) for para in tensor_hv]).data.numpy()

    # print('Hessian error: %f'%LA.norm(hess - numpy.transpose(hess)))
    hess = (hess + numpy.transpose(hess)) / 2

    w, _ = LA.eig(hess)
    return w


def get_acc(net, x, y):
    prediction = torch.max(F.softmax(net(x)), 1)[1]
    if use_cuda:
        pred_y = prediction.data.cpu().numpy().squeeze()
        target_y = y.data.cpu().numpy()
    else:
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
    accuracy = sum(pred_y == target_y) / len(y)
    return accuracy
