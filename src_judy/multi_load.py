# @Author: Xi He <Heerye>
# @Date:   2018-02-23T04:18:44-05:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: load.py
# @Last modified by:   Heerye
# @Last modified time: 2018-03-04T10:30:05-05:00

import pickle
from utils import Worker, Measure

import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
import sys
from brokenaxes import brokenaxes

from settings import data_index


def mylog(v):
    return 0 if v == 0 else numpy.log(v)


def symlog(v):
    return [mylog(i) if i > 0 else 0 for i in v]


def neg_symlog(v):
    return [mylog(-i) if i < 0 else 0 for i in v]


def rename(s):
    return s.replace('_', '-').replace('.', '-') + '.eps'


gap = 100


def make_plots(log_path='./logs'):

    # file_list = [file for file in os.listdir(log_path) if file.endswith(
        # '.pkl') and 'log_3_0.01_32_0' in file]
    batch = 1
    if batch:
        file_list = ['log_3_0.01_1000_0.pkl', 'log_3_0.01_1000_1.pkl', 'log_3_0.01_1000_2.pkl','log_3_0.01_32_0.pkl']
    else:
        file_list = ['log_3_0.01_1000_0.pkl', 'log_3_0.01_1000_1.pkl', 'log_3_0.01_1000_2.pkl','log_3_0.001_1000_0.pkl']


    for x_t in ['epoch']:
        for y_t in ['loss', 'grad', 'acc']:
            fig = plt.figure(1)
            fig.subplots_adjust(bottom=0.12, left=0.16)
            plt.style.use('seaborn-whitegrid')
            bax = brokenaxes(xlims=((0, 20000/gap), (50000/gap,70000/gap)), hspace=.05)

            a = []
            b = []
            tb = []
            for file in file_list:
                print('Handle %s' % file)
                with open(log_path + '/' + file, 'rb') as f:
                    MD = pickle.load(f)

                MD.labels['epoch'] = 'epoch $(k \\times 10^{-2})$'

                md = MD.dict

                # print(len(a), len(b), len(md[x_t]), len(md[y_t]))
                b = b + list(md[y_t])
                if y_t == 'loss' or y_t == 'acc':
                    tb = tb + list(md['t_'+y_t])

            a = range(len(b))

            ga = list(a[::gap]) + [a[-1]]
            ga = [i//gap for i in ga]

            gb = list(b[::gap]) + [b[-1]]

            if y_t == 'acc':
                gb = 1.0 - numpy.array(gb)

            # print(len(ga), len(gb))
            # print(ga, gb)


            bax.plot(ga, gb, label=MD.labels[y_t], marker='o')
            bax.scatter(len(ga), gb[-1], c='g', s=30, marker='s')
            bax.annotate(str(round(
                gb[-1], 3)), xy=(len(ga) - 1, gb[-1]), xytext=(len(ga) - 1, gb[-1]), fontsize=10)

            bax.axvline(60000/gap, color='r', linestyle='--')

            if y_t == 'loss':
                if batch:
                    bax.annotate('bs = 32\nlr = 0.01', xy=(62000//gap, 1.4), xytext=(62000//gap, 1.4), fontsize=15)
                else:
                    bax.annotate('bs = 1000\nlr = 0.001', xy=(62000//gap, 1.4), xytext=(62000//gap, 1.4), fontsize=15)
                bax.annotate('bs = 1000\nlr = 0.01', xy=(10000//gap, 1.4), xytext=(10000//gap, 1.4), fontsize=15)
            elif y_t == 'grad':
                if batch:
                    bax.annotate('bs = 32\nlr = 0.01', xy=(62000//gap, 10), xytext=(62000//gap, 10), fontsize=15)
                else:
                    bax.annotate('bs = 1000\nlr = 0.001', xy=(62000//gap, 10), xytext=(62000//gap, 10), fontsize=15)
                bax.annotate('bs = 1000\nlr = 0.01', xy=(10000//gap, 10), xytext=(10000//gap, 10), fontsize=15)
            elif y_t == 'acc':
                if batch:
                    bax.annotate('bs = 32\nlr = 0.01', xy=(62000//gap, 0.5), xytext=(62000//gap, 0.5), fontsize=15)
                else:
                    bax.annotate('bs = 1000\nlr = 0.001', xy=(62000//gap, 0.5), xytext=(62000//gap, 0.5), fontsize=15)
                bax.annotate('bs = 1000\nlr = 0.01', xy=(10000//gap, 0.5), xytext=(10000//gap, 0.5), fontsize=15)
            else:
                print('error!')


            if y_t == 'loss':
                if len(md['t_loss']):

                    gb = list(tb[::gap]) + [tb[-1]]

                    bax.plot(
                        ga, gb, label=MD.labels['t_loss'], marker='+', linestyle='-.')
                    bax.scatter(
                        len(ga) - 1, gb[-1], c='g', s=30, marker='s')
                    bax.annotate(str(round(
                        gb[-1], 3)), xy=(len(ga) - 1, gb[-1]), xytext=(len(ga) - 1, gb[-1]), fontsize=10)
                if len(md['flat']):
                    for i, v in md['flat'].items():
                        if i % gap == 0:
                            bax.scatter(numpy.ones(len(v)) * i, v,
                                        c=v, s=1, alpha=1, cmap='rainbow')

            if y_t == 'acc':
                if len(md['t_acc']):

                    # b = md['t_acc'][:a[-1] + 1]
                    gb = 1.0-numpy.array(list(tb[::gap]) + [tb[-1]])

                    bax.plot(
                        ga, gb, label=MD.labels['t_acc'], marker='+', linestyle='-.')
                    bax.scatter(
                        len(ga) - 1, gb[-1], c='g', s=30, marker='s')
                    bax.annotate(str(round(
                        gb[-1], 3)), xy=(len(ga) - 1, gb[-1]), xytext=(len(ga) - 1, gb[-1]), fontsize=10)

            bax.legend(prop={'size': 15}, loc='best')
            bax.set_xlabel(MD.labels[x_t], fontsize=15, labelpad=20)
            # plt.yscale('log')
            # if y_t == 'acc':
                # bax.yscale('linear')
            # bax.ylim([1e-2, 2])

            plt.title(x_t + ' v.s. ' + y_t, fontsize=15)
            # plt.savefig(rename('images/' + MD.name + '-' +
                               # x_t + '-' + y_t), format='eps', dpi=1000)
            if batch:
                plt.savefig(rename('images/'+str(data_index)+'_'+y_t+'_batch'), format='eps', dpi=1000)
            else:
                plt.savefig(rename('images/'+str(data_index)+'_'+y_t+'_lr'), format='eps', dpi=1000)
            plt.show()
    plt.close()


if __name__ == '__main__':
    make_plots()
