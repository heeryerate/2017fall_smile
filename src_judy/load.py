# @Author: Xi He <Heerye>
# @Date:   2018-02-23T04:18:44-05:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: load.py
# @Last modified by:   Heerye
# @Last modified time: 2018-03-07T08:45:30-05:00

import pickle
from utils import Worker, Measure

import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
import sys
import math


def mylog(v):
    return 0 if v == 0 else numpy.log(v)


def symlog(v):
    return [mylog(i) if i > 0 else 0 for i in v]


def neg_symlog(v):
    return [mylog(-i) if i < 0 else 0 for i in v]


def rename(s):
    return s.replace('_', '-').replace('.', '-').replace(' ', '-').replace(',','-') + '.eps'


gap = 50


def make_plots(log_path='./logs'):

    # file_list = [file for file in os.listdir(log_path) if file.endswith(
        # '.pkl') and 'log_2_0.1_300_10' in file]
    file_list = [file for file in os.listdir(log_path) if file.endswith('_30.pkl')]
    # file_list = ['log_3_0.01_1000_0.pkl', 'log_3_0.01_1000_1.pkl', 'log_3_0.01_1000_2.pkl','log_3_0.01_32_0']
    # file_list = ['log_3_0.01_1000_3.pkl', 'log_3_0.001_1000_0.pkl', 'log_3_0.01_32_0.pkl']

    for file in file_list:
        try:
            print('Handle %s' % file)
            with open(log_path + '/' + file, 'rb') as f:
                MD = pickle.load(f)

            md = MD.dict
        except:
            print('bad file: %s'%file)
            os.rename('logs/'+file, 'logs/bin/'+file)
            break


        # fig = plt.figure(1)
        # fig.subplots_adjust(bottom=0.13, left=0.16)
        # plt.style.use('seaborn-whitegrid')
        # for i, v in md['eigs'].items():
        #     if i % gap == 0:
        #         plt.scatter(numpy.ones(len(v))*i,symlog(v),c=v, s=1, alpha=1, cmap='rainbow')
        # plt.title('$\log(\lambda_+(∇^2f(x_k)))$',fontsize=15)
        # plt.xlabel(MD.labels['iter'],fontsize=15)
        # plt.savefig(rename('images/'+MD.name+'-l+'), format='eps', dpi=1000)
        # plt.show()
        # plt.close()
        #
        # fig = plt.figure(1)
        # fig.subplots_adjust(bottom=0.13, left=0.16)
        # plt.style.use('seaborn-whitegrid')
        # for i, v in md['eigs'].items():
        #     if i % gap == 0:
        #         plt.scatter(numpy.ones(len(v))*i,neg_symlog(v),c=v, s=1, alpha=1, cmap='rainbow')
        # plt.title('$\log(-\lambda_-(∇^2f(x_k)))$',fontsize=15)
        # plt.xlabel(MD.labels['iter'],fontsize=15)
        # plt.savefig(rename('images/'+MD.name+'-l-'), format='eps', dpi=1000)
        # plt.show()
        # plt.close()
        #
        # fig = plt.figure(1)
        # fig.subplots_adjust(bottom=0.13, left=0.16)
        # plt.style.use('seaborn-whitegrid')
        # for i, v in md['eigs'].items():
        #     if i % gap == 0:
        #         plt.scatter(numpy.ones(len(v))*i,v,c=v, s=1, alpha=1, cmap='rainbow')
        # plt.title('$\lambda(∇^2f(x_k))$',fontsize=15)
        # plt.xlabel(MD.labels['iter'],fontsize=15)
        # plt.savefig(rename('images/'+MD.name+'-l'), format='eps', dpi=1000)
        # plt.show()
        # plt.close()

        # fig = plt.figure(1)
        # fig.subplots_adjust(bottom=0.13, left=0.16)
        # plt.style.use('seaborn-whitegrid')
        # for i, v in md['eigs'].items():
        #     if i % gap == 0:
        #         plt.scatter(numpy.ones(len(v))*i,v,c=v, s=1, alpha=1, cmap='rainbow')
        # plt.title('$\lambda(∇^2f(x_k))$',fontsize=15)
        # plt.xlabel(MD.labels['iter'],fontsize=15)
        # plt.savefig(rename('images/'+MD.name+'-l'), format='eps', dpi=1000)
        # plt.show()
        # plt.close()

        for x_t in ['epoch']:
            # fig = plt.figure(figsize=(14,7*(math.sqrt(5)-1)))
            fig = plt.figure(1)
            fig.subplots_adjust(bottom=0.13, left=0.16)
            att = file.split('_')
            bs = att[3]
            lr = att[2]
            da = att[1]
            seed = att[4][:-4]
            # plot_name = 'bs=%s, lr=%s'%(bs, lr)
            plot_name = 'bs=%s' %(bs)
            fig.suptitle(plot_name, x=0.5,y=0.95, fontsize=20)
            plt.style.use('seaborn-whitegrid')
            for nplot, y_t in enumerate(['loss','acc','grad','lr']):
                plt.subplot(2,2,nplot+1)

                MD.labels['lr']='lr'

                try:
                    a = md[x_t]
                    b = md[y_t][:a[-1] + 1]
                except:
                    print('bad file: %s'%file)
                    os.rename('logs/'+file, 'logs/bin/'+file)
                    break


                ga = a[::gap]
                gb = b[::gap]

                if y_t == 'acc':
                    gb = 1.0 - gb

                # print(len(ga), len(gb))

                plt.plot(ga, gb, label=MD.labels[y_t], marker='o')
                plt.scatter(len(a) - 1, gb[-1], c='g', s=30, marker='s')
                plt.annotate(str(round(
                    gb[-1], 3)), xy=(len(a) - 1, gb[-1]), xytext=(len(a) - 1, gb[-1]), fontsize=10)

                if y_t == 'loss':
                    if len(md['t_loss']):

                        b = md['t_loss'][:a[-1] + 1]
                        gb = b[::gap]

                        plt.plot(
                            ga, gb, label=MD.labels['t_loss'], marker='+', linestyle='-.')
                        plt.scatter(
                            len(a) - 1, gb[-1], c='g', s=30, marker='s')
                        plt.annotate(str(round(
                            gb[-1], 3)), xy=(len(a) - 1, gb[-1]), xytext=(len(a) - 1, gb[-1]), fontsize=10)
                    if len(md['flat']):
                        for i, v in md['flat'].items():
                            if i % gap == 0:
                                plt.scatter(numpy.ones(len(v)) * i, v,
                                            c=v, s=1, alpha=1, cmap='rainbow')

                if y_t == 'acc':
                    if len(md['t_acc']):

                        b = md['t_acc'][:a[-1] + 1]
                        gb = 1.0-b[::gap]

                        plt.plot(
                            ga, gb, label=MD.labels['t_acc'], marker='+', linestyle='-.')
                        plt.scatter(
                            len(a) - 1, gb[-1], c='g', s=30, marker='s')
                        plt.annotate(str(round(
                            gb[-1], 3)), xy=(len(a) - 1, gb[-1]), xytext=(len(a) - 1, gb[-1]), fontsize=10)

                plt.legend(prop={'size': 10}, loc='best')
                plt.xlabel(MD.labels[x_t], fontsize=10)
                # plt.yscale('log')
                # if y_t == 'acc':
                    # plt.yscale('linear')
                # if y_t == 'acc':
                    # plt.ylim([0.1, 0.3])
                # if y_t == 'loss':
                    # plt.ylim([0.3, 0.6])

                # plt.title(x_t + ' v.s. ' + y_t, fontsize=15)
                # plt.supertitle(file, fontsize=15)
                # plt.savefig(rename('images/' + MD.name + '-' +
                                   # x_t + '-' + y_t), format='eps', dpi=1000)
            # plt.show()
            # plt.savefig(rename('images/'+da+'_'+plot_name), format='eps', dpi=1000)
            plt.savefig(rename('images/'+da+'_'+plot_name+'_'+seed), format='eps', dpi=1000)
        plt.close()


if __name__ == '__main__':
    make_plots()
