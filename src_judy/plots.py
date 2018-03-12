# @Author: Xi He <Heerye>
# @Date:   2018-03-01T12:14:22-05:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: plots.py
# @Last modified by:   Heerye
# @Last modified time: 2018-03-03T09:07:07-05:00

import matplotlib
import matplotlib.pyplot as plt
from settings import use_cuda

if use_cuda:
    plt.switch_backend('agg')

fig = plt.figure(1)
fig.subplots_adjust(bottom=0.13, left=0.16)
plt.style.use('seaborn-whitegrid')

plt.plot(range(1,N+1),grad_list, label='$||∇f(x)||$')
plt.plot(len(grad_list), grad_list[-1], 'g*')
plt.annotate(str(round(grad_list[-1],3)), xy=(len(grad_list), grad_list[-1]), xytext=(len(grad_list), grad_list[-1]))

plt.plot(range(1,N+1),diff_list, label='$f(x_k)-f(x_{k-1})$')
plt.plot(len(diff_list), diff_list[-1], 'g*')
plt.annotate(str(round(diff_list[-1],3)), xy=(len(diff_list), diff_list[-1]), xytext=(len(diff_list), diff_list[-1]))

plt.legend(prop={'size': 15}, loc='best')
plt.xlabel('Iterations')
plt.savefig('half_grad_diff_'+str(args.lr)+'.png', format='png', dpi=1000)
# plt.show()
plt.close()

fig = plt.figure(1)
fig.subplots_adjust(bottom=0.13, left=0.16)
plt.style.use('seaborn-whitegrid')

plt.plot(range(1,N+1),acc_list, label='accuracy')
plt.plot(len(acc_list), acc_list[-1], 'g*')
plt.annotate(str(round(acc_list[-1],3)), xy=(len(acc_list), acc_list[-1]), xytext=(len(acc_list), acc_list[-1]))

plt.plot(range(1,N+1),loss_list, label='loss')
plt.plot(len(loss_list), loss_list[-1], 'g*')
plt.annotate(str(round(loss_list[-1],3)), xy=(len(loss_list), loss_list[-1]), xytext=(len(loss_list), loss_list[-1]))

plt.legend(prop={'size': 15}, loc='best')
plt.xlabel('Iterations')
plt.savefig('half_acc_loss_'+str(args.lr)+'.png', format='png', dpi=1000)
# plt.show()
plt.close()

fig = plt.figure(1)
fig.subplots_adjust(bottom=0.13, left=0.16)
plt.style.use('seaborn-whitegrid')

plt.annotate(str(round(min_eig_list[-1],3)), xy=(len(min_eig_list), min_eig_list[-1]), xytext=(len(min_eig_list), min_eig_list[-1]))

for i, v in eig_dic.items():
    plt.scatter(numpy.ones(len(v))*i,v,c=v, s=1, alpha=1, cmap='rainbow')
plt.annotate(str(round(max_eig_list[-1],3)), xy=(len(max_eig_list), max_eig_list[-1]), xytext=(len(max_eig_list), max_eig_list[-1]))

plt.legend(prop={'size': 15}, loc='best')
plt.xlabel('Iterations')
plt.savefig('half_eigs_'+str(args.lr)+'.png', format='png', dpi=1000)
# plt.show()
plt.close()
