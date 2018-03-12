# @Author: Xi He <Heerye>
# @Date:   2018-02-23T04:23:16-05:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: utils.py
# @Last modified by:   Heerye
# @Last modified time: 2018-03-07T07:16:05-05:00


class Worker(object):

    def __init__(self, name, addr):
        self.name = name
        self.addr = addr

    def __str__(self):
        string = u'[<Worker> name:%s addr:%s]' % (self.name, self.addr)
        return string


import collections
from settings import args
import numpy


class Measure(object):
    def __init__(self, name):
        self.name = name
        self.index = ['loss', 'grad', 'acc', 'time', 't_loss', 't_acc','lr']
        self.labels = {'iter': 'iteration (k)', 'loss': '$f(x_k)$',
                       'grad': '$||∇f(x_k)||$', 'acc': 'error', 'time': 'time',
                       'epoch': 'iter(k)','t_loss': 'test $f(x_k)$', 't_acc': 'test error'}
        self.dict = collections.OrderedDict()
        for i in self.index:
            self.dict[i] = numpy.empty((args.N,))
        self.dict['iter'] = []
        self.dict['epoch'] = []
        self.dict['eigs'] = {}
        self.dict['flat'] = {}

    def __str__(self):
        string = u'[<Measure> name:%s]' % (self.name)

    def _zero(self):
        self.dict = collections.OrderedDict()
        for i in self.index:
            self.dict[i] = numpy.empty((args.N,))
        self.dict['iter'] = []
        self.dict['epoch'] = []
        self.dict['eigs'] = {}
        self.dict['flat'] = {}
