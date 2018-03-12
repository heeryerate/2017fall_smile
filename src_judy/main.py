############

#   @File name: main.py
#   @Author:    Heerye
#   @Email: email@example.com

#   @Create date:   2017-09-04 20:24:37

# @Last modified by:   Heerye
# @Last modified time: 2018-03-07T06:56:59-05:00

#   @Description:
#   @Example:

############

from import_header import *

startTime = time.time()
print('Scripts staring ...')

X, Y, t_X, t_Y, F, C, n = generate_data(n_points, n_classes, center_label_list)

train = data_utils.TensorDataset(X, Y)
train_loader = data_utils.DataLoader(train, batch_size=args.b, shuffle=True)

test = data_utils.TensorDataset(t_X, t_Y)
test_loader = data_utils.DataLoader(test, batch_size=len(t_Y), shuffle=False)

print('Data generated ...')
if __debug__:
    print('#(classes): %d, #(training points): %d, #(testing points): %d' %
      (C, len(Y), len(t_Y)))

MD = Measure(name='log_' + str(data_index) + '_' +
             str(args.lr) + '_' + str(args.b))
md = MD.dict
print('Measure logs initilized ...')
if __debug__:
    print('log name: %s' % MD.name)

if args.resume:
    net = torch.load('checkpoints/' + str(data_index) +
                     '_0_1000_2.pt')
else:
    net = Net([F, 10, 10, C])
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
print('Network constructed ...')
# print(net)

if use_optimizer:
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

if advance_measure:
    count, orths, eyes = get_attr(net)


def sgd(net, x, y):
    grad = get_grad(net, x, y)
    for j, para in enumerate(get_params(net)):
        para.data -= args.lr * grad[j].data

elapsed = 0.0
def train(epoch, elapsed=elapsed):
    t1 = time.time()
    for idx, (x, y) in enumerate(train_loader):
        x, y = Variable(x), Variable(y)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        if use_optimizer:
            optimizer.zero_grad()
            loss = get_loss(net, x, y)
            loss.backward()
            optimizer.step()
        else:
            sgd(net, x, y)


        elapsed += time.time() - t1
    md['time'][epoch] = elapsed
    md['lr'][epoch] = lr
    if __debug__:
        print("epoch %d/%d, time %f. " % (epoch, args.N, md['time'][epoch]))

def train_eval(epoch):

    x, y = Variable(X), Variable(Y)

    if use_cuda:
        x, y = x.cuda(), y.cuda()

    grad = get_grad(net, x, y)
    if use_cuda:
        md['loss'][epoch] = get_loss(net, x, y).data.cpu().numpy()
        tmp = 0.0
        for g in grad:
            tmp += LA.norm(g.data.cpu().numpy())**2
    else:
        md['loss'][epoch] = get_loss(net, x, y).data.numpy()
        tmp = 0.0
        for g in grad:
            tmp += LA.norm(g.data.numpy())**2

    md['acc'][epoch] = get_acc(net, x, y)
    md['epoch'].append(epoch)
    md['grad'][epoch] = tmp

    if advance_measure:
        md['eigs'][epoch] = get_eig(net, grad, count, eyes)
        md['flat'][epoch] = get_flatness(net, x, y, args.eps, count, orths)
        flatness = (max(md['flat'][epoch]) - min(md['flat'][epoch])) / args.eps
        if __debug__:
            print('epoch %d/%d, Flatness: %f' % (epoch, args.N, flatness), max(
            md['flat'][epoch]), min(md['flat'][epoch]))
    if __debug__:
        print("epoch %d/%d, loss %f, acc %f, nmg %f. " %
          (epoch, args.N, md['loss'][epoch], md['acc'][epoch], md['grad'][epoch]))


def test_eval(epoch):

    x, y = Variable(t_X), Variable(t_Y)

    if use_cuda:
        x, y = x.cuda(), y.cuda()

    if use_cuda:
        md['t_loss'][epoch] = get_loss(net, x, y).data.cpu().numpy()
    else:
        md['t_loss'][epoch] = get_loss(net, x, y).data.numpy()

    md['t_acc'][epoch] = get_acc(net, x, y)

    if __debug__:
        print("epoch %d/%d, t_loss %f, t_acc %f." %
          (epoch, args.N, md['t_loss'][epoch], md['t_acc'][epoch]))


print("Start Training, time: %f" % (time.time() - startTime))
# for seed in range(1):
for seed in [30]:

    MD._zero()
    md = MD.dict

    torch.manual_seed(seed)

    if args.reuse_params:
        # net = torch.load('checkpoints/' + str(data_index) +
                         # '_' + str(seed) + '.pt')
        net = torch.load('checkpoints/' + str(data_index) +
                         '_0_1000_2.pt')
    else:
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                std = math.sqrt(2.0 / sum(m.weight.data.size()))
                m.weight.data.normal_(0, std)
                m.bias.data.fill_(0.0)

    elapsed = 0.0
    for epoch in range(args.N):

        train(epoch, elapsed=elapsed)
        train_eval(epoch)
        test_eval(epoch)

        elapsed = md['time'][epoch]

        # if md['grad'][epoch] <= 1e-8:
        if epoch == args.N-1:
            torch.save(net, 'checkpoints/' + str(data_index) +
                       '_' + str(seed) + '.pt')
            print('checkpoints saved! saved at %s' % ('checkpoints / '+str(data_index)+'_'+str(seed)+'.pt'))
            break

        if epoch % int(args.N / 4.0) == 1:
            print('*' * 10 + '\nlog saved at ratio %f' %
                  (epoch / args.N) + '\n' + '*' * 10)
            with open('logs/' + MD.name + '_' + str(seed) + '.pkl', 'wb') as f:
                pickle.dump(MD, f)

    with open('logs/' + MD.name + '_' + str(seed) + '.pkl', 'wb') as f:
        pickle.dump(MD, f)

print('The script took {0} seconds!'.format(time.time() - startTime))
print('log saved at %s' % ('logs/' + MD.name + '_' + str(seed) + '.pkl'))
