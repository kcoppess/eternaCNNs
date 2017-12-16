import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from l_cnn1D import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='train a NN to predict'
                                     'structure/chemical mapping data as a function of seq')
    parser.add_argument('-k', '--k', help='which test set')
    parser.add_argument('-resdir', '--resultdir', help='location of results')
    parser.add_argument('-c', '--c')
    parser.add_argument('-f', '--f')
    return parser.parse_args()

args = parse_args()

k = int(args.k) # cross-validation set
Ntest = 25

c = int(args.c)
f = int(args.f)

restore = None #put in file name of restoring
learning_rate = 1e-4
iters = 10000
#Ntrain = 859
#Ntest = 1227
batch_size = 32
save = True
filename = '/scratch/users/kcoppess/results/loc1D'

conv_w_shapes = [ [5, 4, 1, c], \
        [5, 4, c, c], \
        [5, 4, c, c], \
        [5, 4, c, c], \
	[2, 4, c, c], \
	[2, 4, c, c], \
	[2, 4, c, c] ]

conv_b_shapes = [ [c], \
        [c], \
        [c], \
        [c], \
        [c], \
        [c], \
	[c] ]

fc_w_shapes = [ [c, f], \
        [f, f], \
	[f, f], \
	[f, 128] ]

fc_b_shapes = [ [f], \
        [f], \
	[f], \
	[128] ]

print 'loading data...'
all_inputs = np.loadtxt('/scratch/users/kcoppess/processed/small/onelocation_features.csv', delimiter=',') #pd.read_csv('pairmap2D.csv', delimiter=',', header=None, nrows=Ntrain)
all_labels = np.loadtxt('/scratch/users/kcoppess/processed/small/onedim_location.csv', delimiter=',') #pd.read_csv('loops.csv', delimiter=',', header = None, nrows=Ntrain)

if k == 0:
    inputs = all_inputs[(k+1)*Ntest:]
    labels = all_labels[(k+1)*Ntest:]
    test_inputs = all_inputs[:(k+1)*Ntest] #pd.read_csv('pairmap2D.csv', delimiter=',', header=None, skiprows = Ntrain, nrows=Ntest)
    test_labels = all_labels[:(k+1)*Ntest] #pd.read_csv('loops.csv', delimiter=',', header=None, skiprows=Ntrain, nrows=Ntest)
elif k == 3:
    inputs = all_inputs[:k*Ntest]
    labels = all_labels[:k*Ntest]
    test_inputs = all_inputs[k*Ntest:] #pd.read_csv('pairmap2D.csv', delimiter=',', header=None, skiprows = Ntrain, nrows=Ntest)
    test_labels = all_labels[k*Ntest:] #pd.read_csv('loops.csv', delimiter=',', header=None, skiprows=Ntrain, nrows=Ntest)
else:
    inputs = np.concatenate((all_inputs[:k*Ntest], all_inputs[(k+1)*Ntest:]))
    labels = np.concatenate((all_labels[:k*Ntest], all_labels[(k+1)*Ntest:]))
    test_inputs = all_inputs[k*Ntest:(k+1)*Ntest] #pd.read_csv('pairmap2D.csv', delimiter=',', header=None, skiprows = Ntrain, nrows=Ntest)
    test_labels = all_labels[k*Ntest:(k+1)*Ntest] #pd.read_csv('loops.csv', delimiter=',', header=None, skiprows=Ntrain, nrows=Ntest)


print 'construction in progess... {} convolution layers and {} fully connected layers, learning_rate = {}, {} iterations'.format(len(conv_w_shapes), len(fc_w_shapes), learning_rate, iters)
name = '%d_conv_%d_fc_k%d_%.2e' % (c, f, k, learning_rate)
net = CNN(conv_w_shapes, conv_b_shapes, fc_w_shapes, fc_b_shapes, learning_rate, name)

if restore is not None:
    print "restoration..."
    net.restore(restore)
else:
    print "training..."
    loss, accuracy = net.train(inputs, labels, iters, batch_size)
    np.savetxt('%s/%s.loss' % (filename, net.name), loss, delimiter='\t')
    np.savetxt('%s/%s.accuracy' % (filename, net.name), accuracy, delimiter='\t')

print "testing..."

test_pred = np.asarray(net.test(test_inputs))
pred = np.argmax(test_pred, axis=1)
truth = np.argmax(np.asarray(test_labels), axis=1)
predictions = np.concatenate((np.array([pred]),np.array([truth]))).transpose()
correct = np.equal(truth, pred)
test_accuracy= correct.astype(float)
acc = np.mean(test_accuracy)
np.savetxt('%s/%s.predictions' % (filename, net.name), predictions, delimiter='\t')
print "final dev-test error: "+str(acc)

if save:
    print "saving model..."
    net.save(filename, '_testerr%f' % acc)
