import numpy as np
import matplotlib.pyplot as plt

loc = np.loadtxt('small/onelocation_features.csv', delimiter=',')
bas = np.loadtxt('small/onebase_features.csv', delimiter=',')
llabels = np.loadtxt('small/onedim_location.csv', delimiter=',')
blabels = np.loadtxt('small/bases.csv', delimiter=',')

location_features = loc[:75]
test_location = loc[75:]

base_features = bas[:75]
test_base = bas[75:]

loc_labels = llabels[:75]
test_llabels = llabels[75:]

base_labels = blabels[:75]
test_blabels = llabels[75:]

def softmax_loc(x):
    m = np.shape(x)[0]
    sum_exp_diff = np.zeros((m, 128))
    for i in range(128):
        z_i = np.tile(x[:,i], (128,1)).transpose()
        sum_exp_diff += np.exp(z_i - x)
    s = 1. / sum_exp_diff
    return s

def softmax_base(x):
    m = np.shape(x)[0]
    sum_exp_diff = np.zeros((m, 4))
    for i in range(4):
        z_i = np.tile(x[:,i], (4,1)).transpose()
        sum_exp_diff += np.exp(z_i - x)
    s = 1. / sum_exp_diff
    return s

# taken from HW4 Q1 starter code
def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

learning_rate = 1e-2
theta_loc = np.zeros((128*4, 128))
theta_base = np.zeros((128*5, 4))
iterations = 10000

loc_train_acc = []
bas_train_acc = []
loc_cost = []
bas_cost = []
iters = []

print 'training...'
for i in range(iterations):
    y_loc = softmax_loc(np.matmul(location_features, theta_loc))
    y_base = softmax_base(np.matmul(base_features, theta_base))
    if i%500 == 0 and i != 0:
        acc_loc = compute_accuracy(y_loc, loc_labels)
        acc_base = compute_accuracy(y_base, base_labels)
        print 'step {}: loc {}, base {}'.format(i, acc_loc, acc_base)
        loc_train_acc.append(acc_loc)
        bas_train_acc.append(acc_base)
        loc_cost.append(np.linalg.norm(theta_loc))
        bas_cost.append(np.linalg.norm(theta_base))
        iters.append(i)
    theta_loc += -learning_rate*np.matmul(location_features.transpose(), (loc_labels - y_loc))
    theta_base += -learning_rate*np.matmul(base_features.transpose(), (base_labels - y_base))

print 'training complete...'

print 'testing...'

y_loc = softmax_loc(np.matmul(test_location, theta_loc))
y_base = softmax_base(np.matmul(test_base, theta_base))
test_loc_acc = compute_accuracy(y_loc, test_llabels)
test_bas_acc = compute_accuracy(y_base, test_blabels)

print 'Test accuracies: loc {}, bas {}'.format(test_loc_acc, test_bas_acc)

plt.figure(0)
plt.plot(iters, loc_train_acc, linewidth=2, label='Position')
plt.plot(iters, bas_train_acc, linewidth=2, label='Base')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Training Accuracy')
plt.figure(1)
plt.plot(iters, loc_cost, linewidth=2, label='Position')
plt.plot(iters, bas_cost, linewidth=2, label='Base')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Magnitude of Update')
plt.show()
