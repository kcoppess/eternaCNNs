import numpy as np

def base_id(b):
    if b == 'A':
        return 1
    elif b == 'U':
        return 2
    elif b == 'G':
        return 3
    elif b == 'C':
        return 4
    else:
        print "invalid base"
        return 0

def sequence_id(seq, size):
    num_seq = np.zeros(size)
    for i in range(len(seq)):
        num_seq[i] = base_id(seq[i])
    return num_seq

def pairmaps(struc, size):
    pm1 = np.zeros(size)
    pm2 = np.zeros((size,size))
    n = len(struc)
    pm1[0] = -1
    for k in range(n):
        pm2[k,k] = -1
        if struc[k] == '(':
            count = 1
            for j in range(k+1, n):
                if struc[j] == '(':
                    count += 1
                elif struc[j] == ')':
                    count += -1
                if count == 0:
                    pm1[k] = j
                    pm1[j] = k
                    pm2[k,j] = 1
                    pm2[j,k] = 1
                    break
        else:
            if pm1[k] == 0 and pm1[0] != k:
                pm1[k] = -1
    return pm1, pm2
