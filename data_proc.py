# processing move-set data and encoding
import pandas as pd
import numpy as np
from encoding import *
import RNA

experts = []
expert_uid = pd.read_csv('experts.txt', delimiter=',', usecols=[0], parse_dates=['uid'])
for i in range(14):
    experts.append(expert_uid.loc[i,'uid'])

target_structures = pd.read_csv('puzzle-structure-data.txt', delimiter='\t', usecols=[0,1], parse_dates=['pid','structure'])
targets = target_structures.set_index('pid')

playouts = pd.read_csv('moveset6-22a_mod.txt', delimiter='\t', usecols=[1,2,3], parse_dates=['pid', 'uid', 'move_set'])

N = 20

size = 128

onedim_targets = []
twodim_targets = []
onedim_natural = []
twodim_natural = []
onedim_energy = []
twodim_energy = []
onedim_sequences = []
onedim_locations = []
bases = []

energy = np.zeros(size)
energy[0] = 1.
c = 0
i = 0
while c < N and i < 429762:
    od_targets = []
    td_targets = []
    od_natural = []
    td_natural = []
    od_energy = []
    td_energy = []
    od_sequences = []
    od_locations = []
    bas = []

    pid = playouts.loc[i,'pid']
    if playouts.loc[i,'uid'] not in experts:
        i += 1
        continue
    print 'expert'
    target_dotbrack = targets.loc[pid, 'structure']
    print len(target_dotbrack)
    if len(target_dotbrack) > size:
        i += 1
        print 'ignoring'
        continue
    print i
    onedim_target, twodim_target = pairmaps(target_dotbrack, size)
    #uid = playouts.loc[i,'uid']
    try:
        moveset = eval(playouts.loc[i,'move_set'])
    except:
        i += 1
        print 'null'
        continue
    moves = moveset['moves']
    sequence = moveset['begin_from']
    current_sequence = sequence_id(sequence, size)
    
    for m in moves:
        for j in range(len(m)):
            try: # handling reset; just training on moves after reset
                sequence = m[j]['sequence']
                od_targets = []
                td_targets = []
                od_natural = []
                td_natural = []
                od_energy = []
                td_energy = []
                od_sequences = []
                od_locations = []
                bas = []
                print 'caught' 
                current_sequence = sequence_id(sequence, size)
            except:
                (natural_dotbrack, mfe) = RNA.fold(sequence)
                ener1 = mfe * energy
                ener2 = mfe * np.eye(size)
                
                nat1, nat2 = pairmaps(natural_dotbrack, size)
                
                loc = np.zeros(size)
                base = np.zeros(4)
                pos = m[j]['pos'] - 1
                b = base_id(m[j]['base'])
                
                loc[pos] = 1
                base[b - 1] = 1

                
                od_sequences.append(current_sequence)
                od_locations.append(loc)
                bas.append(base)
                od_targets.append(onedim_target)
                td_targets.append(np.concatenate(twodim_target))
                od_energy.append(ener1)
                td_energy.append(np.concatenate(ener2))
                od_natural.append(nat1)
                td_natural.append(np.concatenate(nat2))
                
                current_sequence[pos] = b
                sequence = sequence[:pos]+m[j]['base']+sequence[pos+1:]
    onedim_targets += od_targets
    twodim_targets += td_targets
    onedim_natural += od_natural
    twodim_natural += td_natural
    onedim_energy += od_energy
    twodim_energy += td_energy
    onedim_sequences += od_sequences
    onedim_locations += od_locations
    bases += bas

    c += 1
    i += 1

odp = np.asarray(onedim_targets)
tdp = np.asarray(twodim_targets)
seq = np.asarray(onedim_sequences)
locations = np.asarray(onedim_locations)
b = np.asarray(bases)
one_nat = np.asarray(onedim_natural)
two_nat = np.asarray(twodim_natural)
one_ener = np.asarray(onedim_energy)
two_ener = np.asarray(twodim_energy)

np.savetxt('onedim_natural.csv', one_nat, delimiter=',')
np.savetxt('twodim_natural.csv', two_nat, delimiter=',')
np.savetxt('onedim_energy.csv', one_ener, delimiter=',')
np.savetxt('twodim_energy.csv', two_ener, delimiter=',')
np.savetxt('onedim_targets.csv', odp, delimiter=',')
np.savetxt('twodim_targets.csv', tdp, delimiter=',')
np.savetxt('onedim_sequences.csv', seq, delimiter=',')
np.savetxt('onedim_locations.csv', locations, delimiter=',')
np.savetxt('bases.csv', b, delimiter=',')

