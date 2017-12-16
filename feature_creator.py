import numpy as np

targets = np.loadtxt('twenty/twodim_targets.csv', delimiter=',')
sequences = np.loadtxt('twenty/onedim_sequences.csv', delimiter=',')
locations = np.loadtxt('twenty/onedim_locations.csv', delimiter=',')
natural = np.loadtxt('twenty/twodim_natural.csv', delimiter=',')
energy = np.loadtxt('twenty/twodim_energy.csv', delimiter=',')

size = 128

sequences2 = np.tile(sequences, (1,size))

pairmaps = np.concatenate((targets, natural), axis = 1)
x = np.concatenate((pairmaps, energy), axis = 1)
loc_features = np.concatenate((x, sequences2), axis = 1)
np.savetxt('twolocation_features.csv', loc_features[:100], delimiter=',')
#np.savetxt('processed/onelocation_features.csv', loc_features, delimiter = ',')

#locations2 = np.tile(locations, (1,size))

#base_features = np.concatenate((loc_features, locations), axis = 1)
#np.savetxt('processed/tester_1dbase.csv', base_features[:5], delimiter = ',')
#np.savetxt('processed/onebase_features.csv', base_features, delimiter = ',')

