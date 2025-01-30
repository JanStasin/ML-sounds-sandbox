import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#dict_ts = {'A': {},'B': {}}
dict_mats = {'A': {},'B': {}}


n_fft = 2048 


with open('/Users/jansta/learn/acoustics/ESC-50-master/meta/esc50.csv') as f:
    for line in f.readlines():
        line = line.strip().split(',')
        print(line)
        if line[-1] == 'A':
            #print(f'ts_{line[0][:-4]}.npy')
            #dict_ts['A'][line[3]] = np.load(f'/Users/jansta/learn/acoustics/ts/ts_{line[0][:-4]}.npy', allow_pickle=True)

            mel_spect = np.load(f'/Users/jansta/learn/acoustics/spects/spect_dB_mat_{line[0][:-4]}.npy', allow_pickle=True)
            print(mel_spect.shape)

            if line[3] not in dict_mats['A'].keys():
                dict_mats['A'][line[3]] = [mel_spect]
            else:
                dict_mats['A'][line[3]].append(mel_spect)

        elif line[-1] == 'B':   
            #dict_ts['B'][line[3]] = np.load(f'/Users/jansta/learn/acoustics/ts/ts_{line[0][:-4]}.npy', allow_pickle=True)
            mel_spect = np.load(f'/Users/jansta/learn/acoustics/spects/spect_dB_mat_{line[0][:-4]}.npy', allow_pickle=True)

            if line[3] not in dict_mats['B'].keys():
                dict_mats['B'][line[3]] = [mel_spect]
            else:
                dict_mats['B'][line[3]].append(mel_spect)
print(dict_mats)
#np.save('/Users/jansta/learn/acoustics/dict_ts.npy', dict_ts, allow_pickle=True)
np.save('/Users/jansta/learn/acoustics/dict_mats_dB.npy', dict_mats, allow_pickle=True)

# print(dict_ts['A']['dog'])
# plt.figure(figsize=(10, 4))
# plt.imshow(dict_mats['A']['dog'])
# plt.show()
#print(dict_mats['A']['dog'])