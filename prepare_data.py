# misc functions for the CNN 

import numpy as np
from scipy.ndimage import gaussian_filter1d
import tqdm

def getSpkMat(spike_times, unit_id, start, stop,bin_size, sigma):
        dur = stop - start
        bin_count = int(np.ceil(dur/bin_size))
        bins = np.linspace(start,stop,num=bin_count+1)
        timestamps = np.linspace(start, stop, num=bin_count)
        spkMat = np.zeros((bin_count, len(unit_id)))
        for i, unit in enumerate(tqdm.tqdm(unit_id)):
            spkMat[:, i] = np.histogram(spike_times[unit], bins=bins)[0].tolist()
        spkMat = gaussian_filter1d(spkMat, sigma, axis=0)/bin_size
        spkMat = (spkMat - spkMat.mean(axis=0))/spkMat.std(axis=0)
        return spkMat, timestamps
    
def getSnippets(win, spkMat, timestamps, stimSt):
    t = np.linspace(-win[0], win[1], int((win[1]-win[0])/0.01))
    Data = np.zeros((spkMat.shape[1],1,len(t), len(stimSt)))
    for i, st in enumerate(tqdm.tqdm(stimSt)):
        idx = np.abs(timestamps - (st+win[0])).argmin()
        Data[:,0,:,i] = spkMat[int(idx):int(idx + len(t)),:].T
    return Data, t

