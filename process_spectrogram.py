import random
import os
from timeit import default_timer as timer
from multiprocessing import Process

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal

from obspy import read, read_inventory


def preprocess_waveform(T, inventory, pre_filt=(0.005, 0.006, 30.0, 35.0)):
    T.detrend('demean')
    T.remove_response(inventory=inventory, output='VEL',
                      pre_filt=pre_filt, taper=True)

    return


def normalize(data):
    # return data/np.max(np.abs(data)) # Divide by the maximum
    # return (data - np.mean(data))/(np.max(data) - np.min(data)) # Mean normalization

    return (data - np.mean(data))/np.std(data) # Z-score


def spectrogram(data, sampling_rate, wlen, per_lap, fact_nfft):
    nperseg = round(wlen * sampling_rate)
    nfft = round(nperseg * fact_nfft)
    noverlap = round(nperseg * per_lap)

    f, t, Sxx = scipy.signal.spectrogram(
        x=data, fs=sampling_rate, nfft=nfft, nperseg=nperseg, noverlap=noverlap)

    return f, t, Sxx


def process_spectrogram(f, t, Sxx, fmin=None, fmax=None):
    if fmin is None:
        fmin = f[0]
    if fmax is None:
        fmax = f[-1]

    ii = np.where((f >= fmin) & (f <= fmax))

    Sxx_filter = Sxx[ii]
    Sxx_db = np.log(Sxx_filter)*10

    return f[ii], t, Sxx_db


def percentile(Sxx, keep=(96, 96)):
    vmin = np.percentile(Sxx, (100 - keep[0])/2)
    vmax = np.percentile(Sxx, (100 + keep[1])/2)

    return vmin, vmax


def plot_image(f, t, Sxx, filename, path='', vmin=None, vmax=None, logy=False, size=(1, 1), dpi=512):
    file = os.path.join(path, filename)

    fig = plt.figure(figsize=(size[0], size[1]), dpi=dpi)

    plt.pcolormesh(t, f, Sxx, cmap='jet', vmin=vmin, vmax=vmax)
    if logy:
        plt.yscale('log')

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(file, bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def plot_waveform(T, filename, path='', size=(1, 1), dpi=512):
    t = T.times()
    data = T.data
    file = os.path.join(path, filename)

    fig = plt.figure(figsize=(size[0], size[1]), dpi=dpi)

    plt.plot(t, data, c='k', linewidth=.1)
    plt.xlim(0, t[-1])
    m = np.max((np.max(data), np.min(data)))
    plt.ylim(-m, m)

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(file, bbox_inches='tight', pad_inches=0)
    plt.close()

    return

def plot_dual(f, t, Sxx, T, filename, path='', vmin=None, vmax=None, logy=False, size=(1, 1), dpi=512):
    file = os.path.join(path, filename)
    fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(size[0], size[1]), dpi=dpi)

    ax = axs[0]
    ax.pcolormesh(t, f, Sxx, cmap='jet', vmin=vmin, vmax=vmax)
    if logy:
        ax.set_yscale('log')

    ax = axs[1]
    tt = T.times()
    data = T.data
    #data = data-np.mean(data)
    data = data/np.max(np.abs(data))
    ax.plot(tt, data, c='k', linewidth=.2)
    ax.set_ylim(-1, 1)

    plt.xlim(t[0], t[-1])

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in axs:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    
    plt.savefig(file, bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def slice_trace(T, duration):
    windows = np.arange(T.stats.starttime, T.stats.endtime, duration)

    return [T.slice(window, window + duration) for window in windows]


def process_file(file, path, inventory, out_path='spectrograms'):
    filename = file.split('.')[0]
    t1 = timer()

    st = read(os.path.join(path, file))
    T = st.select(component='Z')[0]
    Tcopy = T.copy()

    preprocess_waveform(Tcopy, inventory=inventory)
    

    filename = f'{Tcopy.stats.starttime.strftime("%Y_%m_%d_%H_%M_%S")}.png'
    data = Tcopy.data
    #data = normalize(data)

    sampling_rate = Tcopy.stats.sampling_rate

    # Middle frenquencies
    wlen = 50
    per_lap = .9
    fact_nfft = 10
    fmin = .05
    fmax = 1
    vmin = -.4
    vmax = 2.1
    logy = True
    name = 'MF'

    f, t, Sxx = spectrogram(data, sampling_rate=sampling_rate,
                            wlen=wlen, per_lap=per_lap, fact_nfft=fact_nfft)
    f, t, Sxx = process_spectrogram(f, t, Sxx, fmin=fmin, fmax=fmax)
    Sxx = normalize(Sxx)

    plot_image(f, t, Sxx, filename=filename, path=os.path.join(out_path, name),
               vmin=vmin, vmax=vmax, logy=logy)

    name = 'MF_LF'
    Tfilter = Tcopy.copy()
    Tfilter.filter('bandpass', freqmin=.01, freqmax=.08)

    plot_dual(f, t, Sxx, Tfilter, filename=filename, path=os.path.join(out_path, name),
            vmin=vmin, vmax=vmax, logy=logy)

    # Low frenquencies
    wlen = 100
    per_lap = .9
    fact_nfft = 10
    fmin = .01
    fmax = .08
    vmin = -.6
    vmax = 2.0
    logy = False
    name = 'LF'

    f, t, Sxx = spectrogram(data, sampling_rate=sampling_rate,
                            wlen=wlen, per_lap=per_lap, fact_nfft=fact_nfft)
    f, t, Sxx = process_spectrogram(f, t, Sxx, fmin=fmin, fmax=fmax)
    Sxx = normalize(Sxx)

    plot_image(f, t, Sxx, filename=filename, path=os.path.join(out_path, name),
               vmin=vmin, vmax=vmax, logy=logy)

    name = 'LF_HF'
    Tfilter = Tcopy.copy()
    Tfilter.filter('bandpass', freqmin=2, freqmax=8)

    plot_dual(f, t, Sxx, Tfilter, filename=filename, path=os.path.join(out_path, name),
            vmin=vmin, vmax=vmax, logy=logy)

    duration = 10*60
    Tslices = slice_trace(T, duration)

    for Tslice in Tslices:
        filename = f'{Tslice.stats.starttime.strftime("%Y_%m_%d_%H_%M_%S")}.png'
        preprocess_waveform(Tslice, inventory=inventory)

        data = Tslice.data
        # data = normalize(data)

        wlen = 10
        per_lap = .9
        fact_nfft = 10
        fmin = 2
        fmax = 8
        vmin = -1.0
        vmax = 2.6
        logy = False
        name = 'HF'

        f, t, Sxx = spectrogram(data, sampling_rate=sampling_rate,
                                wlen=wlen, per_lap=per_lap, fact_nfft=fact_nfft)
        f, t, Sxx = process_spectrogram(f, t, Sxx, fmin=fmin, fmax=fmax)
        Sxx = normalize(Sxx)

        plot_image(f, t, Sxx, filename=filename, path=os.path.join(out_path, name),
                   vmin=vmin, vmax=vmax, logy=logy)
        
        name = 'HF_MF'
        Tfilter = Tcopy.copy()
        Tfilter.filter('bandpass', freqmin=.05, freqmax=1)

        plot_dual(f, t, Sxx, Tfilter, filename=filename, path=os.path.join(out_path, name),
                vmin=vmin, vmax=vmax, logy=logy)

    t2 = timer()
    print(f'{filename} t={t2-t1}')

    return


def process_files(files, path, inventory, out_path='spectrograms'):
    for file in files:
        process_file(file, path=path, inventory=inventory, out_path=out_path)

    return


def process_files_multiprocess(files, path, inventory, n_processes=5, out_path='spectrograms'):
    processes = []
    n_files = len(files)
    slice_size = round(n_files/n_processes)

    for k in range(n_processes):
        i1 = k*slice_size
        if k == n_processes - 1:
            i2 = n_files
        else:
            i2 = (k+1)*slice_size

        process = Process(target=process_files, args=(
            files[i1:i2], path, inventory, out_path))
        processes.append(process)
        process.start()
        print(f'Creating process {k+1}/{n_processes} : files {i1} to {i2}')

    for process in processes:
        process.join()

    return


if __name__ == '__main__':
    matplotlib.use('Agg')

    inventory_path = 'metadata'
    inventory_filename = 'inventory.xml'
    inventory = read_inventory(os.path.join(
        inventory_path, inventory_filename), format='STATIONXML')

    path = 'data'
    files = os.listdir(path)

    process_files_multiprocess(
        files, path, inventory=inventory, n_processes=5, out_path='spectrograms')
