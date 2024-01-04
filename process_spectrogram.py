import os
from timeit import default_timer as timer
from multiprocessing import Process

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal

from obspy import read, read_inventory


matplotlib.use('Agg')


def preprocess_waveform(st, inventory, pre_filt=(0.005, 0.006, 30.0, 35.0)):
    st.detrend('demean')
    st.remove_response(inventory=inventory, output='VEL',
                       pre_filt=pre_filt, taper=True)

    return


def spectrogram(T, wlen, per_lap, fact_nfft):
    nperseg = round(wlen * T.stats.sampling_rate)
    nfft = round(nperseg * fact_nfft)
    noverlap = round(nperseg * per_lap)

    f, t, Sxx = scipy.signal.spectrogram(
        x=T.data, fs=T.stats.sampling_rate, nfft=nfft, nperseg=nperseg, noverlap=noverlap)

    return f, t, Sxx


def process_spectrogram(f, t, Sxx, fmin=None, fmax=None):
    if fmin is None:
        fmin = f[0]
    if fmax is None:
        fmax = f[-1]

    ii = np.where((f >= fmin) & (f <= fmax))

    Sxx_filter = Sxx[ii]
    Sxx_db = np.log(Sxx_filter)*10
    Sxx_centered = Sxx_db - np.mean(Sxx_db)
    Sxx_norm = Sxx_centered/np.std(Sxx_centered)

    return f[ii], t, Sxx_norm


def percentile(Sxx, keep=100):
    vmin = np.percentile(Sxx, (100 - keep)/2)
    vmax = np.percentile(Sxx, (100 + keep)/2)

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


def process_file(file, path, inventory, out_path='spectrograms'):
    filename = file.split('.')[0]
    t1 = timer()

    st = read(os.path.join(path, file))
    preprocess_waveform(st, inventory=inventory)
    T = st.select(component='Z')[0]

    # Bandpass
    f, t, Sxx = spectrogram(T, wlen=100, per_lap=.8, fact_nfft=10)
    f, t, Sxx = process_spectrogram(f, t, Sxx, fmin=.01, fmax=9)
    vmin, vmax = percentile(Sxx, keep=98)

    plot_image(f, t, Sxx, filename=filename,
               path=os.path.join(out_path, 'BP'), vmin=vmin, vmax=vmax, logy=True)

    # High frenquencies
    f, t, Sxx = spectrogram(T, wlen=10, per_lap=.8, fact_nfft=10)
    f, t, Sxx = process_spectrogram(f, t, Sxx, fmin=2, fmax=9)
    vmin, vmax = percentile(Sxx, keep=98)

    plot_image(f, t, Sxx, filename=filename,
               path=os.path.join(out_path, 'HF'), vmin=vmin, vmax=vmax)

    # Middle frenquencies
    f, t, Sxx = spectrogram(T, wlen=50, per_lap=.8, fact_nfft=10)
    f, t, Sxx = process_spectrogram(f, t, Sxx, fmin=.05, fmax=2)
    vmin, vmax = percentile(Sxx, keep=98)

    plot_image(f, t, Sxx, filename=filename,
               path=os.path.join(out_path, 'MF'), vmin=vmin, vmax=vmax, logy=True)

    # Low frenquencies
    f, t, Sxx = spectrogram(T, wlen=100, per_lap=.8, fact_nfft=10)
    f, t, Sxx = process_spectrogram(f, t, Sxx, fmin=.01, fmax=.1)
    vmin, vmax = percentile(Sxx, keep=98)

    plot_image(f, t, Sxx, filename=filename,
               path=os.path.join(out_path, 'LF'), vmin=vmin, vmax=vmax)

    # Waveforms
    T.filter('bandpass', freqmin=2, freqmax=9)

    plot_waveform(T, filename, path=os.path.join(out_path, 'waveforms'))

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
    inventory_path = 'metadata'
    inventory_filename = 'inventory.xml'
    inventory = read_inventory(os.path.join(
        inventory_path, inventory_filename), format='STATIONXML')

    path = 'data'
    files = os.listdir(path)

    process_files_multiprocess(
        files, path, inventory=inventory, n_processes=5, out_path='spectrograms')
