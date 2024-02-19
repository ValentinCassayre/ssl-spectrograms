import os
from timeit import default_timer as timer
from multiprocessing import Process

import numpy as np
import scipy.signal

from obspy import read, read_inventory

from plotting import cmaps, plot_image, plot_waveform, plot_dual


def read_inventory_xml(inventory_filename='inventory.xml', inventory_path='metadata'):
    file = os.path.join(inventory_path, inventory_filename)
    inventory = read_inventory(file, format='STATIONXML')
    
    return inventory

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


def process_spectrogram(f, t, Sxx, Wn=.05, fmin=None, fmax=None):
    if fmin is None:
        fmin = f[0]
    if fmax is None:
        fmax = f[-1]

    ii = np.where((f >= fmin) & (f <= fmax))

    Sxx_filter = Sxx[ii]
    Sxx_db = np.log(Sxx_filter)*10

    b, a = scipy.signal.butter(4, Wn)
    Sxx_smooth = scipy.signal.filtfilt(b, a, Sxx_db, padlen=10)

    return f[ii], t, Sxx_smooth


def percentile(Sxx, p1=100, p2=100):
    vmin = np.percentile(Sxx, (100 - p1))
    vmax = np.percentile(Sxx, p2)

    return vmin, vmax


def slice_trace(T, duration):
    windows = np.arange(T.stats.starttime, T.stats.endtime, duration)

    return [T.slice(window, window + duration) for window in windows]


def process_file(file, path, inventory, out_path='spectrograms'):
    filename = file.split('.')[0]
    t1 = timer()

    st = read(os.path.join(path, file))
    T = st.select(component='Z')[0]
    sampling_rate = T.stats.sampling_rate

    # Middle frenquencies
    duration = 60*60
    wlen = 50
    per_lap = .9
    fact_nfft = 10
    fmin = .05
    fmax = .9
    p1 = 60
    p2 = 98
    logy = True
    Wn = .05
    cmap = cmaps['jet-w-reduced']
    name = 'MF'

    Tslices = slice_trace(T, duration)

    for Tslice in Tslices:
        filename = f'{Tslice.stats.starttime.strftime("%Y_%m_%d_%H_%M_%S")}.png'
        preprocess_waveform(Tslice, inventory=inventory)

        data = Tslice.data

        f, t, Sxx = spectrogram(data, sampling_rate=sampling_rate,
                                wlen=wlen, per_lap=per_lap, fact_nfft=fact_nfft)
        f, t, Sxx = process_spectrogram(f, t, Sxx, Wn=Wn, fmin=fmin, fmax=fmax)
        Sxx = normalize(Sxx)
        vmin, vmax = percentile(Sxx, p1=p1, p2=p2)

        Tfilter = Tslice.copy()
        Tfilter.filter('bandpass', freqmin=fmin, freqmax=fmax)

        plot_dual(f, t, Sxx, Tfilter, filename=filename, path=os.path.join(out_path, name),
                vmin=vmin, vmax=vmax, logy=logy, cmap=cmap)

    # Low frenquencies
    duration = 60*60
    wlen = 100
    per_lap = .9
    fact_nfft = 10
    fmin = .01
    fmax = .08
    p1 = 70
    p2 = 99
    logy = False
    Wn = .07
    cmap = cmaps['jet-w-reduced']
    name = 'LF'

    Tslices = slice_trace(T, duration)

    for Tslice in Tslices:
        filename = f'{Tslice.stats.starttime.strftime("%Y_%m_%d_%H_%M_%S")}.png'
        preprocess_waveform(Tslice, inventory=inventory)

        data = Tslice.data

        f, t, Sxx = spectrogram(data, sampling_rate=sampling_rate,
                                wlen=wlen, per_lap=per_lap, fact_nfft=fact_nfft)
        f, t, Sxx = process_spectrogram(f, t, Sxx, Wn=Wn, fmin=fmin, fmax=fmax)
        Sxx = normalize(Sxx)
        vmin, vmax = percentile(Sxx, p1=p1, p2=p2)

        Tfilter = Tslice.copy()
        Tfilter.filter('bandpass', freqmin=fmin, freqmax=fmax)

        plot_dual(f, t, Sxx, Tfilter, filename=filename, path=os.path.join(out_path, name),
                vmin=vmin, vmax=vmax, logy=logy, cmap=cmap)

    # High frequencies
    duration = 10*60
    wlen = 10
    per_lap = .9
    fact_nfft = 10
    fmin = 2
    fmax = 8
    p1 = 80
    p2 = 95
    logy = False
    Wn = .03
    cmap = cmaps['jet-w-reduced']
    name = 'HF'

    Tslices = slice_trace(T, duration)

    for Tslice in Tslices:
        filename = f'{Tslice.stats.starttime.strftime("%Y_%m_%d_%H_%M_%S")}.png'
        preprocess_waveform(Tslice, inventory=inventory)

        data = Tslice.data

        f, t, Sxx = spectrogram(data, sampling_rate=sampling_rate,
                                wlen=wlen, per_lap=per_lap, fact_nfft=fact_nfft)
        f, t, Sxx = process_spectrogram(f, t, Sxx, Wn=Wn, fmin=fmin, fmax=fmax)
        Sxx = normalize(Sxx)
        vmin, vmax = percentile(Sxx, p1=p1, p2=p2)

        Tfilter = Tslice.copy()
        Tfilter.filter('bandpass', freqmin=fmin, freqmax=fmax)

        plot_dual(f, t, Sxx, Tfilter, filename=filename, path=os.path.join(out_path, name),
                vmin=vmin, vmax=vmax, logy=logy, cmap=cmap)

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
