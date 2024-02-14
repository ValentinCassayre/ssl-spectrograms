import os

import numpy as np
import matplotlib.pyplot as plt


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
