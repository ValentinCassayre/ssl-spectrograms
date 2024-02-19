import os

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt


first = cm.get_cmap('jet', 85)
second = cm.get_cmap('jet', 60)
third = cm.get_cmap('jet', 85)
forth = cm.get_cmap('Reds_r', 26)

colors = np.vstack((first(np.linspace(0, 0.25, 85)),
                       second(np.linspace(0.25, 0.75, 60)),
                       third(np.linspace(0.75, 1, 85)),
                       forth(np.linspace(0, 1, 26))))

cmaps = {'jet-w-reduced' : matplotlib.colors.ListedColormap(colors, name='saturation_3'),
         'wbrg' : matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","powderblue","lightblue","lightskyblue","lightskyblue","yellow","orangered","red","darkred","mistyrose"])}
cmap = cmaps['jet-w-reduced']

def plot_image(f, t, Sxx, filename, path='', vmin=None, vmax=None, logy=False, cmap=cmap, size=(1, 1), dpi=256):
    file = os.path.join(path, filename)

    fig = plt.figure(figsize=(size[0], size[1]), dpi=dpi)

    plt.pcolormesh(t, f, Sxx, cmap=cmap, vmin=vmin, vmax=vmax)
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


def plot_waveform(T, filename, path='', size=(1, 1), dpi=256):
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

def plot_dual(f, t, Sxx, T, filename, path='', vmin=None, vmax=None, logy=False, cmap=cmap, size=(1, 1), dpi=256):
    file = os.path.join(path, filename)
    fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(size[0], size[1]), dpi=dpi)

    ax = axs[0]
    ax.pcolormesh(t, f, Sxx, cmap=cmap, vmin=vmin, vmax=vmax)
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
