import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.gridspec as gridspec


def summary_plots(rootdir, saveloc, barley=False):
    if barley:
        fig = plt.figure(figsize=(8.27, 13.69))
    else:
        fig = plt.figure(figsize=(12.69, 6.27))
        axes = []
    gs0 = gridspec.GridSpec(1, 1)

    if barley:
        barley1 = '../Figures/Barley/wild_barley.png'
        barley2 = '../Figures/Barley/dom_barley.png'

    length = '../Plots/{0}/length.png'.format(rootdir)
    width = '../Plots/{0}/width.png'.format(rootdir)
    depth = '../Plots/{0}/depth.png'.format(rootdir)
    volume = '../Plots/{0}/volume.png'.format(rootdir)
    sa = '../Plots/{0}/surface_area.png'.format(rootdir)
    lwd = '../Plots/{0}/Surface Area - Volume Ratio.png'.format(rootdir)

    gs1 = gridspec.GridSpecFromSubplotSpec(
        (6 if barley else 2), (2 if barley else 3), subplot_spec=gs0[0], hspace=0.01, wspace=0.01)

    if barley:
        axes.append(plt.subplot(gs1[3:, 0]))
        axes.append(plt.subplot(gs1[3:, 1]))

    axes.append(plt.subplot(gs1[0, 0]))
    axes.append(plt.subplot(gs1[0, 1]))
    axes.append(plt.subplot(gs1[0, 2]))

    axes.append(plt.subplot(gs1[1, 0]))
    axes.append(plt.subplot(gs1[1, 1]))
    axes.append(plt.subplot(gs1[1, 2]))

    if barley:
        axes[0].imshow(mpimg.imread(barley1), aspect='auto')
        axes[1].imshow(mpimg.imread(barley2), aspect='auto')
        axes[2].imshow(mpimg.imread(length))
        axes[3].imshow(mpimg.imread(width))
        axes[4].imshow(mpimg.imread(depth))
        axes[5].imshow(mpimg.imread(volume))
        axes[6].imshow(mpimg.imread(sa))
        axes[7].imshow(mpimg.imread(lwd))
    else:
        axes[0].imshow(mpimg.imread(length))
        axes[1].imshow(mpimg.imread(width))
        axes[2].imshow(mpimg.imread(depth))
        axes[3].imshow(mpimg.imread(volume))
        axes[4].imshow(mpimg.imread(sa))
        axes[5].imshow(mpimg.imread(lwd))

    [ax.axis('off') for ax in axes]

    plt.tight_layout()
    plt.savefig(
        '/home/nathan/Dropbox/NPPC/Domestication/Figures/{0}.pdf'.format(saveloc), dpi=400)
    plt.savefig(
        '/home/nathan/Dropbox/NPPC/Domestication/Figures/{0}.png'.format(saveloc), dpi=400)


summary_plots('T. monococcum-T. beoticum', 'einkorn')
summary_plots('T. dicoccum-T. dicoccoides', 'emmer')
summary_plots('T. beoticum-T. dicoccoides', 'wild_einkorn_wild_emmer')
summary_plots('T. monococcum-T. dicoccum', 'dom_einkorn_dom_emmer')
summary_plots('2n-4n-6n', 'ploidy')
#summary_plots('H. spontaneum-H. vulgare', 'barley', barley=True)
