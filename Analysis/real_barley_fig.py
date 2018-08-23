import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.gridspec as gridspec


def barley_plots(rootdir, saveloc):

    fig = plt.figure(figsize=(8.27, 13.69))
    axes = []

    gs0 = gridspec.GridSpec(1, 1)

    barley1 = '../Figures/Barley/wild_seed_1.png'
    barley2 = '../Figures/Barley/wild_seed_2.png'
    barley3 = '../Figures/Barley/dom_seed_1.png'
    barley4 = '../Figures/Barley/dom_seed_2.png'

    length = '../Plots/{0}/length.png'.format(rootdir)
    width = '../Plots/{0}/width.png'.format(rootdir)
    depth = '../Plots/{0}/depth.png'.format(rootdir)
    volume = '../Plots/{0}/volume.png'.format(rootdir)
    sa = '../Plots/{0}/surface_area.png'.format(rootdir)
    lwd = '../Plots/{0}/Surface Area - Volume.png'.format(rootdir)

    gs1 = gridspec.GridSpecFromSubplotSpec(6, 4,
                                           subplot_spec=gs0[0],
                                           hspace=0.05,
                                           wspace=0.05)

    # Grains
    axes.append(plt.subplot(gs1[0, 0]))
    axes.append(plt.subplot(gs1[0, 1]))
    axes.append(plt.subplot(gs1[1, 0]))
    axes.append(plt.subplot(gs1[1, 1]))

    axes[0].text(10, 70, 'A', color='w', size=20)
    axes[0].imshow(mpimg.imread(barley1), aspect='auto')

    # axes[2].text(10, 60, 'B', color='w')
    axes[2].imshow(mpimg.imread(barley2), aspect='auto')

    # axes[3].text(10, 60, 'D', color='w')
    axes[3].imshow(mpimg.imread(barley3), aspect='auto')

    # axes[1].text(10, 60, 'C', color='w')
    axes[1].imshow(mpimg.imread(barley4), aspect='auto')

    axes.append(plt.subplot(gs1[0:2, 2:]))

    axes[4].text(10, 20, 'B', color='k', size=20)
    axes[4].imshow(mpimg.imread(volume)[15:], aspect='auto')

    axes.append(plt.subplot(gs1[2:4, :2]))
    axes[5].text(10, 20, 'C', color='k', size=20)
    axes[5].imshow(mpimg.imread(length)[15:], aspect='auto')

    axes.append(plt.subplot(gs1[2:4, 2:]))
    axes[6].text(10, 20, 'D', color='k', size=20)
    axes[6].imshow(mpimg.imread(width)[15:], aspect='auto')

    axes.append(plt.subplot(gs1[4:, :2]))
    axes[7].text(10, 20, 'E', color='k', size=20)
    axes[7].imshow(mpimg.imread(depth)[15:], aspect='auto')

    axes.append(plt.subplot(gs1[4:, 2:]))
    axes[8].text(10, 20, 'F', color='k', size=20)
    axes[8].imshow(mpimg.imread(lwd)[15:], aspect='auto')

    [ax.axis('off') for ax in axes]

    plt.tight_layout()
    plt.savefig(
        '/home/nathan/Dropbox/NPPC/Domestication/Figures/{0}.png'.format(saveloc), dpi=400)

    # plt.show()


barley_plots('../Plots/H. spontaneum-H. vulgare', 'barley_fig-2')
