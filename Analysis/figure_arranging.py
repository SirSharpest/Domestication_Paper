import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.gridspec as gridspec

spikes = [
    '../Figures/Figure1_Subfigures/{0}.png'.format(f) for f in range(1, 5)]
monos = [
    '../Figures/Figure1_Subfigures/mono-{0}.png'.format(f) for f in range(1, 7)]
dicos = [
    '../Figures/Figure1_Subfigures/dico-{0}.png'.format(f) for f in range(1, 7)]

#fig = plt.figure(figsize=(8.27, 11.69))
axes = []
gs0 = gridspec.GridSpec(2, 2)

for i in range(0, 4):
    gs1 = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=gs0[i//2, i % 2], hspace=0.05, wspace=0.05)
    axes.append(plt.subplot(gs1[0:, 0]))
    axes.append(plt.subplot(gs1[0, 1]))
    axes.append(plt.subplot(gs1[1, 1]))
    axes.append(plt.subplot(gs1[2, 1]))


axes[0].imshow(mpimg.imread(spikes[1]), aspect='auto')
axes[1].imshow(mpimg.imread(monos[3]), aspect='auto')
axes[2].imshow(mpimg.imread(monos[4]), aspect='auto')
axes[3].imshow(mpimg.imread(monos[5]), aspect='auto')
axes[4].imshow(mpimg.imread(spikes[0]), aspect='auto')
axes[5].imshow(mpimg.imread(monos[0]), aspect='auto')
axes[6].imshow(mpimg.imread(monos[1]), aspect='auto')
axes[7].imshow(mpimg.imread(monos[2]), aspect='auto')
axes[8].imshow(mpimg.imread(spikes[3]), aspect='auto')
axes[9].imshow(mpimg.imread(dicos[3]), aspect='auto')
axes[10].imshow(mpimg.imread(dicos[4]), aspect='auto')
axes[11].imshow(mpimg.imread(dicos[5]), aspect='auto')
axes[12].imshow(mpimg.imread(spikes[2]), aspect='auto')
axes[13].imshow(mpimg.imread(dicos[1]), aspect='auto')
axes[14].imshow(mpimg.imread(dicos[2]), aspect='auto')
axes[15].imshow(mpimg.imread(dicos[3]), aspect='auto')


[ax.axis('off') for ax in axes]

plt.tight_layout()
plt.savefig('/home/nathan/Dropbox/NPPC/Domestication/Figures/fig1.png',
            figsize=(8.27, 11.69))
