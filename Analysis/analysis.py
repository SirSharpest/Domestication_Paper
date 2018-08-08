"""
This file is to generate the comparisons
outlined by Candida and Hugo which would be interesting

1. Comparing the monococcums
2. Dioccoides and the Dioccums
3. Avestium and the Spelta
4. Monococcum wild and the Dicoccodies
5. dicoccum and durum
6. dicoccum and aestivum
7. durum and aestivum
8. H spontaneum and H vulgare

Having these done may present some interesting data!

"""

# import library object
from ct_analysing_library.ct_data import CTData as ctd
from ct_analysing_library.statistical_tests import baysian_hypothesis_test, perform_t_test
from ct_analysing_library.graphing import plot_difference_of_means, plot_forest_plot, plot_pca
from ct_analysing_library.data_transforms import perform_pca
from numpy import array as nparr
from scipy import stats
from scipy.stats import boxcox as normalize
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from pylab import cm


def split_on_two_sample_types(df, type1, type2):
    # small func to split on two criteria to reduce code reuse
    result = df[(df['Sample Type'] == type1) | (df['Sample Type'] == type2)]
    result.reset_index(drop=True, inplace=True)
    return result


def aggregate_average_attribute(df, att, median=False):
    if median:
        return df.groupby(['Sample name', 'Sample Type', 'Wild/Domesticated'],
                          as_index=False)[[att]].median()
    return df.groupby(['Sample name', 'Sample Type', 'Wild/Domesticated'],
                      as_index=False)[[att]].mean()


def split_into_groups(df, groups):

    # small func to split on two criteria to reduce code reuse
    result = df[(df['Sample Type'] == groups[0][0]) |
                (df['Sample Type'] == groups[0][1]) |
                (df['Sample Type'] == groups[1][0]) |
                (df['Sample Type'] == groups[1][1]) |
                (df['Sample Type'] == groups[2][0]) |
                (df['Sample Type'] == groups[2][1])]
    result.reset_index(drop=True, inplace=True)
    return result


def plot_spike(df, spikename, ax=None, mi=None, mx=None):

    if ax is None:
        fig, ax = plt.subplots()
    x = df[df['Sample name'] == spikename]['x']
    y = df[df['Sample name'] == spikename]['z']
    s = df[df['Sample name'] == spikename]['volume']

    if mi is None:
        mi = min(s)
        mx = max(s)
    norm = np.array([((i - mi) / (mx-mi)) * 75 for i in s])
    colors_to_use = cm.rainbow(norm/max(norm))
    colmap = cm.ScalarMappable(cmap=cm.rainbow)
    colmap.set_array(colors_to_use)
    ax.scatter(x, y, c=colors_to_use, s=norm, marker='o')
    ax.set_xlim(0, 512)
    fid = list(df[df['Sample name'] == spikename]['folderid'])[0]
    ax.set_title('{0}\n{1}'.format(spikename, fid))

    # if fig:
    #     fig.colorbar(colmap)
    #     return (fig, ax)
    return colmap


def plot_all_spikes_by_geno(df, spike_type):

    tdf = df[df['Sample Type'] == spike_type]
    num_spikes = len(tdf['Sample name'].unique())
    fig, axes = plt.subplots((num_spikes//2)+1, 2,
                             sharex=True, sharey=True, figsize=(4, 50))

    mi = tdf['volume'].min()
    mx = tdf['volume'].max()
    for idx, s in enumerate(tdf['Sample name'].unique()):
        plot_spike(tdf, s, ax=axes[idx//2, idx % 2], mi=mi, mx=mx)


def plot_boxplot(data, attribute, p=None, **kwargs):
    """
    This should just create a single boxplot and return the figure
    and an axis, useful for rapid generation of single plots
    Rather than the madness of the plural function

    Accepts Kwargs for matplotlib and seaborn

    @param data a CTData object or else a dataframe
    @param attribute the attribute to use in the boxplot
    @param **kwargs keyword arguments for matplotlib
    @returns a figure and axes
    """
    fig, ax = plt.subplots(1)
    if type(data) is not DataFrame:
        sns.boxplot(data=data.get_data(), y=attribute,
                    hue='Sample Type', **kwargs)
    else:
        data = data.sort_values(by=['Wild/Domesticated'])
        sns.boxplot(data=data, y=attribute, hue='Wild/Domesticated',  **kwargs)
        # calculate pval

        g1 = data['Sample Type'].unique()[0]
        g2 = data['Sample Type'].unique()[1]

        v1 = (data[data['Sample Type'] == g1][attribute])
        v2 = (data[data['Sample Type'] == g2][attribute])

        if p is None:
            t, p = stats.ttest_ind(v1, v2, equal_var=True)

        if 'mean' in attribute or 'median' in attribute or 'height' in attribute or 'count' in attribute:
            n1 = len(data[data['Sample Type'] == g1]['Sample name'].unique())
            n2 = len(data[data['Sample Type'] == g2]['Sample name'].unique())
        else:
            n1 = len(data[data['Sample Type'] == g1])
            n2 = len(data[data['Sample Type'] == g2])
        plt.gcf().suptitle(
            'P-value {0} for {1}\n| N1 = {2} | N2 = {3}'.format(p,
                                                                attribute,
                                                                n1, n2))
    fig.tight_layout()
    return (fig, ax, p)


def make_plots(df, compare_groups, atts):
    # for each group comparison
    results = {}
    img_locations = {}
    pvalues = {}
    idx = 1
    for g1, g2 in compare_groups:
        # prep data storage
        rootdir = '/home/nathan/Dropbox/NPPC/Domestication/Final Plots'
        directory = '{0}/{1}_{2}'.format(rootdir,
                                         g1, g2)

        # split data
        d = split_on_two_sample_types(df, g1, g2)
        bayes = {}
        pvals = {}
        tmp_gp_loc = {}
        # for each attribute
        for a in atts:
            if 'median' in a:
                directory = '{0}/{1}/{2}'.format(directory,
                                                 '/_spike_medians',
                                                 a)
                dmed = aggregate_average_attribute(d, a, median=True)
                d1 = nparr(dmed[dmed['Sample Type'] == g1][a])
                d2 = nparr(dmed[dmed['Sample Type'] == g2][a])
                d1 = normalize(list(dmed[dmed['Sample Type'] == g1][a]))[0]
                d2 = normalize(list(dmed[dmed['Sample Type'] == g2][a]))[0]
            elif 'mean' in a:
                directory = '{0}/{1}/{2}'.format(directory,
                                                 '/_spike_means',
                                                 a)
                dmean = aggregate_average_attribute(d, a)
                d1 = nparr(dmean[dmean['Sample Type'] == g1][a])
                d2 = nparr(dmean[dmean['Sample Type'] == g2][a])
                #d1 = normalize(list(dmean[dmean['Sample Type'] == g1][a]))[0]
                #d2 = normalize(list(dmean[dmean['Sample Type'] == g2][a]))[0]
            else:
                directory = '{0}/{1}'.format(directory, a)
                d1 = nparr(d[d['Sample Type'] == g1][a])
                d2 = nparr(d[d['Sample Type'] == g2][a])
                #d1 = normalize(list(d[d['Sample Type'] == g1][a]))[0]
                #d2 = normalize(list(d[d['Sample Type'] == g2][a]))[0]

            if not os.path.exists(directory):
                os.makedirs(directory)

            _, p = stats.ttest_ind(d1, d2, equal_var=False)
            # res, summ = baysian_hypothesis_test(d1, d2, g1, g2)

            # bayes[a] = (res['difference of means'] < 0).mean() if (
            #     res['difference of means'] < 0).mean() < 0.5 else (1-(res['difference of means'] < 0).mean())

            # fp = plot_forest_plot(res, res.varnames[0], res.varnames[1])
            # plt.gcf().suptitle('{0} - 95% Credible Interval'.format(a))
            # plt.gca().set_title('')
            # plt.gcf().savefig('{0}/bayes_{1}.png'.format(directory, a))

            # dm = plot_difference_of_means(res)
            # plt.gca().set_title('')
            # plt.gcf().suptitle('{0} - Difference of Means'.format(a))
            # plt.gcf().savefig(
            #     '{0}/bayes_difference_of_means_{1}.png'.format(directory, a))

            fig, ax, p = plot_boxplot(d, a, x='Sample Type', p=p)
            fig.tight_layout()
            fig.savefig('{0}/{1}.png'.format(directory, a))
            tmp_gp_loc['{0}'.format(a)] = '{0}/{1}.png'.format(directory, a)
            directory = '{0}/{1}_{2}'.format(rootdir, g1, g2)
            pvals[a] = p
            idx = idx+1
        pvalues['{0}+{1}'.format(g1, g2)] = pvals
        results['{0}+{1}'.format(g1, g2)] = bayes
        img_locations['{0}_{1}'.format(g1, g2)] = tmp_gp_loc

    # Now do on spike averages
    return (img_locations, results, pvalues)


def figure_arrange(locations):
    orig_atts = ['length', 'width', 'depth', 'volume',
                 'surface_area', 'length_depth_width']
    for g, a in locations.items():
        with PdfPages('../{0}.pdf'.format(g)) as pdf:
            counter = 0
            for att, loc in a.items():
                if att not in orig_atts:
                    continue
                if counter == 0:
                    im1 = mpimg.imread(loc)
                    t1 = att
                    im2 = mpimg.imread(a['mean_{0}'.format(t1)])
                    t2 = 'mean_{0}'.format(t1)
                    counter = counter + 1

                # elif counter == 1:
                    fig, ax = plt.subplots(2, 1, figsize=(11.69, 8.27))
                    fig.suptitle(g)

                    ax[0].imshow(im1)
                    ax[1].imshow(im2)
                    ax[0].set_title(t1)
                    ax[1].set_title(t2)
                    ax[0].axis('off')
                    ax[1].axis('off')
                    pdf.savefig(dpi=300)
                    plt.close('all')
                    counter = 0


def make_pca_figures(df, atts, groups):

    atts = ['median_width', 'median_depth', 'median_volume',
            'median_surface_area', 'median_length_depth_width']

    rootdir = '/home/nathan/Dropbox/NPPC/Domestication/Final Plots'
    dfx, pca, misc = perform_pca(df,
                                 atts,
                                 'Sample Type',
                                 standardise=True)
    g = plot_pca(pca, dfx, 'Sample Type', single_plot=True)
    g.savefig('{0}/{1}'.format(rootdir, 'pca_plot.png'))

    for g1, g2 in groups:
        # prep data storage
        rootdir = '/home/nathan/Dropbox/NPPC/Domestication/Final Plots'
        directory = '{0}/{1}_{2}'.format(rootdir,
                                         g1, g2)
        d = split_on_two_sample_types(df, g1, g2)
        dfx, pca, misc = perform_pca(d,
                                     atts,
                                     'Sample Type',
                                     standardise=True)
        g = plot_pca(pca, dfx, 'Sample Type', single_plot=True)
        g.savefig('{0}/{1}'.format(directory, 'pca_plot.png'))
        plt.close('all')


def do_analysis(df, compare_groups, atts):
    locations, results, pvalues = make_plots(df, compare_groups, atts)
    pd.DataFrame.from_dict(results, orient='index').to_csv(
        '/home/nathan/Dropbox/NPPC/Domestication/Bayesian_Testing.csv')
    pd.DataFrame.from_dict(pvalues, orient='index').round(4).to_csv(
        '/home/nathan/Dropbox/NPPC/Domestication/T-testing.csv')
    figure_arrange(locations)


if __name__ == "__main__":
    # give data locations
    dloc = '/home/nathan/Dropbox/NPPC/Domestication/Analysis/Data'
    dinfo = '/home/nathan/Dropbox/NPPC/Domestication/Analysis/info.xlsx'

    # select attributes we are interested in
    atts = ['length', 'width', 'depth', 'volume',
            'surface_area', 'length_depth_width']

    data = ctd(dloc, True)
    # add spike info
    data.get_spike_info(dinfo)
    # fix colnames
    data.fix_colnames()
    data.clean_data(remove_large=True, remove_small=True)
    data.clean_data(remove_large=True, remove_small=True)
    data.join_spikes_by_rachis()
    data.aggregate_spike_averages(atts, 'Sample name')

    # extract just the data frame
    df = data.get_data()

    # Add aggregated values to atts
    atts.extend(['median_length', 'median_width', 'median_depth',
                 'median_volume', 'median_surface_area',
                 'median_length_depth_width', 'mean_length', 'mean_width', 'mean_depth',
                 'mean_volume', 'mean_surface_area', 'mean_length_depth_width', 'std_length', 'std_width', 'std_depth', 'std_volume', 'std_length_depth_width'])

    # # Then start to compare

    # Remap the namings so that they display well in results
    df['Sample Type'] = df['Sample Type'].map(
        {'T_monococcum_': 'T. monococcum',
         'T_monococcum_wild_': 'T. beoticum',
         'T_dicoccum_': 'T. dicoccum',
         'T_dicoccoides_': 'T. dicoccoides',
         'T_aestivum_': 'T. aestivum',
         'T_spelta_': 'T. spelta',
         'T_durum_': 'T. durum',
         'H_spontaneum_wild_': 'H. spontaneum',
         'H_vulgare_': 'H. vulgare'})

    writer = pd.ExcelWriter('../Aestivum.xlsx')

    df[df['Sample Type'] == 'T. aestivum'].to_excel(writer)

    writer.save()
    writer.close()

    # # Drop the na's
    # compare_groups = [('T. monococcum', 'T. beoticum'),
    #                   ('T. dicoccum', 'T. dicoccoides'),
    #                   ('H. spontaneum', 'H. vulgare')]

    # do_analysis(df, compare_groups, atts)
    # for f in ['T. monococcum', 'T. beoticum', 'T. dicoccum', 'T. dicoccoides', 'H. spontaneum', 'H. vulgare']:
    #     plot_all_spikes_by_geno(df, f)
    #     plt.tight_layout()
    #     plt.savefig('{0}.pdf'.format(f))
    #     plt.close('all')

    # #loc, r, p = make_plots(df, compare_groups, atts)

    # make the multi fig
