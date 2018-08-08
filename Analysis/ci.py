import pandas as pd
import os
from scipy import stats
import numpy as np
from math import sqrt
from scipy.stats import t
from scipy import std, mean
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import cm
from ct_analysing_library.data_transforms import perform_pca
from scipy.spatial import ConvexHull
import itertools
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_pdf import PdfPages
from ct_analysing_library.statistical_tests import baysian_hypothesis_test
from ct_analysing_library.graphing import plot_difference_of_means, plot_forest_plot, plot_pca
from half_viol import half_violinplot

plt.style.use('ggplot')


def split_on_two_sample_types(df, type1, type2):
    result = df[(df['Sample Type'] == type1) | (df['Sample Type'] == type2)]
    result.reset_index(drop=True, inplace=True)
    return result


def data_to_groups(orig_df, groups, attribute):

    if 'mean' in attribute:
        attribute = attribute.replace('mean_', '')
        df = aggregate_average_attribute(orig_df, attribute)
    else:
        df = orig_df

    d1 = df[df['Sample Type'] == groups[0]][attribute]
    d2 = df[df['Sample Type'] == groups[1]][attribute]
    return (d1, d2)


def aggregate_average_attribute(df, att):
    return df.groupby(['Sample name', 'Sample Type', 'Wild/Domesticated', 'Ploidy'],
                      as_index=False)[[att]].mean()


def make_top_bottom(df):
    def top_bottom(s):
        return 'bottom' if s['z'] < s['height']/2 else 'top'

    def allocate_position(s):
        return s['z']//(s['slices']//10)
    df['top/bottom'] = df.apply(top_bottom, axis=1)
    df['position'] = df.apply(allocate_position, axis=1)


def make_pca_figures(orig_df, atts, groups):
    plt.close('all')
    atts = ['width', 'length', 'depth', 'volume',
            'surface_area', 'length_depth_width']

    # df = orig_df.groupby(['Sample name', 'Sample Type', 'Wild/Domesticated'],
    #                      as_index=False)[atts].mean()

    df = orig_df

    dfx, pca, misc = perform_pca(df,
                                 atts,
                                 'Sample Type',
                                 standardise=True)
    g = plot_pca(pca, dfx, 'Sample Type', single_plot=True)

    return g, dfx, pca
    # plt.show()
    # g.savefig('{0}/{1}'.format(rootdir, 'pca_plot.png'))


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height,
                    angle=theta, **kwargs, linewidth=2)

    ax.add_artist(ellip)
    return ellip


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def make_bayesian_plots(df, compare_groups, atts):

    df = df.sort_values(by='Wild/Domesticated')
    g1 = compare_groups[0]
    g2 = compare_groups[1]
    directory = '../Plots/{0}-{1}/Bayesian'.format(compare_groups[0],
                                                   compare_groups[1])
    bayes = {}
    if not os.path.exists(directory):
        os.makedirs(directory)

    for a in atts:

        try:
            d1, d2 = data_to_groups(df, compare_groups, a)
            d1 = np.array(d1)
            d2 = np.array(d2)
            res, summ = baysian_hypothesis_test(
                d1, d2, g1, g2, n_samples=10000)
            bayes[a] = (res['difference of means'] < 0).mean() if (
                res['difference of means'] < 0).mean() < 0.5 else (1-(res['difference of means'] < 0).mean())

            dm = plot_difference_of_means(res)
            plt.gca().set_title('')
            plt.gcf().suptitle('{0} - Difference of Means'.format(a))
            plt.gcf().savefig(
                '{0}/bayes_difference_of_means_{1}.png'.format(directory, a))
        except:
            bayes[a] = 'NaN'
            print('{0}-{1}\t didnt work'.format(g1+g2, a))

    return bayes


def boxplot(df, groups, attribute, show=False, saveloc=None, split=False):

    if split:
        df = split_on_two_sample_types(df, groups[0], groups[1])

    if 'mean' in attribute or 'median' in attribute or 'density' in attribute or 'height' in attribute or 'count' in attribute:
        n1 = len(df[df['Sample Type'] == groups[0]]['Sample name'].unique())
        n2 = len(df[df['Sample Type'] == groups[1]]['Sample name'].unique())
    else:
        n1 = len(df[df['Sample Type'] == groups[0]])
        n2 = len(df[df['Sample Type'] == groups[1]])

    if len(df['Wild/Domesticated'].unique()) == 1:
        df = df.sort_values(by=['Ploidy'])
    else:
        df = df.sort_values(by=['Wild/Domesticated'], ascending=False)

    fig, ax = plt.subplots(figsize=(5.5, 4))

    if not split:
        sns.boxplot(data=df, x='Ploidy', y=attribute,
                    ax=ax, hue='Wild/Domesticated')

    elif len(df['Wild/Domesticated'].unique()) == 1:
        sns.boxplot(data=df, x='Sample Type', y=attribute,
                    ax=ax, hue='Ploidy')
    else:
        sns.boxplot(data=df, x='Sample Type', y=attribute,
                    ax=ax, hue='Wild/Domesticated')

    if split:
        p = do_test(df, groups, attribute)[0]
    else:
        p = 1
    title = attribute.capitalize()
    fig.suptitle(title.replace('_', ' '))

    if attribute != 'length':
        ax.legend().set_visible(False)

    if attribute in ['length', 'width', 'depth']:
        ax.set_xlabel('')
        ax.set_ylabel('{0} cm'.format(attribute.capitalize()))
    if attribute == 'volume':
        ax.set_ylabel(r'{0} $cm^3$'.format(attribute.capitalize()))
    if attribute == 'length_depth_width':
        ax.set_ylabel(r'Length X Width x Depth $cm^3$')
    if attribute == 'surface_area':
        ax.set_ylabel(r'Surface Area $cm^2$')
        ax.set_xlabel('Surface Area')

    if show:
        plt.show(block=False)
    if saveloc:
        fig.savefig(saveloc)


def do_test(df, groups, attribute, say=False):
    d1, d2 = data_to_groups(df, groups, attribute)
    ind_t_test = stats.ttest_ind(d1, d2, equal_var=False)
    N1 = len(d1)
    N2 = len(d2)
    degfree = (N1 + N2 - 2)
    std1 = std(d1)
    std2 = std(d2)
    std_N1N2 = sqrt(((N1 - 1)*(std1)**2 + (N2 - 1)*(std2)**2) / degfree)
    diff_mean = (mean(d1) - mean(d2))
    MoE = t.ppf(0.975, degfree) * std_N1N2 * sqrt(1/N1 + 1/N2)
    tval = np.around(ind_t_test[0], decimals=4)
    pval = np.around(ind_t_test[1], decimals=4)
    low_ci = np.around(diff_mean - MoE, decimals=4)
    high_ci = np.around(diff_mean + MoE, decimals=4)
    if say:
        print('The results of the independent t-test are: \n\tt-value = {:4.3f}\n\tp-value = {:4.3f}'.format(
            tval, pval))
        print('\nThe difference between groups is {:3.1f} [{:3.1f} to {:3.1f}] (mean [95% CI])'.format(
            diff_mean, low_ci, high_ci))

    return (pval, tval, diff_mean, low_ci, high_ci)


def make_results_table(df, groups, attributes_to_test, say=False):
    results = pd.DataFrame(columns=['pval',
                                    'tval',
                                    'diff_mean',
                                    '0.25',
                                    '0.975'])
    for attribute in attributes_to_test:
        pval, tval, diff_mean, lower_ci, upper_ci = do_test(
            df, groups, attribute, say=say)
        s = pd.Series({'pval': pval,
                       'tval': tval,
                       'diff_mean': diff_mean,
                       '0.25': lower_ci,
                       '0.975': upper_ci})
        s.name = attribute
        results = results.append(s)
    return results


def make_violin_plots(df_orig, att, compare_groups, saveloc):
    if 'mean' in att or 'std' in att or 'height' in att or 'density' in att:
        df = aggregate_average_attribute(df_orig, att)
    else:
        df = df_orig.copy(deep=True)
    df = df.sort_values(
        by=['Wild/Domesticated', 'Ploidy'], ascending=[False, True])
    f, ax = plt.subplots(1, figsize=(8, 10))
    ax = half_violinplot(data=df, x='Sample Type',
                         y=att, bw='scott',  linewidth=1, cut=0.5,
                         scale="area", width=.4, inner=None)

    ax = sns.stripplot(data=df, x='Sample Type', y=att,
                       edgecolor="white", size=2, jitter=1, zorder=0)

    ax = sns.boxplot(data=df, x='Sample Type', y=att,
                     zorder=10, showcaps=True,
                     boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                     saturation=1, width=.15,
                     hue=('Wild/Domesticated' if len(df['Ploidy'].unique()) <= 1 else 'Ploidy'))
    sns.despine(left=True)
    f.savefig(saveloc)


def make_boxplots(df, groups, atts, split=True):

    if split:
        t_df = split_on_two_sample_types(df, groups[0], groups[1])
        basedir = '../Plots/{0}-{1}'.format(groups[0], groups[1])
    else:
        t_df = df
        basedir = '../Plots/2n-4n-6n'

    if not os.path.exists(basedir):
        os.makedirs(basedir)
    for a in atts:
        saveloc = '{0}/{1}.png'.format(basedir, a)
        boxplot(t_df, groups, a, saveloc=saveloc, split=split)
        saveloc = '{0}/{1}-half-violin.png'.format(basedir, a)
        # make_violin_plots(t_df, a, groups, saveloc)


def analyse_all(einkorn, emmer, barley, compare_groups, atts, aestivum=None):
    einkorn_results = make_results_table(einkorn, compare_groups[0], atts)
    emmer_results = make_results_table(emmer, compare_groups[1], atts)
    barley_results = make_results_table(barley, compare_groups[2], atts)
    wild_einkorn_wild_emmer_results = make_results_table(
        pd.concat([emmer, einkorn]), compare_groups[3], atts)
    dom_einkorn_dom_emmer_results = make_results_table(
        pd.concat([emmer, einkorn]), compare_groups[4], atts)

    writer = pd.ExcelWriter('../Results/Statistical_Results.xlsx')

    einkorn_results.to_excel(writer, sheet_name='Einkorn Results')
    emmer_results.to_excel(writer, sheet_name='Emmer Results')
    barley_results.to_excel(writer, sheet_name='Barley Results')
    wild_einkorn_wild_emmer_results.to_excel(
        writer, sheet_name='Wild Einkorn - Wild Emmer Results')
    dom_einkorn_dom_emmer_results.to_excel(
        writer, sheet_name='Dom Einkorn - Dom Emmer Results')
    writer.save()
    writer.close()

    if aestivum is not None:
        make_boxplots(pd.concat([einkorn, emmer, aestivum]),
                      compare_groups, atts, split=False)

    make_boxplots(einkorn, compare_groups[0], atts)
    make_boxplots(emmer, compare_groups[1], atts)
    make_boxplots(barley, compare_groups[2], atts)
    make_boxplots(pd.concat([emmer, einkorn]), compare_groups[3], atts)
    make_boxplots(pd.concat([emmer, einkorn]), compare_groups[4], atts)

    # b_name = '/home/nathan/Dropbox/NPPC/Domestication/Bayesian_Testing.xlsx'
    # writer = pd.ExcelWriter(b_name)

    # def write(x, y): return pd.DataFrame.from_dict(
    #     x, orient='index').to_excel(writer, sheet_name=y)

    # write(make_bayesian_plots(einkorn, compare_groups[0], atts), 'einkorn')
    # write(make_bayesian_plots(emmer, compare_groups[1], atts), 'emmer')
    # write(make_bayesian_plots(barley, compare_groups[2], atts), 'barley')
    # write(make_bayesian_plots(pd.concat(
    #     [emmer, einkorn]), compare_groups[3], atts), 'wild einkorn - wild emmer')
    # write(make_bayesian_plots(pd.concat(
    #     [emmer, einkorn]), compare_groups[4], atts), 'dom einkorn - dom emmer')
    # writer.save()
    # writer.close()


def plot_spikes(df, sample_type, ax=None, mi=None, mx=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    x = df[df['Sample Type'] == sample_type]['x']
    y = df[df['Sample Type'] == sample_type]['z']
    s = df[df['Sample Type'] == sample_type]['volume']

    if mi is None:
        mi = min(s)
        mx = max(s)
    norm = np.array([((i - mi) / (mx-mi)) * 100 for i in s])
    colors_to_use = cm.rainbow(norm/max(norm))
    colmap = cm.ScalarMappable(cmap=cm.rainbow)
    colmap.set_array(colors_to_use)
    t = ax.scatter(x, y, c=colors_to_use, s=norm, marker='o')
    ax.set_xlim(150, 400)

    if fig is not None:
        fig.colorbar(colmap)
        return (fig, ax)
    return colmap


def make_figure_2(df, sample_types, saveloc):
    fig, ax = plt.subplots(1, 2, sharey=True,
                           sharex=True, figsize=(10, 14))
    mi = df['volume'].min()
    mx = df['volume'].max()
    me = df['volume'].mean()
    plot_spikes(df, sample_types[0], ax=ax[1], mi=mi, mx=mx)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    colbar = plot_spikes(df, sample_types[1], ax=ax[0], mi=mi, mx=mx)
    colbar = fig.colorbar(colbar, ticks=[0, 0.5, 1], cax=cbar_ax)

    colbar.ax.set_yticklabels([r'{0:3.2f}mm$^3$'.format(mi),
                               r'{0:3.2f}mm$^3$'.format(me),
                               r'{0:3.2f}mm$^3$'.format(mx)])
    ax[0].set_xlim(150, 400)
    ax[1].set_xlim(150, 400)

    ax[0].set_ylabel('Position along Spike')

    ax[0].set_title(sample_types[1])
    ax[1].set_title(sample_types[0])

    fig.savefig(saveloc)


def fig2_b(df_orig,  ax=None, mi=None, mx=None, use_fig=False, fig=None):

    df = df_orig.copy(deep=True)

    mono_names = list(filter(
        lambda n: True if 'wild' not in n else False, einkorn['Sample name'].unique()))[:13]

    mono_names.extend(list(filter(
        lambda n: True if 'wild' in n else False, einkorn['Sample name'].unique())))

    df = df[df['Sample name'].isin(mono_names)]

    df.ix[df['Sample Type'] == 'T. monococcum',
          'x'] = df[df['Sample Type'] == 'T. monococcum']['x']+512

    if ax is None:
        fig, ax = plt.subplots()

    x = df['x']
    y = df['z']
    s = df['volume']

    if mi is None:
        mi = min(s)
        mx = max(s)
    norm = np.array([((i - mi) / (mx-mi)) * 100 for i in s])
    colors_to_use = cm.rainbow(norm/max(norm))
    colmap = cm.ScalarMappable(cmap=cm.rainbow)
    colmap.set_array(colors_to_use)
    t = ax.scatter(x, y, c=colors_to_use, s=norm, marker='o')

    if not use_fig:
        colbar = fig.colorbar(colmap, ticks=[0, 0.5, 1])
        colbar.ax.set_yticklabels([r'{0:3.2f}mm$^3$'.format(mi),
                                   r'{0:3.2f}mm$^3$'.format(
                                       df['volume'].mean()),
                                   r'{0:3.2f}mm$^3$'.format(mx)])

        x1 = [300, 800]
        squad = ['T. beoticum', 'T. monococcum']
        ax.set_xticks(x1)
        ax.set_xticklabels(squad, minor=False)
        return (fig, ax)

    else:
        colbar = fig.colorbar(colmap, ticks=[0, 0.5, 1], ax=ax)
        colbar.ax.set_yticklabels([r'{0:3.2f}mm$^3$'.format(mi),
                                   r'{0:3.2f}mm$^3$'.format(
                                       df['volume'].mean()),
                                   r'{0:3.2f}mm$^3$'.format(mx)])

        x1 = [300, 800]
        squad = ['T. beoticum', 'T. monococcum']
        ax.set_xticks(x1)
        ax.set_xticklabels(squad, minor=False)

    return colmap


def fig3(einkorn):
    fig, axes = plt.subplots(3, 2)

    fig2_b(einkorn,
           ax=axes[0, 0],
           use_fig=True, fig=fig)

    einkorn = einkorn.sort_values(by='Wild/Domesticated', ascending=False)
    einkorn = einkorn.sort_values(by='top/bottom')

    sns.barplot(data=einkorn, x='Sample Type',
                y='density', ax=axes[1, 0])
    sns.barplot(data=einkorn, x='Sample Type',
                y='grain_count', ax=axes[0, 1])
    sns.barplot(data=einkorn, x='Sample Type',
                y='sum_volume', ax=axes[1, 1])

    palette = itertools.cycle(sns.color_palette())
    sns.barplot(data=einkorn[einkorn['Wild/Domesticated'] == 'wild'],
                x='position',
                y='grain_count', ax=axes[2, 0], color=next(palette))

    sns.barplot(data=einkorn[einkorn['Wild/Domesticated'] == 'domesticated'],
                x='position',
                y='grain_count', ax=axes[2, 1], color=next(palette))

    axes[2, 0].set_ylim(0, 35)
    axes[2, 1].set_ylim(0, 35)

    axes[0, 0].set_xlabel('')
    axes[0, 1].set_xlabel('')

    axes[0, 0].set_title('Grain Volume along Spike')
    axes[0, 1].set_title('Grain Count')
    axes[1, 0].set_title('Grain Density')
    axes[1, 1].set_title('Total Grain Volume Per Spike')
    plt.show()


def make_all_pca(einkorn, emmer, barley):

    g, dfx, pca = pca_figure(pd.concat(
        [einkorn, emmer]), atts, compare_groups, saveloc='../Results/pca_einkorn_emmer.pdf')
    writer = pd.ExcelWriter('../Results/PCA_Results_For_All_Wheat.xlsx')
    pd.DataFrame(pca.components_.T, columns=['PC-1', 'PC-2'],
                 index=['mean_width', 'mean_length',
                        'mean_depth', 'mean_volume',
                        'mean_surface_area', 'mean_length_depth_width']).to_excel(writer, sheet_name='Emmer and Einkorn')

    g, dfx, pca = pca_figure(pd.concat(
        [einkorn]), atts, compare_groups, saveloc='../Results/pca_einkorn.pdf')
    pd.DataFrame(pca.components_.T, columns=['PC-1', 'PC-2'],
                 index=['mean_width', 'mean_length',
                        'mean_depth', 'mean_volume',
                        'mean_surface_area', 'mean_length_depth_width']).to_excel(writer, sheet_name='Einkorn')

    g, dfx, pca = pca_figure(
        pd.concat([emmer]), atts, compare_groups, saveloc='../Results/pca_emmer.pdf')
    pd.DataFrame(pca.components_.T, columns=['PC-1', 'PC-2'],
                 index=['mean_width', 'mean_length',
                        'mean_depth', 'mean_volume',
                        'mean_surface_area', 'mean_length_depth_width']).to_excel(writer, sheet_name='Emmer')

    g, dfx, pca = pca_figure(
        pd.concat([barley]), atts, compare_groups, saveloc='../Results/pca_barley.pdf')
    pd.DataFrame(pca.components_.T, columns=['PC-1', 'PC-2'],
                 index=['mean_width', 'mean_length',
                        'mean_depth', 'mean_volume',
                        'mean_surface_area', 'mean_length_depth_width']).to_excel(writer, sheet_name='Barley')

    writer.save()
    writer.close()


def pca_figure(df, atts, compare_groups, saveloc=None):
    from pandas.plotting import table

    x = df.reset_index(drop=True)
    g, dfx, pca = make_pca_figures(x, atts, compare_groups)

    plt.gca().legend(loc='upper left')

    def find_points(x): return [
        x['principal component 1'], x['principal component 2']]

    palette = itertools.cycle(sns.color_palette())

    for u in dfx['Sample Type'].unique():
        points = list(dfx[dfx['Sample Type'] == u].apply(find_points, axis=1))
        hull = ConvexHull(points)
        points = np.array(points)
        plot_point_cov(points, nstd=1.5, alpha=0.3,
                       color=next(palette), ax=g.ax)

    d = np.around(pd.DataFrame(pca.components_.T, columns=['PC-1', 'PC-2'],
                               index=['Width', 'Length', 'Depth', 'Volume',
                                      'Surface A      .', 'L x W x D']), 2)
    p = np.around(d.values, 2)
    # p = abs(p)
    normalized = (p-p.min())/(p.max()-p.min())
    mtable = table(plt.gca(), d, loc='right', colWidths=[
        0.2, 0.2, 0.2], zorder=3, bbox=(1.2, 0.8, 0.3, 0.2))
    table_props = mtable.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell.set_width(0.2)

    plt.subplots_adjust(right=0.7)

    plt.gcf().suptitle('')
    plt.gcf().savefig(saveloc)
    plt.gcf().savefig(saveloc.replace('pdf', 'png'))

    # for simplex in hull.simplices:
    #     g.ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')
    # g.ax.fill(points[hull.vertices, 0],
    #           points[hull.vertices, 1], color=next(palette), alpha=0.5)

    # plt.show(block=False)
    return g, dfx, pca


atts = ['length', 'width', 'depth', 'volume',
        'surface_area', 'length_depth_width',
        'mean_length', 'mean_width', 'mean_depth', 'mean_volume',
        'mean_surface_area', 'mean_length_depth_width', 'density', 'Surface Area - Volume Ratio']


# atts = ['length', 'width', 'depth', 'volume',
#        'surface_area', 'length_depth_width']

compare_groups = [('T. monococcum', 'T. beoticum'),
                  ('T. dicoccum', 'T. dicoccoides'),
                  ('H. spontaneum', 'H. vulgare'),
                  ('T. beoticum', 'T. dicoccoides'),
                  ('T. monococcum', 'T. dicoccum')]


def height_to_mm(x): return (x['height'] * 68.8)/1000


def z_to_mm(x): return (x['z'] * 68.8)/1000


def density(x): return x['sum_volume']/x['height']


einkorn = pd.read_excel('../all_data_tidy.xlsx',
                        sheet_name='{0}-{1}'.format(compare_groups[0][0],
                                                    compare_groups[0][1]))
emmer = pd.read_excel('../all_data_tidy.xlsx',
                      sheet_name='{0}-{1}'.format(compare_groups[1][0],
                                                  compare_groups[1][1]))
barley = pd.read_excel('../all_data_tidy.xlsx',
                       sheet_name='{0}-{1}'.format(compare_groups[2][0],
                                                   compare_groups[2][1]))

aestivum = pd.read_excel('../all_data_tidy.xlsx',
                         sheet_name='T. aestivum')


einkorn['slices'] = einkorn['height']
emmer['slices'] = emmer['height']
barley['slices'] = barley['height']


def sa_ratio(x): return x['surface_area']/x['volume']


einkorn['Surface Area - Volume Ratio'] = einkorn.apply(sa_ratio, axis=1)
emmer['Surface Area - Volume Ratio'] = emmer.apply(sa_ratio, axis=1)
barley['Surface Area - Volume Ratio'] = barley.apply(sa_ratio, axis=1)

make_top_bottom(einkorn)
make_top_bottom(emmer)
make_top_bottom(barley)

einkorn['density'] = einkorn.apply(density, axis=1)
emmer['density'] = emmer.apply(density, axis=1)
barley['density'] = barley.apply(density, axis=1)

einkorn['height'] = einkorn.apply(height_to_mm, axis=1)
emmer['height'] = emmer.apply(height_to_mm, axis=1)
barley['height'] = barley.apply(height_to_mm, axis=1)

einkorn['z_mm'] = einkorn.apply(z_to_mm, axis=1)
emmer['z_mm'] = emmer.apply(z_to_mm, axis=1)
barley['z_mm'] = barley.apply(z_to_mm, axis=1)


plt.close('all')

# make_all_pca(einkorn, emmer, barley)
analyse_all(einkorn, emmer, barley, compare_groups, atts, aestivum=aestivum)
