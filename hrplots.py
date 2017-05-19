"""Script which takes care of setting up plots in various HR-like diagrams."""

import matplotlib.pyplot as plt

def logg_teff_plot(teff, logg, style="k.", **kwargs):
    '''Creates a plot in the Teff-logg space.

    Teff and logg are valuse which should be plotted. Style should be the plot
    style for the points; they are black dots by default. All other values will
    be passed to the underlying errorbar routine.
    '''
    plt.errorbar(teff, logg, fmt=style, **kwargs)
    invert_x_axis()
    invert_y_axis()
    plt.xlabel("Teff")
    plt.ylabel("log g")

def invert_x_axis(axes=None):
    '''Inverts the x-axis for given axes (useful for Teff)'''
    if axes is None:
        axes = plt.gca()
    xlims = axes.get_xlim()
    if xlims[0] < xlims[1]:
        # Invert the axis
        axes.set_xlim(xlims[::-1])
        
def invert_y_axis(axes=None):
    '''Inverts the y-axis for given axes (useful for log g)'''
    if axes is None:
        axes = plt.gca()
    ylims = axes.get_ylim()
    if ylims[0] < ylims[1]:
        # Invert the axis
        axes.set_ylim(ylims[::-1])

def compare_two_logg_teff(logg1, teff1, logg2, teff2, label1, label2):
    '''Make a 2x2 grid comparing two sets of log(g) vs Teff.'''
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex="col", sharey="row")

    subgiant_indices = logg1 > 4.2
    dwarf_indices = logg1 <= 4.2
    plt.sca(ax1)
    logg_teff_plot(teff1[subgiant_indices], logg1[subgiant_indices], style='r.')
    logg_teff_plot(teff1[dwarf_indices], logg1[dwarf_indices], style='b.')
    plt.xlabel("")
    plt.ylabel("{0} log(g)".format(label1))
    plt.sca(ax2)
    logg_teff_plot(teff2[subgiant_indices], logg1[subgiant_indices], style='r.')
    logg_teff_plot(teff2[dwarf_indices], logg1[dwarf_indices], style='b.')
    plt.xlabel("")
    plt.ylabel("")
    plt.sca(ax3)
    logg_teff_plot(teff1[subgiant_indices], logg2[subgiant_indices], style='r.')
    logg_teff_plot(teff1[dwarf_indices], logg2[dwarf_indices], style='b.')
    plt.xlabel("{0} Teff".format(label1))
    plt.ylabel("{0} log(g)".format(label2))
    plt.sca(ax4)
    logg_teff_plot(teff2[subgiant_indices], logg2[subgiant_indices], style='r.')
    logg_teff_plot(teff2[dwarf_indices], logg2[dwarf_indices], style='b.')
    plt.xlabel("{0} Teff".format(label2))
    plt.ylabel("")
    ax1.set_xlim(5500, 4000)
    ax1.set_ylim(5.0, 2.0)
    ax4.set_xlim(5600, 4200)
    ax4.set_ylim(4.8, 3.5)
    
