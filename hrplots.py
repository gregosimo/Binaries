"""Script which takes care of setting up plots in various HR-like diagrams."""

import matplotlib.pyplot as plt

def logg_teff_plot(teff, logg, style="k.", **kwargs):
    '''Creates a plot in the Teff-logg space.

    Teff and logg are valuse which should be plotted. Style should be the plot
    style for the points; they are black dots by default. All other values will
    be passed to the underlying errorbar routine.
    '''
    try:
        ax = kwargs.pop("axis")
    except KeyError:
        ax = plt.gca()
    ax.errorbar(teff, logg, fmt=style, **kwargs)
    invert_x_axis(ax)
    invert_y_axis(ax)
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

def standalone_logg_teff_comparison(
    logg1, teff1, logg2, teff2, label1, label2):
    '''Make a 2x2 grid comparing two sets of log(g) vs Teff.'''
    f, axes = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row")

    two_logg_teff_on_axes(f, axes, logg1, teff1, logg2, teff2, label1, label2)

def logg_teff_comparison_subsample(
    fulllogg1, fullteff1, fulllogg2, fullteff2, samplelogg1, sampleteff1, 
    samplelogg2, sampleteff2, label1, label2):
    '''Plot a subsample of objects against a full sample.'''
    f, axes = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row")
    
    two_logg_teff_on_axes(
        f, axes, fulllogg1, fullteff1, fulllogg2, fullteff2, label1, label2, 
        sgkw={"marker": ".", "c": "cyan"}, dwkw={"marker": ".", "c": "pink"})
    two_logg_teff_on_axes(
        f, axes, samplelogg1, sampleteff1, samplelogg2, sampleteff2, label1,
        label2, sgkw={"marker": "o", "c": "b", "ms": 7}, 
        dwkw={"marker": "o", "c": "r", "ms": 7})
                          

def two_logg_teff_on_axes(
    f, axes, logg1, teff1, logg2, teff2, label1, label2, 
    sgkw={"marker": ".", "c": "b"}, dwkw={"marker": ".", "c": "r"}):
    '''Make logg teff plots on the given 2x2 grid.'''
    ((ax1, ax2), (ax3, ax4)) = axes

    subgiant_indices = logg1 < 4.2
    dwarf_indices = logg1 >= 4.2
    logg_teff_plot(
        teff1[subgiant_indices], logg1[subgiant_indices], axis=ax1, **sgkw)
    logg_teff_plot(teff1[dwarf_indices], logg1[dwarf_indices], axis=ax1, **dwkw)
    ax1.set_xlabel("")
    ax1.set_ylabel("{0} log(g)".format(label1))
    logg_teff_plot(teff2[subgiant_indices], logg1[subgiant_indices], axis=ax2,
                   **sgkw)
    logg_teff_plot(teff2[dwarf_indices], logg1[dwarf_indices], axis=ax2, **dwkw)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    logg_teff_plot(teff1[subgiant_indices], logg2[subgiant_indices], axis=ax3,
                   **sgkw)
    logg_teff_plot(teff1[dwarf_indices], logg2[dwarf_indices], axis=ax3, **dwkw)
    ax3.set_xlabel("{0} Teff".format(label1))
    ax3.set_ylabel("{0} log(g)".format(label2))
    logg_teff_plot(teff2[subgiant_indices], logg2[subgiant_indices], axis=ax4,
                   **sgkw)
    logg_teff_plot(teff2[dwarf_indices], logg2[dwarf_indices], axis=ax4, **dwkw)
    ax4.set_xlabel("{0} Teff".format(label2))
    ax4.set_ylabel("")
    ax1.set_xlim(5500, 4000)
    ax1.set_ylim(5.0, 2.0)
    ax4.set_xlim(5600, 4200)
    ax4.set_ylim(4.8, 3.5)
