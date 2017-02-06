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

def three_panel_hrdiagram(teff, luminosity, BVcolor, Vmag, JKcolor, Kmag):
    '''Makes a three-panel HR diagram.

    The first panel should be a Teff-Luminosity diagram. This is purely
    theoretical and should reflect what the isochrones as a whole are doing.

    The second panel will be a B-V vs M_V diagram. This will describe how the
    stars behave in the optical.

    The last panel will be a J-Ks vs M_Ks diagram. This will describe how the
    stars behave in the NIR.'''

    plt.subplot(131)
    plt.plot(teff, luminosity, line_style="-")
    # TBD
