from astropy.table import Table
import numpy as np
import astropy_util as au

import sample_characterization as samp
import mist
import path_config as paths

mets_of_interest = [-0.4, 0.0, 0.4]
vels_of_interest = [10]

# Get processed jen files.
@au.memoized
def jen_fast_boundary():
    '''Get the boundary corresponding to the fast launch condition.

    This table has already converted the coordinates to have Teff and M_K.'''
    jen_fast = read_Jen_SGB_fast_launch()
    make_additional_columns(0.0, 10, jen_fast)
    return jen_fast

@au.memoized
def jen_slow_boundary():
    '''Get the boundary corresponding to the slow launch condition.

    This table has already converted the coordinates to have Teff and M_K.'''
    jen_slow = read_Jen_SGB_slow_launch()
    make_additional_columns(0.0, 10, jen_slow)
    return jen_slow

# Read in files

def read_Jen_SGB_slow_launch(filepath=paths.JEN_SLOW_LAUNCH_SGB_PATH):
    '''Jen's boundary for the slow launch condition.'''
    return read_Jen_SGB_boundary(filepath)

def read_Jen_SGB_fast_launch(filepath=paths.JEN_FAST_LAUNCH_SGB_PATH):
    '''Jen's boundary for the fast launch condition.'''
    return read_Jen_SGB_boundary(filepath)

def read_Jen_SGB_boundary(filepath):
    '''Read in a file that has Jen's boundaries for the subgiant branch.
    
    This function just needs the path to the boundary file and will read it in.'''
    tab = Table.read(
        filepath, format="ascii.csv", fill_values=("0.000000000000000", "0"))
    return tab

# Get the columns 

def format_Jen_column(met, vel, param):
    '''Label the column for Jen's lookup table.

    Jen's table has columns in the form of [met],[v]km/s,[param] to specify the
    boundary at a given metallicity, velocity threshold, and Teff/Radius
    column.'''
    template = "{0:.1f},{1:d}km/s,{2}"
    return template.format(met, vel, param)

# Make additional columns

def make_additional_columns(
        met, vel, boundarytable, radcol="Rad", teffcol="Teff", lumcol="Lum", 
        bccol="BC_K", MKcol="M_K"):
    '''Make all additional columns for a given metallicity and velocity.
    
    Labels for the column names should be passed as keyword arguments. They all
    have reasonable default values.'''
    make_luminosity_col(
        met, vel, boundarytable,radcol=radcol, teffcol=teffcol, lumcol=lumcol)
    make_bc_k_col(
        met, vel, boundarytable, teffcol=teffcol, bccol=bccol)
    make_MK_col(
        met, vel, boundarytable, lumcol=lumcol, bccol=bccol, MKcol=MKcol)

def make_luminosity_col(
        met, vel, boundarytable, radcol="Rad", teffcol="Teff", lumcol="Lbol"):
    '''Generate a column of luminosity from the SB-Law.

    This generates a luminosity column for the given metallicity and velocity
    threshold.'''
    radius = boundarytable[format_Jen_column(met, vel, radcol)]
    teff = boundarytable[format_Jen_column(met, vel, teffcol)]
    lum = radius**2 * (teff / 5778)**4
    boundarytable[format_Jen_column(met, vel, lumcol)] = lum

def make_bc_k_col(
        met, vel, boundarytable, teffcol="Teff", bccol="BC_K"):
    '''Generate a bolometric correction for the given star.

    Uses the MIST stellar atmospheres to generate a bolometric correction for
    the given star.'''
    teff = boundarytable[format_Jen_column(met, vel, teffcol)]
    # I need to do a little bit of manipulation since MIST doesn't handle
    # masked values well.
    good_indices = ~teff.mask
    boundarytable[format_Jen_column(met, vel, bccol)] = np.ma.zeros(len(teff))
    logteff = np.log10(teff[good_indices])
    mets = np.ones(len(logteff))*met
    # Maybe there should be a conversion from [Z/H] to [Fe/H], but it's not a
    # big difference.
    bcs = np.diag(samp.calc_model_over_feh_fixed_age_alpha(
        logteff, mist.MISTIsochrone.logteff_col, "BC K", mets, 1e9))
    boundarytable[format_Jen_column(met, vel, bccol)][good_indices] = bcs
    boundarytable[format_Jen_column(met, vel, bccol)].mask = ~good_indices
    
def make_MK_col(
        met, vel, boundarytable, lumcol="Lbol", bccol="BC K", MKcol="M_K"):
    '''Generate a column of K-band absolute magnitudes
    
    Apply the bolometric correction to the luminosity to generate a K-band
    absolute magnitude.'''
    lum = boundarytable[format_Jen_column(met, vel, lumcol)]
    BCK = boundarytable[format_Jen_column(met, vel, bccol)]
    Mbol = 4.74 - 2.5 * np.log10(lum)
    MK = Mbol - BCK
    boundarytable[format_Jen_column(met, vel, MKcol)] = MK
