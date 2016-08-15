import os

import numpy as np
import numpy.core.defchararray as npstr
from astropy.table import Table

def read_Casagrande_10_Table_4(
    tblpath="./Casagrande_10_Table_4.txt"):
    '''Read Table 4 in Casagrande et al (2010).

    This table contains the coefficients for fitting effective temperatures to
    colors (and metallicities). These will also be useful for inverting the
    relationship to get colors as a function of effective temperature.'''

    cas = Table.read(
        tblpath, format="ascii.fixed_width_no_header", data_start=1,
        col_starts=(0, 25, 30, 49, 54, 64, 72, 80, 96, 112, 128, 144, 152),
        col_ends=(23, 28, 32, 52, 57, 71, 79, 95, 111, 127, 143, 151, 173),
        names=["Color", "Met_start", "Met_end", "Color_start", "Color_end", 
               "a0", "a1", "a2", "a3", "a4", "a5", "N", "unc"] )
    cas["Color"] = remove_latex_subscript_formatting(cas["Color"])
    return cas

def read_Casagrande_10_Table_5(
    tblpath="/home/regulus/simonian/Binaries/Casagrande_10_Table_5.txt"):
    '''Read Table 5 in Casagrande et al (2010).

    This table contains the coefficients for fitting bolometric fluxes from the
    Earth.
    '''

    cas = Table.read(
        tblpath, format="ascii.fixed_width_no_header", data_start=1,
        col_starts=(0, 16, 41, 46, 65, 70, 80, 88, 104, 120, 136, 152, 168, 184, 192),
        col_ends=(15, 39, 44, 48, 68, 74, 87, 103, 119, 135, 151, 167, 183, 191, 203),
        names=["Band", "Color", "Met_start", "Met_end", "Color_start",
               "Color_end", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "N",
               "unc"] )
    cas["Band"] = remove_latex_subscript_formatting(cas["Band"])
    cas["Color"] = remove_latex_subscript_formatting(cas["Color"])
    return cas

def remove_latex_subscript_formatting(col):
    '''Removes subscript formatting from A&A columns.
    
    For now, this will by definition strip dollar signs, and turn "_{\rm X}" 
    into just "X". More fancy rules may be added if they are called for.'''

    stripped_col = npstr.strip(col, "$")
    detexedcol = npstr.replace(npstr.replace(stripped_col, "_{\\rm ", ""), 
                               "}", "")
    return detexedcol

def distribute_color_subscript(col):
    '''Distributes the subscript for a color to the full band.

    A shorthand notation for noting which system a particular color was
    observed in is to surround the bands with parentheses, and then add a
    subscript to indicate that both bands are taken with that articular
    subscript. This function removes the parentheses, and distributes the
    subscript to both fo the bands. For example, (R-I)C becomes RC-IC. This
    will be easier to disentangle.
    '''
    newcol = col.copy()
    shortened_indices = npstr.startswith(newcol, "(")
    shortened_entries = newcol[shortened_indices]

    # From outide to inside:
    # Replace left parenthesis.
    # Add ending suffix to first band
    # Remove ending parenthesis.
    distributed_entries = [
        npstr.replace(npstr.replace(npstr.replace(
            shortened, ")", ""), "-", shortened[-1] + "-"), "(", "") 
        for shortened in shortened_entries]

    newcol[shortened_indices] = distributed_entries
    return newcol

def split_color(colorcol):
    '''Splits an array of string colors into two arrays of bands.

    This takes an array of strings such as "RC-J" or "(B-V)T" and splits them
    into two arrays which just have the bands, such as "RC" & "J" and "BT" &
    "VT". This can handle colors which need to be distributed.
    '''
    distributed_colors = distribute_color_subscript(colorcol)
    bluecolor = np.array(distributed_colors)
    redcolor = np.array(distributed_colors)
    splitcolors = npstr.split(distributed_colors, "-")

    for i,lst in enumerate(splitcolors):
        bluecolor[i] = lst[0]
        redcolor[i] = lst[1]
    
    return bluecolor, redcolor

def Casagrande_Teff(color, colorvals, metallicity, extrapolation_exception=True):
    '''Calculate the Teff of a star using Casagrande (2010) calibration.

    This function calculates the Teff of a star using a given color. The string
    color needs to be given as the color argument. The actual values of color
    should be given as colorvals. And the [Fe/H] values of the stars should be
    given as metallicity. 

    The color should be given in the condensed mode, not in the distributed
    mode.

    If the colors or metallicity given are outside of the range considered by
    Casagrande et al (2010), then if extrapolation_exception is enabled, this
    function will raise a ValueError. Otherwise, it will continue normally.
    '''
    casagrande_teff_table = read_Casagrande_10_Table_4()

    casagrande_row = casagrande_teff_table[
            casagrande_teff_table["Color"] == color]

    out_of_color_bound = np.any(np.logical_or(
           colorvals < casagrande_row["Color_start"], 
           colorvals > casagrande_row["Color_end"]))
    out_of_met_bound = np.any(np.logical_or(
            metallicity < casagrande_row["Met_start"],
            metallicity > casagrande_row["Met_end"]))

    if out_of_color_bound and out_of_met_bound and extrapolation_exception:
        raise ValueError("Both Metallicity and Colors are outside of the "
            "calibration bounds.")
    elif out_of_color_bound and extrapolation_exception:
        raise ValueError("Colors are outside of the calibration bounds.")
    elif out_of_met_bound and extrapolation_exception:
        raise ValueError("Metallicities are outside of the calibration "
                "bounds.")

    theta_eff = (casagrande_row["a0"] + casagrande_row["a1"] * colorvals + 
                  casagrande_row["a2"] * colorvals**2 + 
                  casagrande_row["a3"] * colorvals * metallicity + 
                  casagrande_row["a4"] * metallicity + 
                  casagrande_row["a5"] * metallicity**2)
    teff = 5040 / theta_eff
    return teff

def Casagrande_inverted_color(color, teffs, metallicity, 
        extrapolate_exception=True):
    '''Inverts Casagrande et al (2010) to get color from Teff and [Fe/H].

    This function is used to get empirical, calibrated colors from a
    theoretical Teff and metallicity.

    If the metallicity or Teff is outside of the calibration bounds, this
    function will throw an error, unless the extrapolate_exception flag is
    turned off.
    '''
    casagrande_teff_table = read_Casagrande_10_Table_4()

    casagrande_row = casagrande_teff_table[
            casagrande_teff_table["Color"] == color]

    out_of_teff_bound = np.any(np.logical_or(
           teffs > Casagrande_Teff(
               color, casagrande_row["Color_start"], metallicity), 
           teffs < Casagrande_Teff(
               color, casagrande_row["Color_end"], metallicity)))
    out_of_met_bound = np.any(np.logical_or(
            metallicity < casagrande_row["Met_start"],
            metallicity > casagrande_row["Met_end"]))

    if out_of_teff_bound and out_of_met_bound and extrapolation_exception:
        raise ValueError("Both Metallicity and Colors are outside of the "
            "calibration bounds.")
    elif out_of_teff_bound and extrapolation_exception:
        raise ValueError("Colors are outside of the calibration bounds.")
    elif out_of_met_bound and extrapolation_exception:
        raise ValueError("Metallicities are outside of the calibration "
                "bounds.")

    # In this case, I'm basically inverting Eq. 3 in Casagrande et al (2010) by
    # treating it as a quadratic equation in color. Therefore, these will be
    # the components of the solution:
    a = casagrande_row["a2"]
    b = casagrande_row["a1"] + casagrande_row["a3"] * metallicity
    theta_eff = 5060 / teffs
    c = (casagrande_row["a0"] + casagrande_row["a4"] * metallicity +
         casagrande_row["a5"] * metallicity**2 - theta_eff)

    color = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    return color


if __name__ == "__main__":
    
    CASAGRANDE_TABLE_DIR = "."
    print(Casagrande_Teff("V-H", 1.0, -1.0))
