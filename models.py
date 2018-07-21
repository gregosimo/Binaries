import bisect

from scipy.interpolate import lagrange, interp1d
from astropy.table import Table
import numpy as np

class StellarEvolutionaryTrack(object):
    '''A generic class for a stellar evolutionary track.

    This class will hold the data and metadata for a Stellar Evolutionary
    Track. Important tasks for this object include loading from a file
    automatically as well as interpolating values at a given age.'''
    def __init__(self, datatable, age_col, mass, feh, alpha):
        '''Create an evolutionary track object.

        The object contains a table, which should hold the stellar parameters
        as a function of age. The age column should be specified as age_col.
        The metallicity and alpha abundance should also be specified as feh and
        alpha.'''

        self.tracktable = datatable
        self.age_col = age_col
        self.feh = feh
        self.alpha = alpha

    def interpolate_at_age(self, age, interp_style="average"):
        '''Interpolate the parameters of the star at a given age (given in Gyr).'''
        if interp_style is "lagrange":
            poly_size=4
            ind = bisect.bisect_right(self.tracktable[self.age_col], age)
            cols = [
                col for col in self.tracktable.colnames if col not in self.age_col]
            rowdict = {self.age_col: age}
            for col in cols:
                tableslice = slice(ind-poly_size//2, ind+poly_size//2)
                polyinterp = lagrange(
                    self.tracktable[self.age_col][tableslice], 
                    self.tracktable[col][tableslice])
                rowdict[col] = polyinterp(age)
            newtab = Table(rows=[rowdict])
            return newtab
        elif interp_style is "average":
            ind = bisect.bisect_left(self.tracktable[self.age_col], age)
            tableslice = slice(ind-1, ind+1)
            ageslice = self.tracktable[self.age_col][tableslice]
            cols = [
                col for col in self.tracktable.colnames if col not in self.age_col]
            rowdict = {self.age_col: age}
            for col in cols:
                colslice = self.tracktable[col][tableslice]
                rowdict[col] = (colslice[0] + (colslice[1] - colslice[0]) /
                               (ageslice[1] - ageslice[0]) * (
                                   age - ageslice[0]))
            newtab = Table(rows=[rowdict])
            return newtab

class StellarIsochrone(object):
    '''A generic class for Stellar Isochrones.
    
    This class holds data and metadata for a Stellar Isochrone. Important tasts
    for this object include loading from a file automatically and interpolating
    between columns.
    
    NOTE That a StellarIsochrone object holds a SET of isochrones, not a single
    one.'''
    def __init__(
        self, feh, alpha, mixing_length, Y, Z, vvcrit, bandstr, iso_dict):
        '''Take in attributes needed to define a DSEP isochrone.'''
        self.mixing_length = mixing_length
        self.Y = Y
        self.Z = Z
        self.feh = feh
        self.alpha = alpha
        self.vvcrit = vvcrit
        self.bandstr = bandstr
        self.iso_dict = iso_dict

    def interpolate_isochrone_cols(
            self, age, invals, incol, outcol, interp_kind="linear", 
                mask_outside_bounds=True):
        '''Interpolate between the columns of an isochrone object.
        
        Interpolate the values invals between the columns of the
        StellarInterpolator: incol and outcol. An age should be specified for
        the interpolation.
        
        The kind of interpolation to be done should be given as interp_kind, which
        by default is linear because of the high density of points.'''
        met_table = self.iso_table(age)
        restricted_table = interpolation_table_increasing_stretch(
            met_table, mono_col=incol)
        xvals, yvals = restricted_table[incol], restricted_table[outcol]
        in_ordered, out_ordered = ensure_array_increasing(xvals, yvals)
        in_fixed, out_fixed = fix_duplicate_array_values(
            in_ordered, out_ordered)

        nonmasked_in = np.ma.compressed(invals)
        assert len(nonmasked_in) == len(invals)

        if mask_outside_bounds:
            kinterp = interp1d(
                in_fixed, out_fixed, kind=interp_kind, fill_value=np.nan, 
                bounds_error=False)
            interp_out = np.ma.masked_invalid(kinterp(nonmasked_in))
        else:
            kinterp = interp1d(
                in_fixed, out_fixed, kind=interp_kind, bounds_error=True)
            interp_out = kinterp(nonmasked_in)

        return interp_out

    

def bin_nearby_table_values(table, bin_col, decimals):
    '''Bin the table according to values nearby in bin_col.
    
    Sometimes the outputs of stellar models show odd clustering behavior, where
    points are outputted with independent coordinates very close to each other.
    Those points sometimes capture behavior which isn't physical and may
    actually be numerical. As a way of downsampling that behavior, this
    function will take multiple points that are within the given relative
    tolerance and treat them as equivalent.'''
    # Because np.around only deals with decimal places, I want to make sure all
    # the values are between 1-10.
    exponents = 10**np.floor(np.log10(table[bin_col]))
    normalized_col = table[bin_col] / exponents
    rounded_col = np.around(normalized_col, decimals=decimals) * exponents
    tablegroup = table.group_by(rounded_col)
    newtable = tablegroup.groups.aggregate(np.mean)
    return newtable

def fix_duplicate_array_values(xvals, yvals):
    '''Remove duplicate x-values from arrays.

    One of the problems with DSEP isochrones is that occasionally, there will
    be two adjacent points that have the same x-value, but have different
    y-values. This function will attempt to find those duplicate points and fix
    them.'''
    # Note: This algorithm assumes that there are only two simultaneous
    # duplications. Doing it for n simultaneous duplications might be tricky.
    # The DSEP interpolation causes there to be very slight numerical errors in
    # the answers. As a result, quantities that should be identical can be
    # scattered above or below what they are.
    dupmask = np.abs(np.diff(xvals)) > 1.01e-5
    valarray = np.vstack([xvals, yvals])
    # If the cases of duplication are isolated:
    if np.all(np.logical_or(dupmask[1:], dupmask[:-1])):
        meanvals = np.mean([
            valarray[:, np.hstack([np.ones(1, dtype=bool), dupmask])], 
            valarray[:, np.hstack([dupmask, np.ones(1, dtype=bool)])]], axis=0)
    else:
        splitlist = np.hsplit(valarray, np.where(dupmask)[0]+1)
        meanvals = np.concatenate([
            np.mean(vals, axis=1)[:,np.newaxis] for vals in splitlist], axis=1)
    newx = meanvals[0,:]
    newy = meanvals[1,:]
    return newx, newy

def interpolation_table_increasing_stretch(isochrone, mono_col="LogTeff"):
    '''Cut off the low-mass portion of the table which is increasing.
    
    This function is an alternative to restrict_interpolation_table because it
    ensures that a well-behaved part of the isochrone is used for
    interpolation.'''
    col = isochrone[mono_col]
    # If the column is increasing, then these must be positive.
    coldiff = np.diff(col)
    # Get the first positive index.
    first_index = np.where(coldiff >= 0)[0][0]
    # Find where the index next dips below zero.
    try:
        last_index = first_index+np.where(coldiff[first_index:] < 0)[0][0]
    # There might be a more elegant way of doing this.
    except IndexError:
        last_index = len(coldiff)
    else:
        # Check for one-off blips and reinterpolate them.
        while coldiff[last_index] + coldiff[last_index+1] > 0:
            isochrone.remove_row(last_index)
            coldiff = np.diff(isochrone[mono_col])
            last_index = first_index+np.where(coldiff[first_index:] < 0)[0][0]

    return isochrone[first_index:last_index]

def ensure_array_increasing(xvals, yvals):
    '''Ensure the provided xvalues are increasing.

    Because creating a spline requires that xvalues are strictly increasing,
    this function assumes that the provided xvals array is either strictly
    increasing or decreasing, and if decreasing, it reverses it to ensure that
    it's increasing. The yvals are also reversed if that's the case.
    '''
    if xvals[0] > xvals[-1]:
        newxvals = xvals[::-1]
        newyvals = yvals[::-1]
    else:
        newxvals = xvals
        newyvals = yvals

    # These may require a bit of resorting due to interpolation errors. One
    # quality flag I'd like to ensure is that no drastic sorting changes occur.
    sorted_xvals_indices = np.argsort(newxvals)
    sorted_xvals = newxvals[sorted_xvals_indices]
    sorted_yvals = newyvals[sorted_xvals_indices]
    xdiffs = np.diff(newxvals)
    assert abs(min(xdiffs)) <= 1*min(xdiffs[xdiffs > 0])

    return sorted_xvals, sorted_yvals
