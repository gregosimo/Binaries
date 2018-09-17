import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table, vstack
import bisect

import path_config as paths
import dsep
import hrplots as hr
import biovis_colors as bc
import models

band_translation = {"V": "Bessell_V", "H": "2MASS_H", "K": "2MASS_Ks", "Ks": "2MASS_Ks"}

class MISTIsochrone(models.StellarIsochrone):
    '''A class that encapsulates a MIST isochrone.'''
    # These are characteristic of all MIST Isochrones
    age_col = "isochrone_age_yr"
    mass_col = "initial_mass"
    logteff_col = "log_Teff"
    logL_col = "log_L"
    # Note that the radius is inferred, not directly interpolated.
    radius_col = "radius"
    def __init__(self, feh, fulltable, alpha=0, Yinit=0.2703, Zinit=1.42857e-2, 
                 vvcrit=0.00, AV=0.0, MIST_version=1.1, MESA_version=7503, 
                 bandstr="UBVRIplus", make_rad_col=True):

        if make_rad_col:
            fulltable[self.radius_col] = 10**(0.5*(
                fulltable[self.logL_col] - 4*(
                    fulltable[self.logteff_col] - np.log10(5777))))
        # This should make a dictionary which has age as a key and that
        # subtable as a value.
        fullgroups = fulltable.group_by(self.age_col)
        iso_dict = {
            np.round(key[0] / 10**(np.floor(np.log10(key[0]))),
                     2)*10**(np.floor(np.log10(key[0]))): val for key, val in zip(
                fullgroups.groups.keys, fullgroups.groups)}
        super().__init__(feh, alpha, 0, Yinit, Zinit, vvcrit, bandstr, iso_dict)
        self.AV=AV
        self.MIST_version=MIST_version
        self.MESA_version=MESA_version

        self.increasing_colnames = set(band_translation.values())
        self.decreasing_colnames = set([self.mass_col, self.logteff_col,
                                        self.logL_col])

    def iso_table(self, age):
        '''Return the table corresponding to the isochrone at the given age.
        The given age should be given in log10(yr).'''
        self.replace_with_tracks(age, transition_mass=0.88)
        fullagetable = self.iso_dict[age]
        # Sometimes there are EEPS which have almost the same mass, but other
        # values dominated by noise. So I am going to bin the whole table so
        # that masses have only 3 decimal places.
        fullagegroups = fullagetable.group_by(np.round(
            fullagetable[self.mass_col], 3))
        meanagetable = fullagegroups.groups.aggregate(np.mean)
        return meanagetable

    @classmethod
    def isochrone_from_file(
        cls, feh, alpha=0.0, vvcrit=0.0, bandstr="UBVRIplus", MIST_version=1.1,
            MIST_PATH=paths.MIST_ISOCHRONES):
        '''Read in a MIST isochrone from a file.'''
        filename = build_MIST_filename(feh)
        MIST_table = Table.read(
            str(MIST_PATH / filename), format="ascii.fast_commented_header",
            header_start=12, guess=False, data_start=0)
        mist = cls(
            feh, MIST_table, alpha=alpha, vvcrit=vvcrit, 
            MIST_version=MIST_version, bandstr=bandstr)
        return mist

    def replace_with_tracks(self, age, transition_mass=1.0):
        '''Replace the lower-mass portion of isochrones with tracks.

        Because the isochrones sometimes have gaps in them for whatever
        reason, this function will take the isochrone at the given age and
        replace the portion with mass less than transition_mass with entries
        calculated directly from evolutionary tracks.'''
        met_table = self.iso_dict[age]
        isochrone_min_mass_index = bisect.bisect_left(
            met_table[self.mass_col], transition_mass)
        isochrone_highmass_indices = slice(
            isochrone_min_mass_index, len(met_table))
        # I could hypothetically read this from the directory. But that's more
        # effort than I'd like to spend.
        masses = np.linspace(0.1, 1.3, 60+1, endpoint=True)
        model_max_mass_index = bisect.bisect_left(masses, transition_mass)
        masslist = []
        for mass in masses[0:model_max_mass_index]:
            mist_track = MISTEvolutionaryTrack.track_from_file(
                mass, self.feh, vvcrit=self.vvcrit, bandstr=self.bandstr,
                MIST_version=self.MIST_version)
            # Age is stored as log10(age) in the isochrone, but is linear in
            # the evolutionary track...
            try:
                if self.age_col.startswith("log"):
                    newiso = mist_track.interpolate_at_age(
                        10**age, interp_style="average")
                else:
                    newiso = mist_track.interpolate_at_age(
                        age, interp_style="average")
            except ValueError:
                assert isochrone_min_mass_index == len(met_table)
                break
            newiso.remove_column(mist_track.age_col)
            newiso["EEP"] = 0
            if self.age_col.startswith("log"):
                newiso[self.age_col] = np.log10(age)
            else:
                newiso[self.age_col] = age
            newiso[self.mass_col] = mass
            newiso["star_mass"] = mass
            newiso["[Fe/H]_init"] = mist_track.feh
            newiso["[Fe/H]"] = mist_track.feh
            del(newiso["Z_surf"])
            masslist.append(newiso)
        newtable = vstack(masslist)
        if self.radius_col in met_table.colnames:
            newtable[self.radius_col] = 10**(0.5*(
                newtable[self.logL_col] - 4*(
                    newtable[self.logteff_col] - np.log10(5777))))
        combined_table = vstack([
            newtable, met_table[isochrone_highmass_indices]])
        self.iso_dict[age] = combined_table

class MISTEvolutionaryTrack(models.StellarEvolutionaryTrack):
    '''A model of the MIST Evolutionary Tracks.'''
    @classmethod
    def track_from_file(
            cls, mass, feh, vvcrit=0.0, bandstr="UBVRIplus",
            MIST_version=1.1, MIST_PATH=paths.MIST_PATH):
        '''Read in the MIST tracks.'''
        mist_folder = name_track_folder(MIST_version, vvcrit, bandstr, feh)
        mist_file = name_track_file(mass)
        MIST_table = Table.read(
            str(MIST_PATH / mist_folder / mist_file),
            format="ascii.commented_header", header_start=14, data_start=0)
        # This is an intensive operation that may not be necessary given the
        # crude interpolation I'm doing.
#       cleaned_table = models.bin_nearby_table_values(
#           MIST_table, "star_age", 2)
        cleaned_table = MIST_table
        mist = cls(cleaned_table, "star_age", mass, feh, 0.0)
        mist.MIST_version = MIST_version
        return mist

def name_track_folder(MIST_version, vvcrit, bandstr, feh):
    '''Name the folder that contains evolutionary tracks.'''
    templatestr = "MIST_v{0:.1f}_vvcrit{1:.1f}_{2}_feh_{3}{4:4.2f}_tracks"
    newstr = templatestr.format(
        MIST_version, vvcrit, bandstr, dsep.assign_DSEP_sign(feh), abs(feh))
    return newstr

def name_track_file(mass):
    '''Name a MIST track file. This specifies a mass within the track folder.'''
    massformat = int(np.round(mass, 4) * 10000)
    templatestr = "{0:07d}M.track.eep.cmd"
    newstr = templatestr.format(massformat)
    return newstr

def interpolate_MIST_isochrone_cols(
        iso, age, interp_in, incol="log_Teff", outcol="2MASS_Ks",
        interp_kind="linear", mask_outside_bounds=True):
    '''Interpolate between the columns of an isochrone object.
    
    The full MISTIsochrone object should be passed as an argument, along with
    an age, followed by the input
    values which should be interpolated. The columns to interpolate between
    should be given as incol and outcol.
    
    The kind of interpolation to be done should be given as interp_kind, which
    by default is linear because of the high density of points.'''
    met_table = iso.iso_table(age)
    restricted_table = dsep.interpolation_table_increasing_stretch(
        met_table, mono_col=incol)
    invals, outvals = restricted_table[incol], restricted_table[outcol]
    in_ordered, out_ordered = dsep.ensure_array_increasing(invals, outvals)
    in_fixed, out_fixed = dsep.fix_duplicate_array_values(
        in_ordered, out_ordered)

    nonmasked_in = np.ma.compressed(interp_in)
    assert len(nonmasked_in) == len(interp_in)

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

def read_MIST_isochrone(feh, MIST_PATH=paths.MIST_ISOCHRONES):
    '''Read in a MIST isochrone into a MISTIsochrone object.'''
    filename = build_MIST_filename(feh)

    MIST_table = Table.read(
        str(MIST_PATH / filename), format="ascii.commented_header",
        header_start=12,
        guess=False, data_start=0, comment="\s*#")
    return MIST_table

def build_MIST_filename(feh, alpha=0.0, vvcrit=0.0, bandstr="UBVRIplus",
                        MISTversion=1.1):
    '''Create a string that specifies the mist isochrone with the given params.'''

    mist_template =("MIST_v{0:3.1f}_feh_{1}{2:4.2f}_afe_{3}{4:3.1f}_"
                    "vvcrit{5:3.1f}_{6}.iso.cmd")
    filestr = mist_template.format(
        MISTversion, dsep.assign_DSEP_sign(feh), abs(feh), 
        dsep.assign_DSEP_sign(alpha), abs(alpha), vvcrit, bandstr)

    return filestr

def test_teff_k_interpolation(
        age, feh, outcol, alpha=0, vvcrit=0.0, bandstr="UBVRIplus"):
    '''Test temperature interpolation by removing and predicting single points.

    Performs a rough test of interpolation by removing single points and
    predicting what the value of those points ought to be.'''
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw = {"height_ratios":[2, 1]},
                               sharex=True)
    iso = MISTIsochrone.isochrone_from_file(feh, alpha=alpha)
    met_table = iso.iso_table(age)
    restricted_table = models.interpolation_table_increasing_stretch(
        met_table, mono_col=iso.logteff_col)
    teffvals, colvals = restricted_table[iso.logteff_col], restricted_table[outcol]
    teff_ordered, col_ordered = models.ensure_array_increasing(
        teffvals, colvals)
    teff_fixed, col_fixed = models.fix_duplicate_array_values(
        teff_ordered, col_ordered)

    # These will hold the |predicted-actual| values for each point, except the
    # first and last.
    linear_offtable = np.zeros(len(teff_fixed)-2)
    cubic_offtable = np.zeros(len(teff_fixed)-2)
    
    # Now make interpolators with missing pieces.
    for i, (logT, outval) in enumerate(zip(teff_fixed[1:-1], col_fixed[1:-1])):
        mask = np.ones(len(teff_fixed), dtype="bool")
        mask[i+1] = 0
        assert np.count_nonzero(mask) == len(mask)-1
        assert logT == teff_fixed[~mask][0]
        masked_teff = teff_fixed[mask]
        masked_out = col_fixed[mask]

        kinterp_linear = interp1d(masked_teff, masked_out, kind="linear")
        kinterp_cubic = interp1d(masked_teff, masked_out, kind="cubic")

        linear_offtable[i] = np.abs(kinterp_linear(logT) - outval)
        cubic_offtable[i] = np.abs(kinterp_cubic(logT) - outval)

    # Show the interpolation on a finer grid.
    test_teffs = np.linspace(teff_fixed[0], teff_fixed[-1], 1000)
    kinterp_linear = interp1d(teff_fixed, col_fixed, kind="linear")
    kinterp_cubic = interp1d(teff_fixed, col_fixed, kind="cubic")
    test_linear = kinterp_linear(test_teffs)
    test_cubic = kinterp_cubic(test_teffs)

    # Plot the interpolation
    hr.absmag_teff_plot(10**test_teffs, test_linear, color=bc.red, ls="-",
                        marker="", label="Linear", axis=a0)
    hr.absmag_teff_plot(10**test_teffs, test_cubic, color=bc.black, ls="-",
                        marker="", label="Cubic", axis=a0)
    hr.absmag_teff_plot(
        10**teff_fixed, col_fixed, color=bc.blue, ls="", marker="o", axis=a0)
    a0.set_xlabel("")
    a0.set_ylabel("Ks")
    a0.legend(loc="upper right")
    a0.set_title("Age: {0:.2f} Gyr; [Fe/H]: {1:.2f}".format(age, feh))

    # Plot the difference
    a1.semilogy(10**teff_fixed[1:-1], linear_offtable, color=bc.red, ls="-",
                marker="")
    a1.semilogy(10**teff_fixed[1:-1], cubic_offtable, color=bc.black, ls="-",
                marker="")
    a1.set_xlabel("Teff (K)")
    a1.set_ylabel("Error (mag)")

def test_feh_interpolation(
        age, teffs, outcol, alpha=0.0, vvcrit=0.0, bandstr="UBVRIplus"):
    '''Plot the interpolation over [Fe/H].'''
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw = {"height_ratios":[2, 1]},
                               sharex=True)
    # Now iterate through the [Fe/H] values.
    input_fehs = np.array([
#        -4.0, -3.5, -3.0, 
        -2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5,
        -0.25, 0.0, 0.25, 0.5])
    interp_fehs = np.zeros(len(input_fehs))
    k_array = np.ma.zeros((len(input_fehs), len(teffs)))
    for i, ifeh in enumerate(input_fehs):
        iso = MISTIsochrone.isochrone_from_file(ifeh, alpha=alpha)
        interp_fehs[i] = iso.feh
        k_array[i,:] = iso.interpolate_isochrone_cols(
            age, np.log10(teffs), iso.logteff_col, outcol,
            interp_kind="linear")

    # These will hold the |predicted-actual| values for each point, except the
    # first and last.
    linear_offtable = np.ma.zeros((len(interp_fehs)-2, len(teffs)))
    cubic_offtable = np.ma.zeros((len(interp_fehs)-2, len(teffs)))
    # Now step through [Fe/H].
    for j, (intfeh, outval) in enumerate(
            zip(interp_fehs[1:-1], k_array[1:-1,0])):
        mask = np.ones(len(interp_fehs), dtype="bool")
        mask[j+1] = 0
        fullmask = np.logical_and(mask[:,np.newaxis], np.logical_not(k_array.mask))
        for i in range(len(teffs)):
            masked_feh = interp_fehs[fullmask[:,i]]
            masked_out = k_array[:,i][fullmask[:,i]]

            kinterp_linear = interp1d(
                masked_feh, masked_out, kind="linear", bounds_error=False,
            fill_value=np.nan)
            kinterp_cubic = interp1d(
                masked_feh, masked_out, kind="cubic", bounds_error=False,
                fill_value=np.nan)

            linear_offtable[j,i] = np.abs(np.ma.masked_invalid(
                kinterp_linear(intfeh)) - outval)
            cubic_offtable[j,i] = np.abs(np.ma.masked_invalid(
                kinterp_cubic(intfeh)) - outval)

    teffindex = 0
    # Show the interpolation on a finer grid.
    test_fehs = np.linspace(interp_fehs[0], interp_fehs[-1], 1000)
    masked_feh = interp_fehs[~np.ma.getmask(k_array)[:,teffindex]]
    masked_vals = np.ma.compressed(k_array[~np.ma.getmask(k_array)[:,teffindex]])
    kinterp_linear = interp1d(masked_feh, masked_vals, kind="linear",
                              bounds_error=False, fill_value=np.nan)
    kinterp_cubic = interp1d(masked_feh, masked_vals, kind="cubic",
                             bounds_error=False, fill_value=np.nan)
    test_linear = kinterp_linear(test_fehs)
    test_cubic = kinterp_cubic(test_fehs)

    # Plot the interpolation
    a0.plot(
        test_fehs, test_linear, color=bc.red, ls="-", marker="", label="Linear")
    a0.plot(
        test_fehs, test_cubic, color=bc.black, ls="-", marker="", label="Cubic")
    a0.plot(
        interp_fehs, k_array[:,teffindex], color=bc.blue, ls="", marker="o")
    hr.invert_y_axis(a0)
    a0.set_xlabel("")
    a0.set_ylabel("Ks")
    a0.legend(loc="upper left")
    a0.set_title("MIST Age: {0:.2f} Gyr; Teff: {1:4d} K".format(
        age, teffs[teffindex]))

    # Plot the difference
    a1.semilogy(interp_fehs[1:-1], linear_offtable[:,teffindex], color=bc.red, ls="-",
                marker="")
    a1.semilogy(interp_fehs[1:-1], cubic_offtable[:,teffindex], color=bc.black, ls="-",
                marker="")
    a1.set_xlabel("[Fe/H]")
    a1.set_ylabel("Error (mag)")

def plot_age_differences(ages):
    '''Make a plot showing the differences in raw model ages for MIST
    isochrones.'''
    iso = MISTIsochrone.isochrone_from_file(0.0)
    iso.replace_with_tracks(1.0)
    iso.replace_with_tracks(4.46)
    iso.replace_with_tracks(9.0)
    youngtable = iso.iso_table(1.0)
    medtable = iso.iso_table(4.46)
    oldtable = iso.iso_table(9.0)

    youngtable = youngtable[youngtable["initial_mass"] < 1.3]
    medtable = medtable[medtable["initial_mass"] < 1.3]
    oldtable = oldtable[oldtable["initial_mass"] < 1.3]

    hr.absmag_teff_plot(10**youngtable["log_Teff"], youngtable["2MASS_Ks"],
                        color=bc.blue, marker="o", ls="-")
    hr.absmag_teff_plot(10**medtable["log_Teff"], medtable["2MASS_Ks"],
                        color=bc.orange, marker="o", ls="-")
    hr.absmag_teff_plot(10**oldtable["log_Teff"], oldtable["2MASS_Ks"],
                        color=bc.red, marker="o", ls="-")
