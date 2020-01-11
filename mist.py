import bisect
import urllib
import posixpath
import io
import zipfile
import shutil
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table, vstack
import requests
from bs4 import BeautifulSoup

import path_config as paths
import dsep
import hrplots as hr
import biovis_colors as bc
import models
import sed

band_translation = {
    "B": "Bessell_B", "V": "Bessell_V", "R": "Bessell_R", 
    "J": "2MASS_J", "H": "2MASS_H", "K": "2MASS_Ks", "Ks": "2MASS_Ks", 
    "g": "SDSS_g", "r": "SDSS_r", "i": "SDSS_i", 
    "G": "Gaia_G_DR2Rev", "BP": "Gaia_BP_DR2Rev", "RP": "Gaia_RP_DR2Rev"}

LATEST_MIST_VERSION = 1.2

class MISTIsochrone(models.StellarIsochrone):
    '''A class that encapsulates a MIST isochrone.'''
    # These are characteristic of all MIST Isochrones
    # I can move these to the StellarIsochrone class using @property and
    # NotImplementedError.
    age_col = "isochrone_age_yr"
    mass_col = "initial_mass"
    logteff_col = "log_Teff"
    logL_col = "log_L"
    # Note that the radius is inferred, not directly interpolated.
    radius_col = "radius"
    def __init__(
            self, feh, fulltable, alpha=0, Yinit=0.2703, Zinit=1.42857e-2, 
            vvcrit=0.00, AV=0.0, MIST_version=LATEST_MIST_VERSION, 
            MESA_version=7503, bandstr="UBVRIplus", make_rad_col=True, 
            bol_bands=[], colors=[]):

        if make_rad_col:
            self._add_radius_column(fulltable, logg=False)
        for bol in bol_bands:
            colname = "BC {0}".format(bol)
            self._add_bolometric_correction(
                fulltable, band_translation[bol], colname)
        for color in colors:
            self._add_color_column(
                fulltable, color, colorcol=color)
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
        self.phot_bands = list(set(band_translation.values()))

        self.increasing_colnames = (
            set(band_translation.values()) | set(colors))
        self.decreasing_colnames = set([self.mass_col, self.logteff_col,
                                        self.logL_col])

    def iso_table(self, age):
        '''Return the table corresponding to the isochrone at the given age.
        The given age should be given in years.'''
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
            cls, feh, alpha=0.0, vvcrit=0.0, bandstr="UBVRIplus", 
            bol_bands=["V", "K"], colors=["BP-RP"], 
            MIST_version=LATEST_MIST_VERSION, MIST_PATH=paths.MIST_PATH,
            abridged=True):
        '''Read in a MIST isochrone from a file.'''
        if abridged:
            iso_folder = abridged_isochrone_folder(
                MIST_version, vvcrit, bandstr)
        else:
            iso_folder = full_isochrone_folder(
                MIST_version, vvcrit, bandstr)
        filename = build_MIST_filename(
            feh, vvcrit=vvcrit, alpha=alpha, bandstr=bandstr,
            MIST_version=MIST_version)
        MIST_table = Table.read(
            str(MIST_PATH / iso_folder / filename), 
            format="ascii.fast_commented_header", header_start=12, guess=False, 
            data_start=0)
        mist = cls(
            feh, MIST_table, alpha=alpha, vvcrit=vvcrit, 
            MIST_version=MIST_version, bandstr=bandstr, bol_bands=bol_bands,
            colors=colors)
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
        # It may be possible to just get this from the directory.
        masses = np.linspace(0.1, 1.3, 60+1, endpoint=True)
        # A bug in version 1.2 of the MIST isochrones is that some grid points
        # are truncated at 1.8e8. Aaron says this should be fixed in MIST 2.0.
        # But for now, I'm going to just try working around it.
        if self.MIST_version == 1.2:
            if self.feh == 0.0:
                masses = np.delete(masses, np.where(np.isclose(masses, 0.36)), 0)
            elif self.feh == 0.25:
                masses = np.delete(masses, np.where(np.isclose(masses, 0.32)), 0)
                masses = np.delete(masses, np.where(np.isclose(masses, 0.34)), 0)
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
        # Find a better way of doing this. Maybe this will require rethinking
        # how the radius is added to the table. Perhaps calculate radius and
        # bolometric corrections in the iso_table routine.
        if self.bandstr == "UBVRIplus":
            newtable["BC V"] = (
                -2.5 * newtable[self.logL_col] + 4.75 -
                newtable[band_translation["V"]])
            newtable["BC K"] = (
                -2.5 * newtable[self.logL_col] + 4.75 -
                newtable[band_translation["K"]])
        if "BP-RP" in met_table.colnames:
            newtable["BP-RP"] = (
                newtable[band_translation["BP"]] -
                newtable[band_translation["RP"]])
        combined_table = vstack([
            newtable, met_table[isochrone_highmass_indices]])
        self.iso_dict[age] = combined_table

    def _add_radius_column(self, fulltable, radcol=radius_col, logg=False):
        '''Add a radius column to this Isochrone's table.

        The column will be defined by the keywrod argument radcol. If the logg
        keyword is true, the radius will be defined using Mass and log(g). If
        the keyword is false, then the radius will be defined using Lbol and
        Teff.'''
        if logg:
            fulltable[radcol] = np.sqrt(
                fulltable[self.mass_col] / 10**(fulltable[self.logg_col] - 4.44))
        else:
            fulltable[radcol] = 10**(0.5*(
                fulltable[self.logL_col] - 4*(
                    fulltable[self.logteff_col] - np.log10(5777))))

    def _add_bolometric_correction(self, fulltable, band, bccol):
        '''Add a bolometric correction column.
        
        This method basically calculates a model Mbol - MK value for each EEP.'''
        mbol = -2.5 * fulltable[self.logL_col] + 4.75
        fulltable[bccol] = mbol - fulltable[band]

    def _add_color_column(self, fulltable, color, colorcol=""):
        '''Add a column representing a color.'''
        if not colorcol:
            colorcol = color
        blue, red = sed.split_color(color)
        fulltable[colorcol] = (
            fulltable[band_translation[blue]] -
            fulltable[band_translation[red]])

class MISTEvolutionaryTrack(models.StellarEvolutionaryTrack):
    '''A model of the MIST Evolutionary Tracks.'''
    age_col = "star_age"
    logteff_col = "log_Teff"
    logg_col = "log_g"
    logL_col = "log_L"
    # Note that the radius is inferred, not directly interpolated.
    radius_col = "radius"
    masses = np.arange(0.1, 2.0+0.02, 0.02)

    def __init__(self, fulltable, mass, feh, alpha, make_rad_col=True):
        '''Set up a MIST Evolutionary Track.

        In order to do so, there needs to be a table which corresponds to a
        timeseries which is specified by mass, iron and alpha abundance.'''
        super().__init__(fulltable, self.age_col, mass, feh, alpha)
        if make_rad_col:
            self._add_radius_column(fulltable, logg=False)


    @classmethod
    def track_from_file(
            cls, mass, feh, vvcrit=0.0, bandstr="UBVRIplus",
            MIST_version=LATEST_MIST_VERSION, MIST_PATH=paths.MIST_PATH):
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
        mist = cls(cleaned_table, mass, feh, 0.0)
        mist.MIST_version = MIST_version
        return mist

    @classmethod
    def nearest_mass(cls, mass):
        '''Return the mass of the track with the nearest mass to the given mass.

        Rounds the given mass to the track with the closest mass to the given
        mass. Since the tracks come in 0.05 mass intervals, this will handle
        rounding to the nearest 0.05.'''
        interval = 0.05
        newmass = np.around(mass / interval) * interval
        return newmass


    def restrict_phase(self, phasenums):
        '''Restrict the evolutionary track to a given phase.
        
        The phases are listed in the detailed information about MIST columns
        here: http://waps.cfa.harvard.edu/MIST/README_tables.pdf. In summary,
        they are:
        -1: PMS
        0: MS
        2: RGB 
        3: CHeB
        4: EAGB
        5: TPAGB
        6: postAGB
        9: WR
        For very massive stars, be careful about overlap between MS and WR.'''
        tables = []
        for phase in phasenums:
            if phase not in [-1, 0, 2, 3, 4, 5, 6, 6, 9]:
                raise ValueError("Don't recognize phase number {0:d}".format(
                    phase))
            tables.append(self.tracktable[
                self.tracktable["phase"] == phase])
        self.tracktable = vstack(tables)
        self.tracktable.sort(self.age_col)

    def _add_radius_column(self, fulltable, radcol=radius_col, logg=False):
        '''Add a radius column to this Isochrone's table.

        The column will be defined by the keywrod argument radcol. If the logg
        keyword is true, the radius will be defined using Mass and log(g). If
        the keyword is false, then the radius will be defined using Lbol and
        Teff.'''
        if logg:
            fulltable[radcol] = np.sqrt(
                fulltable[self.mass_col] / 10**(fulltable[self.logg_col] - 4.44))
        else:
            fulltable[radcol] = 10**(0.5*(
                fulltable[self.logL_col] - 4*(
                    fulltable[self.logteff_col] - np.log10(5777))))

    def age_at_radius(self, radius):
        '''Interpolate the age of the star when it becomes a certain radius.'''
        ind = bisect.bisect_left(self.tracktable[self.radius_col], radius)
        if ind == len(self.tracktable):
            raise ValueError("Track does not reach desired age.")
        rad1 = self.tracktable[self.radius_col][ind]
        rad2 = self.tracktable[self.radius_col][ind+1]
        age1 = self.tracktable[self.age_col][ind]
        age2 = self.tracktable[self.age_col][ind+1]
        newage = (age2 + (age2 - age1) / (rad2 - rad1) * (radius - rad2))
        return newage

class MISTIsoTemp(object):
    '''A series of subgiant datapoints at fixed temperature.'''
    age_col = "star_age"
    mass_col = "Mass"
    logg_col = "log_g"
    logL_col = "log_L"
    # Note that the radius is inferred, not directly interpolated.
    radius_col = "radius"

    def __init__(
            self, teff, startmass=1.0, endmass=2.0, step=0.02, feh=0.0,
            phases=[0, 2]):
        '''Make a sequence of mass at fixed teff.'''

        isotemp_list = []
        masses = []

        for m in np.arange(startmass, endmass, step):
            track = MISTEvolutionaryTrack.track_from_file(m, feh)
            track.restrict_phase(phases)
            # Get the age of the track when it is on the subgiant branch.
            try:
                ref_track_age = track.age_at_col(
                    track.logteff_col, np.log10(teff), col_increases=False)
            except IndexError:
                # If the track doesn't hit the given teff, ignore.
                continue
            interp_track = track.interpolate_at_age(ref_track_age)
            isotemp_list.append(interp_track)
            masses.append(m)

        isotemp_table = vstack(isotemp_list)
        isotemp_table[self.mass_col] = masses
        self.tracktab = isotemp_table


def download_MIST_isochrone(
        MIST_version, vvcrit, age_scale, age_list, feh, bandstr,
        folder=paths.MIST_PATH):
    '''Download the MIST isochrone for a specific [Fe/H].
    
    The MIST version must be 1.2 (as of June 2019). vvcrit must be either 0.0
    or 0.4. The age_scale denotes whether you want to specify age as a linear
    or log value. It should either have the value of "linear" or "log10".'''
    if MIST_version != 1.2:
        raise ValueError("Only MIST version {0:.1f} is available.".format(
            MIST_version))
    if vvcrit not in [0.0, 0.4]: 
        raise ValueError(
            "Only values of vvcrit are {0:.1f} and {1:1.f}".format(0.0, 0.4))
    if age_scale not in ["linear", "log10"]:
        raise ValueError(
            'Only age scales are "{0}" and "{1}"'.format("linear", "log10"))
    if bandstr not in ["UBVRIplus", "SDSSugriz"]:
        raise ValueError(
            "Do not support synthetic isochrones other than {0} yet".format(
                "UBVRIplus"))

    MIST_payload = {
        "version": "{0:.1f}".format(MIST_version), 
        "v_div_vcrit": "vvcrit{0:.1f}".format(vvcrit), 
        "age_scale": age_scale,
        "age_type": "list",
        "age_list": " ".join(format(x, ".2g") for x in age_list),
        "FeH_value": feh, 
        "output_option": "photometry",
        "output": bandstr,
        "Av_value": 0.0}

    ISO_URL = "http://waps.cfa.harvard.edu/MIST/iso_form.php"

    r_form = requests.post(ISO_URL, data=MIST_payload)
    # If something went wrong, throw an exception.
    r_form.raise_for_status()

    mist_soup = BeautifulSoup(r_form.text, "html.parser")

    links = mist_soup.find_all("a")

    if len(links) == 0:
        raise ValueError("No links on downloaded page.")
    elif len(links) > 1:
        raise ValueError("Multiple links on downloaded page.")

    url = list(urllib.parse.urlparse(r_form.url))
    iso_path = links[0].get("href")
    new_path = posixpath.join(posixpath.dirname(url[2]), iso_path)
    url[2] = new_path
    newurl = urllib.parse.urlunparse(url)

    r_file = requests.get(newurl)
    # If something went wrong, throw an exception.
    r_file.raise_for_status()

    file_stream = io.BytesIO(r_file.content)

    input_folder = (
        folder / abridged_isochrone_folder(MIST_version, vvcrit, bandstr))
    input_folder.mkdir(exist_ok=True)
    # Zipfile only takes path-like objects in versions greater than 3.6.2
    with zipfile.ZipFile(file_stream, 'r') as mist_zip:
        dest_path = input_folder / build_MIST_filename(
            feh, vvcrit=vvcrit, bandstr=bandstr, MIST_version=MIST_version)
        names = mist_zip.namelist()
        if len(names) == 0:
            raise ValueError("Isochrone Zip file has no contents.")
        elif len(names) > 1:
            raise ValueError("Too many isochrones in zip file")
        source = mist_zip.open(names[0], 'r')
        destination = dest_path.open("wb")
        shutil.copyfileobj(source, destination)

def download_MIST_evolutionary_track(
        MIST_version, vvcrit, feh, bandstr, Mmin=0.1, Mmax=1.3, dM=0.02,
        folder=paths.MIST_PATH):
    '''Download the MIST isochrone for a specific [Fe/H].'''
    if MIST_version != 1.2:
        raise ValueError("Only MIST version {0:.1f} is available.".format(1.2))
    if vvcrit not in [0.0, 0.4]: 
        raise ValueError(
            "Only values of vvcrit are {0:.1f} and {1:1.f}".format(0.0, 0.4))
    if bandstr not in ["UBVRIplus", "SDSSugriz"]:
        raise ValueError(
            "Do not support synthetic isochrones other than {0} yet".format(
                "UBVRIplus"))
    MIST_payload = {
        "version": "{0:.1f}".format(MIST_version), 
        "v_div_vcrit": "vvcrit{0:.1f}".format(vvcrit), 
        "mass_type": "range",
        "mass_range_low": Mmin,
        "mass_range_high": Mmax,
        "mass_range_delta": dM,
        "new_met_value": feh, 
        "output_option": "photometry",
        "output": bandstr,
        "Av_value": 0.0}

    ISO_URL = "http://waps.cfa.harvard.edu/MIST/track_form.php"

    r_form = requests.post(ISO_URL, data=MIST_payload)
    # If something went wrong, throw an exception.
    r_form.raise_for_status()

    mist_soup = BeautifulSoup(r_form.text, "html.parser")

    links = mist_soup.find_all("a")

    if len(links) == 0:
        raise ValueError("No links on downloaded page.")
    elif len(links) > 1:
        raise ValueError("Multiple links on downloaded page.")

    url = list(urllib.parse.urlparse(r_form.url))
    iso_path = links[0].get("href")
    new_path = posixpath.join(posixpath.dirname(url[2]), iso_path)
    url[2] = new_path
    newurl = urllib.parse.urlunparse(url)

    r_file = requests.get(newurl)
    # If something went wrong, throw an exception.
    r_file.raise_for_status()

    file_stream = io.BytesIO(r_file.content)

    input_folder = folder / name_track_folder(MIST_version, vvcrit, bandstr, feh)
    input_folder.mkdir(exist_ok=True)
    # Zipfile only takes path-like objects in Python versions greater than 3.6.2
    with zipfile.ZipFile(file_stream, 'r') as mist_zip:
        mist_zip.extractall(path=str(input_folder))

def download_MIST_isochrones(
        MIST_version=1.2, vvcrit=0.0, bandstr="UBVRIplus", 
        folder=paths.MIST_PATH):
    '''Download all of the MIST isochrones needed for interpolating.'''
    age_list = [1.25e8, 1e9, 4.5e9, 9e9]
    metallicity = np.concatenate(
        [np.arange(-4, -2, 0.5), np.arange(-2, 0.75, 0.25)])
    for feh in metallicity:
        print("Downloading [Fe/H]={0:.2f}".format(feh))
        download_MIST_isochrone(
            MIST_version, vvcrit, "linear", age_list, feh, bandstr,
            folder=folder)

def download_MIST_tracks(
        MIST_version=1.2, vvcrit=0.0, bandstr="UBVRIplus",
        folder=paths.MIST_PATH):
    '''Download all of the MIST Evolutionary Tracks needed.'''
    metallicity = np.concatenate(
        [np.arange(-4, -2, 0.5), np.arange(-2, 0.75, 0.25)])
    for feh in metallicity:
        print("Downloading [Fe/H]={0:.2f}".format(feh))
        download_MIST_evolutionary_track(
            MIST_version, vvcrit, feh, bandstr, Mmin=0.1, Mmax=1.3, dM=0.02,
            folder=folder)
        

def name_track_folder(MIST_version, vvcrit, bandstr, feh):
    '''Name the folder that contains evolutionary tracks.'''
    templatestr = "MIST_v{0:.1f}_vvcrit{1:.1f}_{2}_feh_{3}{4:4.2f}_tracks"
    newstr = templatestr.format(
        MIST_version, vvcrit, bandstr, dsep.assign_DSEP_sign(feh), abs(feh))
    return newstr

def name_track_file(mass):
    '''Name a MIST track file. This specifies a mass within the track folder.'''
    # Adding 0.5 because occasionally the value will be 13999.9999 instead of
    # 14.0.
    massformat = int(np.round(mass, 4) * 10000 + 0.5)
    templatestr = "{0:07d}M.track.eep.cmd"
    newstr = templatestr.format(massformat)
    return newstr

def abridged_isochrone_folder(MIST_version, vvcrit, bandstr):
    '''Name of the folder that contains abridged isochrones.

    This is a custom folder which only holds the isochrones for desired
    ages.'''
    templatestr = "MIST_v{0:.1f}_abridged_vvcrit{1:.1f}_{2}"
    newstr = templatestr.format(MIST_version, vvcrit, bandstr)
    return newstr

def full_isochrone_folder(MIST_version, vvcrit, bandstr):
    '''Name of the folder that contains the fullisochrones.

    This folder holds the full grid of isochrones..'''
    templatestr = "MIST_v{0:.1f}_vvcrit{1:.1f}_{2}"
    newstr = templatestr.format(MIST_version, vvcrit, bandstr)
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
                        MIST_version=1.2):
    '''Create a string that specifies the mist isochrone with the given params.'''

    mist_template =("MIST_v{0:3.1f}_feh_{1}{2:4.2f}_afe_{3}{4:3.1f}_"
                    "vvcrit{5:3.1f}_{6}.iso.cmd")
    filestr = mist_template.format(
        MIST_version, dsep.assign_DSEP_sign(feh), abs(feh), 
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

def plot_evtrack():
    '''Plot subgiant evolutionary tracks.'''
    masses = np.arange(0.86, 1.18, 0.08)
    for m in masses:
        track = MISTEvolutionaryTrack.track_from_file(m, 0.0)
        ms_isochrone = track.tracktable[track.tracktable["phase"] == 2.0]
        hr.absmag_teff_plot(
            10**ms_isochrone["log_Teff"], ms_isochrone["2MASS_Ks"], marker=".",
            ls="-")
