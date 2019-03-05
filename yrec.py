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

band_translation = {
    "V": "Mv", "B-V": "B-V", "V-I": "V-Ic", "V-K": "V-K", "V-Ks": "V-K", "J-K":
    "J-K", "H-K": "H-K"}

class YRECIsochrone(models.StellarIsochrone):
    '''A class that encapsulates a Baraffe isochrone.'''
    # These are characteristic of all MIST Isochrones
    # I can move these to the StellarIsochrone class using @property and
    # NotImplementedError.
    age_col = "age"
    mass_col = "Mass/Ms"
    teff_col = "Teff(K)"
    logL_col = "Log(L/Ls)"
    # Note that the radius is inferred, not directly interpolated.
    radius_col = "R/Rs"
    def __init__(self, feh, fulltable, alpha=0, Yinit=0.2703, Zinit=1.42857e-2, 
                 vvcrit=0.00, AV=0.0, bandstr="CIT2", bol_bands=["V"]):
        for bol in bol_bands:
            colname = "BC {0}".format(bol)
            self._add_bolometric_correction(
                fulltable, band_translation[bol], colname)
        # This should make a dictionary which has age as a key and that
        # subtable as a value.
        fullgroups = fulltable.group_by(self.age_col)
        iso_dict = {
            np.round(key[0] / 10**(np.floor(np.log10(key[0]))),
                     2)*10**(np.floor(np.log10(key[0]))): val for key, val in zip(
                fullgroups.groups.keys, fullgroups.groups)}
        super().__init__(feh, alpha, 0, Yinit, Zinit, vvcrit, bandstr, iso_dict)
        self.AV=AV
        self.increasing_colnames = set("V")
        self.decreasing_colnames = (
            set([self.mass_col, self.teff_col, self.logL_col]) |
            set(band_translation.keys()) - set("V"))

    def iso_table(self, age):
        '''Return the table corresponding to the isochrone at the given age.
        The given age should be given in log10(yr).'''
        return self.iso_dict[age]

    @classmethod
    def isochrone_from_file(
        cls, YREC_PATH=paths.YREC_PATH):
        '''Read in a Baraffe isochrone from a file.'''
        filename = "lj98corz01757katmjhk.txt"
        yrec_pleiades = Table.read(
            str(YREC_PATH / filename), 
            format="ascii.basic", header_start=847, guess=False, 
            data_start=848, data_end=893)
        yrec_pleiades["age"] = 0.12
        yrec_obj = cls(
            0.0, yrec_pleiades, alpha=0.0, vvcrit=0.0, bandstr="")
        return yrec_obj

    def _add_bolometric_correction(self, fulltable, band, bccol):
        '''Add a bolometric correction column.
        
        This method basically calculates a model Mbol - MK value for each EEP.'''
        mbol = -2.5 * fulltable[self.logL_col] + 4.75
        fulltable[bccol] = mbol - fulltable[band]

