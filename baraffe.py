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

band_translation = {"V": "Mv", "H": "Mh", "K": "Mk", "Ks": "Mk"}

class BaraffeIsochrone(models.StellarIsochrone):
    '''A class that encapsulates a Baraffe isochrone.'''
    # These are characteristic of all MIST Isochrones
    # I can move these to the StellarIsochrone class using @property and
    # NotImplementedError.
    age_col = "age"
    mass_col = "M/Ms"
    teff_col = "Teff"
    logL_col = "L/Ls"
    # Note that the radius is inferred, not directly interpolated.
    radius_col = "R/Rs"
    def __init__(self, feh, fulltable, alpha=0, Yinit=0.2703, Zinit=1.42857e-2, 
                 vvcrit=0.00, AV=0.0, bandstr="CIT2", bol_bands=["V", "K"]):
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
        self.increasing_colnames = set(band_translation.values())
        self.decreasing_colnames = set([self.mass_col, self.teff_col,
                                        self.logL_col])

    def iso_table(self, age):
        '''Return the table corresponding to the isochrone at the given age.
        The given age should be given in log10(yr).'''
        return self.iso_dict[age]

    @classmethod
    def isochrone_from_file(
        cls, bandstr="CIT2", BARAFFE_PATH=paths.BARAFFE_PATH):
        '''Read in a Baraffe isochrone from a file.'''
        filename = "BHAC15_iso.{0}".format(bandstr)
        Baraffe_young = Table.read(
            str(BARAFFE_PATH / filename), 
            format="ascii.commented_header", header_start=14, guess=False, 
            data_start=473, data_end=500, comment='!')
        Baraffe_young["age"] = 0.12
        Baraffe_old = Table.read(
            str(BARAFFE_PATH / filename), 
            format="ascii.commented_header", header_start=49, guess=False, 
            data_start=651, data_end=675, comment='!')
        Baraffe_old["age"] = 1
        Baraffe_table = vstack([Baraffe_young, Baraffe_old])
        baraffe_obj = cls(
            0.0, Baraffe_table, alpha=0.0, vvcrit=0.0, bandstr=bandstr)
        return baraffe_obj

    def _add_bolometric_correction(self, fulltable, band, bccol):
        '''Add a bolometric correction column.
        
        This method basically calculates a model Mbol - MK value for each EEP.'''
        mbol = -2.5 * fulltable[self.logL_col] + 4.75
        fulltable[bccol] = mbol - fulltable[band]

