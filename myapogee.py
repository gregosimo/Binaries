import urllib
import os
from pathlib import Path

from astropy.io import fits
import requests

DR14_URL = "https://sas.sdss.org/infrared/spectrum/view/"
DR14_VISITS = urllib.parse.urljoin(DR14_URL, "visits")
DR14_DATA = urllib.parse.urljoin(DR14_URL, "data/")
SAS_PATH = (Path(os.environ["SDSS_LOCAL_SAS_MIRROR"]) / "dr14" / "apogee" / 
            "spectra" / "redux")
APRED_VERS = "r8"
APSTAR_VERS = "stars"
ASPCAP_VERS = "l31c"
RESULTS_VERS = "l31c.1"
TELESCOPE = "apo25m"

###############################################################################
# Routines to download APOGEE spectra #
###############################################################################

def download_combined_apogee_spectra(apogee_ids, location_ids):
    '''Download multiple combined spectra.

    Downloads multiple combined spectra into the $SDSS_LOCAL_SAS_MIRROR tree
    based on lists of apogee_ids and location_ids. The two lists should be of
    equal length. The download follows the rules of
    download_combined_apogee_spectrum.'''
    assert(len(apogee_ids) == len(location_ids))
    for (apo, loc) in zip(apogee_ids, location_ids):
        download_combined_apogee_spectrum(apo, loc)

def download_combined_apogee_spectrum(apogee_id, location_id, dest=""):
    '''Download the combined APOGEE spectrum for a target object.

    Given an APOGEE_ID and LOCATION_ID, download the combined visit spectrum
    for an object. Note that if dlbase is not specified, then the file will be
    downloaded into the SDSS tree given by
    SAS_PATH/APRED_VERS/APSTAR_VERS/TELESCOPE/LOCATION_ID/.
    
    Authentication should be handled by the .netrc file.'''
    fullurl = construct_apogee_spectrum_url("combined", "fits", apogee_id,
            location_id)
    filename = "apStar-{0}-{1}.fits".format(APRED_VERS, apogee_id)
    if not dest:
        dest = (SAS_PATH / APRED_VERS / APSTAR_VERS / TELESCOPE / 
                  str(location_id))
    download_apogee_product(fullurl, dest, filename)

def download_aspcap_apogee_spectra(apogee_ids, location_ids):
    '''Download multiple combined spectra.

    Downloads multiple combined spectra into the $SDSS_LOCAL_SAS_MIRROR tree
    based on lists of apogee_ids and location_ids. The two lists should be of
    equal length. The download follows the rules of
    download_combined_apogee_spectrum.'''
    assert(len(apogee_ids) == len(location_ids))
    for (apo, loc) in zip(apogee_ids, location_ids):
        download_aspcap_apogee_spectrum(apo, loc)

def download_aspcap_apogee_spectrum(apogee_id, location_id, dest=""):
    '''Download the ASPCAP spectrum for a target object.

    Given an apogee_id and location_id, download the aspcap spectrum for an
    object. Note that if dlbase is not specified, then the file will be
    downloaded into the $SDSS_LOCAL_SAS_MIRROR given by
    SAS_PATH/APRED_VERS/APSTAR_VERS/ASPCAP_VERS/RESULTS_VERS/LOCATION_ID/'''
    fullurl = construct_apogee_spectrum_url("aspcap", "fits", apogee_id,
            location_id)
    filename = "aspcapStar-{0}-{1}-{2}.fits".format(APRED_VERS,
            RESULTS_VERS, apogee_id)
    if not dest:
        dest = (SAS_PATH / APRED_VERS / APSTAR_VERS / ASPCAP_VERS /
                RESULTS_VERS / str(location_id))
    download_apogee_product(fullurl, dest, filename)

def construct_apogee_spectrum_url(stars, file_format,
        apogee_id, location_id, baseurl=DR14_DATA):
    '''Constructs a URL to get an APOGEE spectrum.

    The baseurl specifies where the "data" folder is. The stars can be either
    "combined" or "aspcap" for the combined spectrum or aspcap spectrum. the
    file_format can be either "fits" or "csv". Lastly, the apogee_id and
    location_id of the target are needed. This is enough to locate the spectrum
    in the SAS.'''
    fullurl = urllib.parse.urljoin(
            baseurl, 
            "stars={0}/format={1}/apogee_id={2}/location_id={3:d}".format(
                stars, file_format, apogee_id, location_id))
    return fullurl


def download_apogee_product(url, destination, filename):
    '''Downloads a target at a url to a destination.

    The url should be the full url where the file is. It does not allow for any
    post requests.'''
    destination.mkdir(parents=True, exist_ok=True)
    fullpath = destination / filename
    if not fullpath.is_file():
        r = requests.get(url)
        r.raise_for_status()
        with open(str(destination / filename), 'wb') as outfile:
            outfile.write(r.content)

###############################################################################
# Navigate the SAS archive #
###############################################################################

def combined_spectrum_SAS_path(
    location_id, apogee_id="", basepath=SAS_PATH, apred_vers=APRED_VERS,
    apstar_vers=APSTAR_VERS, telescope=TELESCOPE):
    '''Construct the file path for a combined spectrum on the SAS.

    The necessary information to construct a SAS directory is the location_id.
    If a filename is also desired for the path, then it will also be included
    if apogee_id is also specified. The other quantities are inferred from the 
    module; however, they can be overridden if need be.'''
    dirpath = (basepath / apred_vers / apstar_vers / telescope / 
               str(location_id))
    if apogee_id:
        filename = combined_spectrum_filename(apogee_id, apred_vers=apred_vers)
        fullpath = dirpath / filename
    else:
        fullpath = dirpath

    return fullpath

def aspcap_spectrum_SAS_path(
    location_id, apogee_id="", basepath=SAS_PATH, apred_vers=APRED_VERS,
    apstar_vers=APSTAR_VERS, aspcap_vers=ASPCAP_VERS,
    results_vers=RESULTS_VERS):
    '''Construct the file path for an aspcap spectrum on the SAS.

    The necessary information to construct a SAS directory is the location_id.
    If a filename is also desired for the path, then it will also be included
    if apogee_id is also specified. The other quantities are inferred from the
    module; however, they can be overridden if need be.'''
    dirpath = (basepath / apred_vers / apstar_vers / aspcap_vers / 
                results_vers / str(location_id))

    if apogee_id:
        filename = aspcap_spectrum_filename(apogee_id, apred_vers=apred_vers)
        fullpath = dirpath / filename
    else:
        fullpath = dirpath

    return fullpath

def combined_spectrum_filename(apogee_id, apred_vers=APRED_VERS):
    '''Output the standard filename for an APOGEE combined spectrum.

    The filename has the format "apStar-(apred_vers)-(apogee_id).fits".'''
    filename = "apStar-{0}-{1}.fits".format(apred_vers, apogee_id)
    return filename

def aspcap_spectrum_filename(apogee_id, apred_vers=APRED_VERS,
                             results_vers=RESULTS_VERS):
    '''Output the standard filename for an APOGEE ASPCAP spectrum.

    The filename has the format
    "aspcapStar-(apred_vers)-(results_vers)-(apogee_id).fits".'''
    filename = "aspcapStar-{0}-{1}-{2}.fits".format(apred_vers,
            results_vers, apogee_id)
    return filename

###############################################################################
# Routines to read in APOGEE spectra #
###############################################################################

class APOGEEStar:
    '''A class representing a single apStar dataset.

    This function essentially contains all the information in the
    apStar-(apred_vers)-(apogee_id).fits file. Methods to access and manipulate
    the data will be provided as needed.'''
    def __init__(
        self, location_id, apogee_id, saspath=SAS_PATH, apred_vers=APRED_VERS,
        apstar_vers=APSTAR_VERS, telescope=TELESCOPE):
        '''Load in the apStar entry for this apogee_id.

        The two main quantities needed to uniquely identify the target in the
        SAS tree are the location_id and apogee_id. The other quantities will
        be drawn from the global variables in this module. They can be
        overridden if necessary.
        '''
        hdulist = read_combined_apogee_data(
            location_id, apogee_id, basepath=saspath, apred_vers=apred_vers,
            apstar_vers=apstar_vers, telescope=telescope)
        # This is a FITS "header" which behaves like a dict.
        self._header = hdulist[0].header
        nvisits = self._header["NVISITS"]
        specdata = hdulist[1].data
        specheader = hdulist[1].header
        errdata = hdulist[2].data
        errheader = hdulist[2].header
        maskdata = hdulist[3].data
        maskheader = hdulist[3].header

        # For 1 visit, all spectra point to one spectrum.
        if nvisits == 1:
            main_spectrum = APOGEESpectrum(
                specdata, errdata, maskdata, specheader, errheader, maskheader)
            self._pixspectrum = main_spectrum
            self._globspectrum = main_spectrum
            self._visitspectra = [main_spectrum]
        elif nvisits > 1:
            self._pixspectrum = APOGEESpectrum(
                specdata[0,:], errdata[0,:], maskdata[0,:], specheader,
                errheader, maskheader)
            self._globspectrum = APOGEESpectrum(
                specdata[1,:], errdata[1,:], maskdata[1,:], specheader,
                errheader, maskheader)
            self._visitspectra = [APOGEESpectrum(
                specdata[i,:], errdata[i,:], maskdata[i,:], specheader,
                errheader, maskheader) for i in range(2, 2+nvisits)]
        else:
            raise ValueError("Invalid value of NVISITS: {0}.".format(nvisits))

        # I'll fetch the other data if it's necessary. I don't think sky
        # spectra are particularly important at this time.
        self._rvtable = hdulist[9].data
        hdulist.close()

    @property
    def spectrum(self):
        '''Get the preferred spectrum for the star.
        
        This will either be the pixel-weighted spectrum or the globally
        weighted spectrum depending on what is set as "preferred".'''
        return self.pixspectrum

    @property
    def pixel_spectrum(self):
        '''Get the pixel-weighted spectrum.'''
        return self._pixspectrum

    @property
    def global_spectrum(self):
        '''Get the globally-weighted spectrum.'''
        return self._globspectrum

    @property
    def visit_mjds(self):
        '''Get the MJDs of the observations.'''
        return self._rvtable["MJD"]

    @property
    def rvs(self):
        '''Get the radial velocities of the system.'''
        return self._rvtable["VHELIO"]

    @property
    def rv_errors(self):
        '''Get the radial velocity uncertainties of the system.'''
        return self._rvtable["VHELIO"]

    @property
    def twomass_id(self):
        '''Get the 2MASS ID of the object.'''
        return self._header["OBJID"]

    def visit_spectrum(self, mjd):
        '''Get the visit spectrum at the given mjd.'''
        index = np.nonzero(self.visit_mjds == mjd)
        assert(len(index[0]) <= 1)
        spec = self._visitspectra[index[0][0]]
        return spec

    def plot_rv(self):
        '''Plot the APOGEE RV timeseries of this system.
        
        This will only plot when an object has been visited multiple times.
        Otherwise, it will print the single RV measurement.'''
        rvs = self.rvs
        if len(rvs) == 1:
            print("The RV of this system is {0:2f} km/s".format(rvs[0]))
        elif len(rvs) > 1:
            plt.errorbar(self.mjds, rvs, self.rv_errors, 'bo')
            plt.xlabel("MJD")
            plt.ylabel("RV (km/s)")
            plt.title("{0} RV curve".format(self.twomass_id))
        else:
            raise ValueError("{0} has no observations.".format(
                self.twomass_id)) 

class APOGEESpectrum(object):
    '''Class which holds an APOGEE spectrum.

    This class is optimized to hold spectra which come from APOGEE. In
    particular, they will not need to be given a wavelength array because it
    should be uniform.'''
    def __init__(self, fluxes, uncertainties, mask, fluxheader, uncheader,
                 maskheader):
        '''Create an APOGEE spectrum. 

        To instantiate a spectrum, you will need the actual flux values for
        each of the pixels as well as the fits header that comes with them.'''
        self._fluxes = fluxes
        self._uncertainties = uncertainties
        self._mask = mask
        self._fluxheader = fluxheader
        self._uncheader = uncheader
        self._maskheader = maskheader

        self._wavelengths = 10**(np.arange(
            fluxheader["CRVAL1"], 
            fluxheader["CRVAL1"] + fluxheader["CDELT1"] * len(fluxes),
            fluxheader["CDELT1"]))/1e4


def read_combined_apogee_data(
    apogee_id, location_id, path="", sasbase=SAS_PATH, apred_vers=APRED_VERS,
    apstar_vers=APSTAR_VERS, telescope=TELESCOPE):
    '''Read a combined APOGEE data file from the file system.

    The APOGEE_ID and LOCATION_ID are necessary to do this. Without a path
    specified, the function will look in the SDSS tree at
    $SDSS_LOCAL_SAS_MIRROR. If path is a path to a directory, then this
    function will read the file "apStar-(APRED)-(apogee_id).fits" in that
    directory. If it is a path to a file, then it will read that file.
    
    This will return an astropy HDUList. Note that the hdulist should be CLOSED
    when it is no longer needed.'''
    if not path:
        path = combined_spectrum_SAS_path(
            location_id, apogee_id, basepath=sasbase, apred_vers=apred_vers,
            apstar_vers=apstar_vers, telescope=telescope)
    else:
        if not path.is_file():
            filename = combined_spectrum_filename(apogee_id)
            path = path / filename

    hdulist = fits.open(str(path))
    return hdulist

