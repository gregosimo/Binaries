import urllib
import os

import requests

DR14_URL = "https://sas.sdss.org/infrared/spectrum/view/"
DR14_VISITS = urllib.parse.urljoin(DR14_URL, "visits")
DR14_DATA = urllib.parse.urljoin(DR14_URL, "data/")
SAS_PATH = os.environ["SDSS_LOCAL_SAS_MIRROR"]
APRED_VERS = "r8"
APSTAR_VERS = "stars"
ASPCAP_VERS = "l31c"
RESULTS_VERS = "l31c.1"
TELESCOPE = "apo25m"


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

def download_aspcap_apogee_spectrum(apogee_id, location_id, dest=""):
    '''Download the ASPCAP spectrum for a target object.

    Given an apogee_id and location_id, download the aspcap spectrum for an
    object. Note that if dlbase is not specified, then the file will be
    downloaded into the SDSS trr given by
    SAS_PATH/APRED_VERS/APSTAR_VERS/ASPCAP_VERS/RESULTS_VERS/LOCATION_ID/'''
    fullurl = construct_apogee_spectrum_url("aspcap", "fits", apogee_id,
            location_id)
    filename = "aspcapStar-{0}-{1}-{2}.fits".format(APRED_VERS,
            RESULTS_VER, apogee_id)
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
    fullurl = urllib.parse.url(
            baseurl, 
            "stars={0}/format={1}/apogee_id={2}/location_id={3:d}".format(
                stars, file_format, apogee_id, location_id))
    return fullurl


def download_apogee_product(url, destination, filename):
    '''Downloads a target at a url to a destination.

    The url should be the full url where the file is. It does not allow for any
    post requests.'''
    r = requests.get(url)
    outfile = open(str(destination / filename))
    outfile.write(r.content)
    outfile.close()
