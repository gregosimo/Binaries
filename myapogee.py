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
TELESCOPE = "apo25m"


def download_combined_apogee_spectrum(
    apogee_id, location_id, commiss=0, dlbase=""):
    '''Download the combined APOGEE spectrum for a target object.

    Given an APOGEE_ID and LOCATION_ID, download the combined visit spectrum
    for an object. Note that if dlbase is not specified, then the file will be
    downloaded into the SDSS tree given by
    SAS_PATH/APRED_VERS/APSTAR_VERS/TELESCOPE/LOCATION_ID/.
    
    Authentication should be handled by the .netrc file.'''
    get_payload = {"apogee_id": apogee_id, "location_id": location_id, "stars":
                   "combined", "format": "fits"}
    payload_string = ("stars=combined/format=fits/" 
                      "apogee_id={0}/location_id={1:d}".format(apogee_id,
                                                              location_id))
    r = requests.get(urllib.parse.urljoin(DR14_DATA, payload_string))
