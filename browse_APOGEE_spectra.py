import argparse
import cmd
import fileinput
import webbrowser

from astropy.table import Table
from astropy.io.ascii import InconsistentTableError
import numpy.core.defchararray as npstr
import numpy as np

import astropy_util as au
import path_config as paths

DEFAULT_DLSB_DB = str(paths.DLSB_DB)
DEFAULT_NULL_DB = str(paths.NODL_DB)

# Maybe this isn't the correct way to do things.
class DLSB_prompt(cmd.Cmd):
    '''Record whether the apogee_id is a DLSB.

    Bring up a prompt asking the user if the given apogee_ID is a DLSB or
    not. If the user indicates yes, then the apogee_id will be appended to the
    dl_handle. If not, then it will be written to nl_handle.'''

    def __init__(self, dl_handle, nl_handle, apogee_id):
        cmd.Cmd.__init__(self)
        self.apid = apogee_id
        self.intro = ("Does {0} appear to be a Double-lined Spectroscopic "
                      "Binary? (NO/yes/skip)\n").format(self.apid)
        self.prompt = "APOGEE:> "

        if dl_handle.mode != "a":
            raise ValueError(
                "DLSB Database needs to be opened in append mode.")
        if nl_handle.mode != "a":
            raise ValueError(
                "Null Database needs to be opened in append mode.")

        self.dl_handle = dl_handle
        self.nl_handle = nl_handle

    def do_yes(self, arg):
        '''Confirm that the object is a DLSB.'''
        print(self.apid, file=self.dl_handle)
        return True

    def do_y(self, arg):
        '''Alias for yes.'''
        return self.do_yes(arg)

    def do_no(self, arg):
        '''Confirm that the object is NOT a DLSB'''
        print(self.apid, file=self.nl_handle)
        return True

    def do_n(self, arg):
        '''Alias for no'''
        return self.do_no(arg)

    def do_skip(self, arg):
        '''Do not make a decision on the object. Useful for ambiguous objects
        that ought to be revisited.'''
        return True

    def do_s(self, arg):
        '''Alias for skip'''
        return self.do_skip(arg)

    def default(self, line):
        '''Don't recognize the command.'''
        print("Do not recognize command: {0}".format(line))

    def emptyline(self):
        '''Default behavior is to say no.'''
        return self.do_no('')


def make_SAS_URL(apogee_ID, loc_ID):
    '''Take the apogee ID and Loc ID to make a SAS URL.

    Based on an APOGEE ID and Loc ID, will return a URL which goes to the
    APOGEE DR14 SAS page for that object.
    '''
    urlbase = ("https://sas.sdss.org/infrared/spectrum/view/stars=aspcap"
               "?apogee_id={0}&location_id={1:d}&commiss=0")
    url = urlbase.format(apogee_ID.strip(), loc_ID)
    return url

def read_DLSB_db(db_path=DEFAULT_DLSB_DB):
    '''Read in the DLSB database at the given path.

    The format of the DLSB database should be a list of APOGEE IDs, one per
    line, with no header. If the file is empty or does not exist, an empty
    table will be read.'''
    try:
        dlsbs = Table.read(db_path, names=["APOGEE_ID"],
                           format="ascii.no_header")
    except (FileNotFoundError, InconsistentTableError):
        dlsbs = Table(names=["APOGEE_ID"])
    return dlsbs

def read_null_db(db_path=DEFAULT_NULL_DB):
    '''Read in the non-DLSB database at the given path.

    The format of the non-DLSB database should be a list of APOGEE IDs, one per
    line, with no header. If the file is empty or does not exist, an empty
    table will be read.'''
    try:
        nulls = Table.read(db_path, names=["APOGEE_ID"],
                           format="ascii.no_header")
    except (FileNotFoundError, InconsistentTableError):
        nulls = Table(names=["APOGEE_ID"])
    return nulls

# Maybe try an "audit" to see how reproducible finding DLSBs is. Although this
# is sufficiently different that it may be better to make a whole new module
# than integrate it into this one.
def find_DLSBs(apids, locids, dlsbpath=DEFAULT_DLSB_DB, nullpath=DEFAULT_NULL_DB,
               verbose=False, apidcol="APOGEE_ID", locidcol="LOCATION_ID"):
    '''Inspect APOGEE spectra for Double-lined Spectroscopic Binaries.

    Automates the process of going through a list of APOGEE_IDs and LOC_IDs and
    pulling up the SDSS Science Archive Server (SAS). It will present a prompt
    to mark whether it is a DLSB or not. If so, the corresponding APOGEE_ID 
    will be appended to a text file which acts as a database of known DLSBs. 
    If not, it will be appended to a different text file to indicate that it 
    has been inspected.

    Before beginning the prompts, this object will remove all objects which
    already have entries in either of the two databases. This is to minimize
    duplication of effort if a separate dataset is inspected.'''

    # Filter out the objects that have already been seen.
    dlsbs = read_DLSB_db(db_path=dlsbpath)
    nulls = read_null_db(db_path=nullpath)

    if verbose:
        orig_length = len(apotable)
        print("{0:d} objects read in.".format(orig_length))
    apids = npstr.strip(apids)
    observed_indices = np.logical_or(
        au.mark_selections_in_columns(apids, npstr.strip(dlsbs["APOGEE_ID"])),
        au.mark_selections_in_columns(apids, npstr.strip(nulls["APOGEE_ID"])))
    new_apids = apids[~observed_indices]
    new_locids = locids[~observed_indices]
    print(len(new_apids))

    if verbose:
        num_removed = orig_length - len(apids)
        print("{0:d} objects were already classified.".format(num_removed))

    prompt = "APOGEE:>"
    total_nums = len(new_apids)
    # Now iterate through the table.
    for (i, (apoid, locid)) in enumerate(zip(new_apids, new_locids)): 
        SAS_url = make_SAS_URL(apoid, locid)
        print("Target {0:d}/{1:d}...".format(i+1, total_nums))
        if verbose:
            print("Opening {0}".format(SAS_url))
        webbrowser.open(SAS_url)
        with open(dlsbpath, 'a') as dl_handle:
            with open(nullpath, 'a') as nl_handle:
                askuser = DLSB_prompt(dl_handle, nl_handle, apoid)
                askuser.cmdloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dlsb-database", action="store", nargs="?", default=DEFAULT_DLSB_DB)
    parser.add_argument(
        "--null-database", action="store", nargs="?", default=DEFAULT_NULL_DB)
    parser.add_argument("FILES", action="store", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Move everything into a table.
    with fileinput.input(files=args.FILES) as f:
        apotable = Table.read(f, format="ascii.no_header", 
                              names=("APOGEE_ID", "LOC_ID"))

    find_DLSBs(
        apotable["APOGEE_ID"], apotable["LOCATION_ID"], 
        dlsbpath=args.dlsb_database, nullpath=args.null_database)



