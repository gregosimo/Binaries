import argparse
import cmd
import fileinput
import webbrowser

import astropy_util as au

DEFAULT_DLSB_DB = "./DLSB.txt"
DEFAULT_NULL_DB = "./noDL.txt"

# Maybe this isn't the correct way to do things.
class DLSB_prompt(cmd.Cmd):
    '''Record whether the apogee_id is a DLSB.

    Bring up a prompt asking the user if the given apogee_ID is a DLSB or
    not. If the user indicates yes, then the apogee_id will be appended to the
    dl_handle. If not, then it will be written to nl_handle.'''

    def __init__(self, dl_handle, nl_handle, apogee_id):
        cmd.Cmd.__init__(self)
        self.apid = apogee_id
        self.intro = ("Does {0} appear to be a Double-lined Spectroscopic"
                      "Binary? (no/yes/skip)\n").format(self.apid)
        self.prompt = "APOGEE:>"

        if dl_handle.mode != "a":
            raise ValueError(
                "DLSB Database needs to be opened in append mode.")
        if nl_handle.mode != "a":
            raise ValueError(
                "Null Database needs to be opened in append mode.")

    def do_yes(self, arg):
        '''Confirm that the object is a DLSB.'''
        dl_handle.write(self.apid)
        return True

    def do_y(self, arg):
        '''Alias for yes.'''
        return self.do_yes(arg)

    def do_no(self, arg):
        '''Confirm that the object is NOT a DLSB'''
        nl_handle.write(self.apid)
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


def make_SAS_URL(apogee_ID, loc_ID):
    '''Take the apogee ID and Loc ID to make a SAS URL.

    Based on an APOGEE ID and Loc ID, will return a URL which goes to the
    APOGEE DR14 SAS page for that object.
    '''
    urlbase = ("https://sas.sdss.org/infrared/spectrum/view/stars=aspcap]"
               "?apogee_id={0}&location_id={1:d}&commiss=0")
    url = urlbase.format(apogee_ID, loc_ID)
    return url

# Maybe try an "audit" to see how reproducible finding DLSBs is. Although this
# is sufficiently different that it may be better to make a whole new module
# than integrate it into this one.
def find_DLSBs(apotable, dlsbpath=DEFAULT_DLSB_DB, nullpath=DEFAULT_NULL_DB,
               verbose=False):
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
    dlsbs = Table.read(dlsbpath, names=["APOGEE_ID"])
    nulls = Table.read(nullpath, names=["APOGEE_ID"])

    if verbose:
        orig_length = len(apotable)
        print("{0:d} objects read in.".format(orig_length))

    apotable = au.filter_column_from_subtable(
        apotable, "APOGEE_ID", dlsbs["APOGEE_ID"])
    apotable = au.filter_column_from_subtable(
        apotable, "APOGEE_ID", nulls["APOGEE_ID"])

    if verbose:
        num_removed = orig_length - len(apotable)
        print("{0:d} objects were already classified.".format(num_removed))

    prompt = "APOGEE:>"
    # Now iterate through the table.
    for row in apotable:
        apoid = row["APOGEE_ID"]
        locid = row["LOC_ID"]

        SAS_url = make_SAS_URL(apoid, locid)
        if verbose:
            print("Opening {0}".format(SAS_url))
        webbrowser.open(SAS_url)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dlsb-database", action="store", nargs="?", default=DEFAULT_KICDB)
    parser.add_argument(
        "--null-database", action="store", nargs="?", default=DEFAULT_NULL_DB)
    parser.add_argument("FILES", action="store", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Move everything into a table.
    with fileinput.input(files=args.FILES) as f:
        apotable = Table.read(f, format="ascii.no_header", 
                              names=("APOGEE_ID", "LOC_ID"))

    find_DLSBs(
        apotable, dlsbpath=args.dlsb_database, nullpath=args.null_database)



