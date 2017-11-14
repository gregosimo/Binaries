'''
This is a script to populate the ipython namespace with all of the subsets of
interest when checking the rotation properties of the APOGEE'''
from functools import partial

import numpy as np
from astropy.table import vstack, Table
import matplotlib.pyplot as plt
import scipy

import read_catalog as catin
import catalog
import astropy_util as au
import sample_characterization as samp
import hrplots as hr


class DataSplitter:
    '''A class to simplify exploring various subsets of particular datasets.

    An object of this class will hold a master dataset, and will be able to
    make cuts along various dimensions in order to explore the contents of the
    sample within the various cuts.'''

    def __init__(self, data, splitgroups=None, indices=None):
        '''Initialize Datasplitter to have tab as the master database.
        
        The data argument should be an astropy table.'''
        self.data = data
        if not splitgroups:
            self.splitgroups = {}
        else:
            self.splitgroups = splitgroups
        if not indices:
            self.indices = {}
        else:
            self.indices = indices

    def split_by_col(self, col, splitvalues, splitnames,
                     invert_inequality=False):
        '''Make a simple split in the dataset using one column.

        The column name used for the split should be given as col. The values
        marking the split boundaries should be given as a sequence in
        splitvalues. Labels for the different subsets of col should be given in
        splitnames as a sequence. The size of splitnames should be 1 greater
        than splitvalues. Splitvalues should sorted in ascending order for
        splitnames to make sense.

        The column will be split by vals < splitvalues[0], splitvalues[0]
        <= vals < splitvalues[1], ..., splitvalues[-2] <= vals <
        splitvalues[-1], vals >= splitvalues[-1]. Setting the
        invert_inequality flag to true will change <= to < and > to >=.
        '''
        colvalues = self.data[col]

        # Anytime an ordered comparison needs to be made, things can get weird 
        # with masked values. Let me know if any are encountered.
        assert not np.any(np.ma.getmask(colvalues))
        
        try:
            ordered_splitvalues = sorted(splitvalues)
        except TypeError:
            # In this case there is only one object.
            splitvalues = [splitvalues]
            ordered_splitvalues = splitvalues
        if ordered_splitvalues != splitvalues:
            raise ValueError("Splitvalues needs to be sorted.")

        # Invert_inequality basically transforms < to <= and >= to >
        if not invert_inequality:
            # First make the lowest table.
            self.indices[splitnames[0]] = colvalues < splitvalues[0]
            # Then make intermediate tables.
            for i, (low, high) in enumerate(zip(
                    splitvalues[:-1], splitvalues[1:])):
                self.indices[splitnames[i+1]] = np.logical_and(
                    colvalues >= low, colvalues < high)
            # Now make the highest table.
            self.indices[splitnames[-1]] = colvalues >= splitvalues[-1]
        else:
            # First make the lowest table.
            self.indices[splitnames[0]] = colvalues <= splitvalues[0]
            # Then make intermediate tables.
            for i, (low, high) in zip(splitvalues[:-1], splitvalues[1:]):
                self.indices[splitnames[i+1]] = np.logical_and(
                    colvalues > low, colvalues <= high)
            # Now make the highest table.
            self.indices[splitnames[-1]] = colvalues > splitvalues[-1]

    def subsample(self, namelist):
        '''Get a specified subsample.

        Subsamples are specified by passing a list of subsample names, and the
        intersection of all of those subsamples will be returned. Note that
        individual splits are mutually-exclusive, so if two subsamples are
        specified within the same split, the returned table will be empty. This
        function will try to raise ValueError when this occurs.'''
        index_list = [np.ones(len(self.data))]
        for name in namelist:
            try:
                index = self.indices[name]
            except KeyError:
                if isinstance(namelist, str):
                    raise ValueError("Please pass a list, not a string")
                else:
                    raise ValueError(
                        "{0} is not a valid subsample name. Run names() to get "
                        "currently available subsample names.".format(name))
            index_list.append(index)
        fullindex = au.multi_logical_and(*index_list)
        subsample = self.data[fullindex]
        if len(subsample) == 0:
            self._check_namelist_for_conflicts(namelist)

        return subsample

    def _check_namelist_for_conflicts(self, namelist):
        '''Check the namelist to find conflicting entries.'''
        nameset = set(namelist)
        for sg in self.splitgroups.values():
            comboset = nameset & sg
            if len(comboset) > 1:
                raise ValueError("{0} are conflicting!".format(comboset))

    def _check_indices_partition(self, crit):
        '''Check that indices for a given criterion partition data.

        For the given criteria for subsampling data, verify that every row
        in the dataset is counted exactly once for all divisions. This will
        ensure the subsets are complete, and that they are mutually exclusive.
        '''
        divnames = self.splitgroups[crit]
        indices = [self.indices[div] for div in divnames]
        # Check that every row is counted at least once.
        assert np.all(np.sum(indices, axis=0) == 1)

    def names(self):
        '''Print out all the valid subsamples in this dataset.'''
        for k, v in self.splitgroups.items():
            print("{0}: {1}".format(k, v))

    def __repr__(self):
        '''Print a Python representation of the DataSplitter.'''
        reprstr = "{0}({1}, splitgroups={2}, indices={3})".format(
            self.__class__, repr(self.data), repr(self.splitgroups), 
            repr(self.indices))
        return reprstr

    def generate_partition_census(self, categories):
        '''Break down the sample into subsamples with numbers.

        This function will return a tree-like structure containing a census of
        various subdivisions of the data. Each node will contain a 2-tuple with
        an integer as the first value, and a dictionary containing the rest of
        the tree as a second value. The first value represents the total number
        of objects in the tree below.'''
        return self._traverse_partition_census(categories, len)

    def _traverse_partition_census(self, categories, datafunc, prevsamps=[]):
        '''Traverse tree and get subsample outputs..'''
        catdict = {}
        if not categories:
            return catdict
        else:
            currentcat = categories[0]
            samples = self.splitgroups[currentcat]
            for sampname in samples:
                cursamps = prevsamps + [sampname]
                samptable = self.subsample(cursamps)
                samplen = datafunc(samptable)
                catdict[sampname] = (
                    samplen, self._traverse_partition_census(
                        categories[1:], datafunc, prevsamps=cursamps))
        return catdict

def format_census_tree(census_tree, indents=""):
    '''Write the partition census to a string.

    Splits up the dataset according to the list in categories and then
    makes a human-readable string of how objects in the categories are
    distributed.'''
    fullstr = ""
    for k, v in census_tree.items():
        if v[0] != 0:
            curstr = "{0}+ {1}: {2:d}\n".format(indents, k, v[0])
            nextstr = format_census_tree(v[1], indents+"| ")
            fullstr = fullstr + curstr + nextstr
    return fullstr

class StarSplitter(DataSplitter):
    '''Split dataset with stellar properties.

    Current stellar properties are: Teff and Log(g).'''

    def split_logg(self, col, splitvalues, splitnames, logg_crit="logg",
                   invert_inequality=False):
        '''Split the data by log(g).

        Since there are many different ways to measure log(g), the desired
        column should be given as col. The values to split about should be
        given in splitvalues, and the names of the classes should be given in
        splitnames. The type of log(g) measurement used should be given in
        logg_crit, for example APOGEE or Huber log(g). This will associate the
        splitnames group with the specific log(g) measurement.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(col, splitvalues, splitnames, invert_inequality)
        self.splitgroups[logg_crit] = set(splitnames)
        self._check_indices_partition(logg_crit)

    def split_teff(self, col, splitvalues, splitnames, teff_crit="teff",
                   invert_inequality=False):
        '''Split the data by Teff.

        Since there are many different ways to measure Teff, the desired
        column should be given as col. The values to split about should be
        given in splitvalues, and the names of the classes should be given in
        splitnames. The type of Teff measurement used should be given in
        teff_crit, for example APOGEE or Huber Teff. This will associate the
        splitnames group with the specific Teff measurement.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(col, splitvalues, splitnames, invert_inequality)
        self.splitgroups[teff_crit] = set(splitnames)
        self._check_indices_partition(teff_crit)

class McQuillanSplitter(StarSplitter):
    '''Keep an organized database of various cuts on McQuillan data.'''

    def __init__(self, data=None):
        '''Initialize a splitter of McQuillan data.'''
        if not data:
            data = catin.mcquillan_with_stelparms()
            super().__init__(data)

        self.split_logg(
            "logg", 3.5, ["Huber giant", "Huber dwarf"], logg_crit="Huber logg")
        self.split_teff(
            "teff", [5500, 6500], ["Jen Cool Huber", "Cool Huber", "Hot Huber"], 
            teff_crit="Huber Teff")

        self.split_period("Prot", [1, 5], ["very rapid", "rapid", "slow"])

    def split_period(self, col, splitvalues, splitnames, period_crit="period",
                     invert_inequality=False):
        '''Split the data by period.

        Since there are many different ways to measure period, the desired
        column should be given as col. The values to split about should be
        given in splitvalues, and the names of the classes should be given in
        splitnames. The type of period measurement used should be given in
        period_crit, for example McQuillan period. This will associate the
        splitnames group with the specific period measurement.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(col, splitvalues, splitnames, invert_inequality)
        self.splitgroups[period_crit] = set(splitnames)
        self._check_indices_partition(period_crit)

class APOGEESplitter(StarSplitter):
    '''Keep an organized database of various cuts on APOGEE data.'''

    def __init__(self, data=None):
        '''Initialize a splitter of APOGEE data.
        
        If the data parameter is passed, then it will be set to the full data
        sample. If not, then it will be read in manually.'''
        if not data:
            data = catin.dr14_with_KIC_stelparms()
            data["LOGG_FIT"] = data["FPARAM"][:,1]
        super().__init__(data)

        self.split_logg(
            "logg", 3.5, ["Huber giant", "Huber dwarf"], logg_crit="Huber logg")
        self.split_logg(
            "LOGG_FIT", 3.5, ["APOGEE giant", "APOGEE dwarf"], 
            logg_crit="APOGEE logg")

        self.split_teff(
            "teff", [5500, 6500], ["Jen Cool Huber", "Cool Huber", "Hot Huber"], 
            teff_crit="Huber Teff")
        self.split_teff(
            "TEFF", [5500, 6500], 
            ["Jen Cool APOGEE", "Cool APOGEE", "Hot APOGEE"], 
            teff_crit="APOGEE Teff")

        self.split_vscatter(1, ["RV Nonvar", "RV Var"])

        self.split_vsini(7, ["Vsini nondet", "Vsini det"])

        self.split_spectroscopic_rapid_rotators(
            [1, 5], ["Very rapid rotators", "Rapid rotators", "Slow rotators"])

        self.split_dlsb()

        self.split_asteroseismic_dwarfs()

        self.split_McQuillan_periods(kiccol="kepid")

        self.split_by_ASPCAP_flags()




    def split_vscatter(self, splitvalues, splitnames, col="VSCATTER",
                       vscatter_crit="VSCATTER", invert_inequality=False):
        '''Split the data by vscatter.

        Split the sample based on the boundaries given in splitvalues. The
        names for the categories should be given in splitnames. If the vscatter
        values are in a column other than VSCATTER, it can be specified with
        the col keyword. If there are other sources of vscatter, then they can
        be specified in vscatter_crit.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(col, splitvalues, splitnames, invert_inequality)
        self.splitgroups[vscatter_crit] = set(splitnames)
        self._check_indices_partition(vscatter_crit)

    def split_vsini(self, splitvalues, splitnames, col="VSINI",
                    vsini_crit="VSINI", invert_inequality=False):
        '''Split the data by vsini.

        Split the sample based on the boundaries given in splitvalues. The
        names for the categories should be given in splitnames. If the vsini
        values are in a column other than VSINI, it can be specified with
        the col keyword. If there are other sources of vsini, then they can
        be labeled separately by vsini_crit.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(col, splitvalues, splitnames, invert_inequality)
        self.splitgroups[vsini_crit] = set(splitnames)
        self._check_indices_partition(vsini_crit)

    def split_spectroscopic_rapid_rotators(
        self, splitperiods, splitnames, radius_col="radius", 
        vsini_col="VSINI", det_limit=7, rapid_crit="Spec rapid", 
        invert_inequality=False):
        '''Split sample based on spectroscopic measures of rapid rotation.

        Split the sample by vsini consistent with equatorial velocities of
        splitperiods. The function with translate the rotation periods to
        equatorial velocities using the radius in order to make the cut. For
        objects with vsini lower than the detection limit, they will be assumed
        to be rotating at the detection limit. Note
        that because of the inclination, some period rapid-rotators may fall
        into the slower-rotating bins. The column containing the vsinis and
        radius are given in vsini_col and radius_col. If there are other types
        of cuts made, they can be labeled separately by rapid_crit

        For more information on invert_inequality, see split_by_col.
        '''
        tempcol = au.generate_random_string(12)
        self.data[tempcol] = catalog.vsini_to_period(
            np.maximum(self.data[vsini_col], 7), self.data[radius_col])[0]
        self.split_by_col(tempcol, splitperiods, splitnames, invert_inequality)
        self.splitgroups[rapid_crit] = set(splitnames)
        self._check_indices_partition(rapid_crit)
        del(self.data[tempcol])
        assert tempcol not in self.data.colnames

    def split_dlsb(
        self, apid_col="APOGEE_ID", dl_names=("DLSB", "No DLSB", "Unknown DLSB"),
        dlsb_crit="DLSB", dlsb_db=catalog.DLSB_PATH, 
        nodl_db=catalog.NON_DLSB_PATH):
        '''Split sample based on presence of double-lines.

        Split the sample based on previous observations of double-lined
        spectroscopic binaries. The matching is done via the APOGEE ID of the
        targets. Databases which contain APOGEE IDs of confirmed DLSBs and
        non-DLSBs are in dlsb_db and nodl_db. Labels for the confirmed DLSB,
        confirmed non-DLSB, and unconfirmed classes should be given as a tuple
        in dl_names. Other analyses of double-lined spectroscopic binaries can
        be specified by dlsb_crit.
        '''
        apids = self.data[apid_col]
        self.indices[dl_names[0]] = catalog.mark_DLSB_indices(
            apids, dlsb_db=dlsb_db)
        self.indices[dl_names[1]] = catalog.mark_non_DLSB_indices(
            apids, nodl_db=nodl_db)
        self.indices[dl_names[2]] = np.logical_not(np.logical_or(
            self.indices[dl_names[0]], self.indices[dl_names[1]]))
        assert not np.any(np.logical_and(
            self.indices[dl_names[0]], self.indices[dl_names[1]]))
        self.splitgroups[dlsb_crit] = set(dl_names)
        self._check_indices_partition(dlsb_crit)

    def split_asteroseismic_dwarfs(
        self, astero_names=("Asteroseismic", "Non-asteroseismic"), 
        apid_col="APOGEE_ID", astero_crit="astero"):
        '''Separate asteroseismic dwarfs in dataset.

        Separates the asteroseismic dwarfs from the dataset. The asteroseismic
        dwarfs are currently fetched from the APOKASC catalog. Other
        determinations of the asteroseismic dwarfs can be specified in
        astero_crit.'''
        apokasc = catin.read_APOKASC_catalog()[["2MASS_ID", "RADIUS_DW"]]
        ast_dwarf = catalog.filter_invalid_APOGEE_entries(apokasc, "RADIUS_DW")
        self.indices[astero_names[0]] = au.mark_selections_in_columns(
            self.data[apid_col], ast_dwarf["2MASS_ID"])
        self.indices[astero_names[1]] = np.logical_not(
            self.indices[astero_names[0]])
        self.splitgroups[astero_crit] = set(astero_names)
        self._check_indices_partition(astero_crit)

    def split_McQuillan_periods(
        self, mcq_names=("Mcq", "No Mcq"), kiccol="KIC", mcq_crit="Mcq"):
        '''Separate objects with and without McQuillan periods.

        Splits off objects with and without McQuillan periods. The names for
        the two classes should be given in mcq_names. The two datasets are
        cross-matched by KIC numbers in kiccol.
        '''
        mcq = catin.read_McQuillan_catalog()
        self.indices[mcq_names[0]] = au.mark_selections_in_columns(
            self.data[kiccol], mcq["KIC"])
        self.indices[mcq_names[1]] = np.logical_not(self.indices[mcq_names[0]])
        self.splitgroups[mcq_crit] = set(mcq_names)
        self._check_indices_partition(mcq_crit)

    def split_by_ASPCAP_flags(
        self, qual_names=("Bad", "Warn", "vsini", "Good"),
        aspcapcol="ASPCAPFLAGS", aspcap_crit="ASPCAP"):
        '''Separate objects by ASPCAP quality flags.

        Splits off objects based on the quality indicated by the ASPCAP
        flags. The four classes are those with the STAR_BAD flag, those with
        only the STAR_WARN flag, those with only the VSINI_WARN flag, and those
        without any of the previous three aforementioned flags. It will look
        for ASPCAP flags in the column given in aspcapcol. If there is more
        than one set of aspcapflags, they can be distinguished using the
        aspcap_crit flag.'''
        flags = self.data[aspcapcol]
        # Pure bad indices
        self.indices[qual_names[0]] = catalog.bad_ASPCAP_indices(
            flags, warn=False)
        badwarn_indices = catalog.bad_ASPCAP_indices(flags, warn=True)
        self.indices[qual_names[1]] = np.logical_and(
            np.logical_not(self.indices[qual_names[0]]), badwarn_indices)
        self.indices[qual_names[2]] = np.logical_and(
            np.logical_not(badwarn_indices), catalog.warn_VSINI_indices(flags))
        self.indices[qual_names[3]] = np.logical_not(np.logical_or(
            badwarn_indices, self.indices[qual_names[2]]))
        self.splitgroups[aspcap_crit] = set(qual_names)
        self._check_indices_partition(aspcap_crit)

def gen_samp(name):
    '''Function to generate the given sample objects which was broken down.'''
    try:
        return globals()[name]
    except KeyError:
        return vstack([gen_samp(ob) for ob in gendict[name]])

def HR_Param_Check():
    '''Check how logg and teff parameters match with each other.'''
    fulltable = gen_samp("good_dr14")
    dwarfs = gen_samp("cool_kic_dwarfs")

    # I want a 2x2 plot of the full HR diagram.
    hr.logg_teff_comparison_subsample(
        fulltable["logg"], fulltable["teff"], fulltable["LOGG_FIT"],
        fulltable["TEFF"], dwarfs["logg"], dwarfs["teff"], dwarfs["LOGG_FIT"],
        dwarfs["TEFF"], "KIC", "APOGEE") 

def KIC_APOGEE_Param_Diff():
    '''See trends in Teff and log(g) differences with Teff.'''
    fulltable = gen_samp("good_dr14")

    tempdiff = (fulltable["teff"] - fulltable["TEFF"]) / fulltable["teff"]
    loggdiff = fulltable["logg"] - fulltable["LOGG_FIT"]
    linex = [min(fulltable["teff"]), max(fulltable["teff"])]
    
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.plot(fulltable["teff"], tempdiff, ls="", marker="*")
    ax1.plot(linex, [0,0], ls="-", marker="", lw=2)
    ax1.set_xlabel("KIC Teff")
    ax1.set_ylabel("Fractional KIC - APOGEE Teff Difference")
    hr.invert_x_axis(ax1)
    ax2.plot(fulltable["teff"], loggdiff, ls="", marker="*")
    ax2.plot(linex, [0,0], ls="-", marker="", lw=2)
    ax2.set_xlabel("KIC Teff")
    ax2.set_ylabel("KIC - APOGEE Log(g) Difference")
    hr.invert_x_axis(ax2)

def rotation_dist():
    '''Plot the vsini distribution over Teff.'''
    full_samp = gen_samp("cool_kic_dwarfs")
    samp = full_samp[full_samp["VSINI"] > 0]

    linex = [min(samp["teff"]), max(samp["teff"])]

    plt.plot(samp["teff"], samp["VSINI"], ls="", marker="*")
    plt.plot(linex, [7, 7], ls="--", c="orange", marker="")
    plt.xlabel("KIC Teff")
    plt.ylabel("VSINI")
    hr.invert_x_axis()

    nbins = 20
    top_percent, edges, indices = scipy.stats.binned_statistic(
        samp["teff"], samp["VSINI"], statistic=lambda x: np.percentile(x, 66.7), 
        bins=nbins)
    median_percent, edges, indices = scipy.stats.binned_statistic(
        samp["teff"], samp["VSINI"], statistic=lambda x: np.percentile(x, 50.0), 
        bins=nbins)
    bottom_percent, edges, indices = scipy.stats.binned_statistic(
        samp["teff"], samp["VSINI"], statistic=lambda x: np.percentile(x, 33.3), 
        bins=nbins)
    
    midpoints = (edges[:-1]+edges[1:])/2
    plt.errorbar(midpoints, median_percent, yerr=[
        median_percent-bottom_percent, top_percent-median_percent], marker="o",
                 c="r", ls="", lw=2, zorder=3)
