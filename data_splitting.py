'''
This is a script to populate the ipython namespace with all of the subsets of
interest when checking the rotation properties of the APOGEE'''
from functools import partial

import numpy as np
import numpy.core.defchararray as npstr
from astropy.table import vstack, Table
import matplotlib.pyplot as plt
import scipy

import read_catalog as catin
import catalog
import astropy_util as au
import sample_characterization as samp
import hrplots as hr
import rotation_consistency as rot
import eclipsing_binaries as ebs
import path_config as paths


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

    def split_by_col(self, col, splitvalues, splitnames, crit,
                     null_value=None, invert_inequality=False):
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

        If a null_value is given, then splitnames will have to have an extra
        name which classifies all targets that are equal to null_value.
        Null_value can be NaN, any finite value, or np.ma.masked.

        Note that due to the exclusion mechanism as part of the DataSplitter,
        that splitnames are not allowed to begin with a tilde (~) character.
        '''
        # Don't split when there are no splitvalues.
        if splitvalues == []:
            raise ValueError("Need values to split on")

        colvalues = self.data[col]

        # Anytime an ordered comparison needs to be made, things can get weird 
        # with masked values. Let me know if any are encountered.
        if null_value is not np.ma.masked:
            assert not np.any(np.ma.getmask(colvalues))
        if null_value is not np.nan:
            assert np.all(np.logical_not(np.isnan(colvalues)))
        
        try:
            ordered_splitvalues = sorted(splitvalues)
        except TypeError:
            # In this case there is only one object.
            splitvalues = [splitvalues]
            ordered_splitvalues = splitvalues
        if np.any(ordered_splitvalues != splitvalues):
            raise ValueError("Splitvalues needs to be sorted.")

        # Check to make sure that splitvalues doesn't have any strings that
        # start with a tilde.
        _check_namelist_for_tildes(splitnames)

        # If something goes wrong, I'll want to have a backup to restore the
        # object. But I haven't faced a use-case for this yet. So just have an
        # outline of something to do.
        backup_indices = self._backup_crit(crit)
        
        # If there's already an existing criteria, delete all the indices
        # corresponding to that criteria. This will be a total replacement.
        if crit in self.splitgroups:
            self.delete_crit(crit)

        # Once all of the old indices are deleted, check if any of the new
        # indices are in the index array. If they are, then there's a conflict
        # and the function should throw an exception.
        for indexname in splitnames:
            if indexname in self.indices:
                self._restore_crit(crit, backup_indices)
                raise ValueError(
                    "Splitnames conflicts with other index names.")

        indexlist = []
        try:
            # Invert_inequality basically transforms < to <= and >= to >
            if not invert_inequality:
                # First make the lowest table.
                indexlist.append(np.ma.filled(colvalues < splitvalues[0], 0))
                # Then make intermediate tables.
                for i, (low, high) in enumerate(zip(
                        splitvalues[:-1], splitvalues[1:])):
                    indexlist.append(np.ma.filled(np.logical_and(
                        colvalues >= low, colvalues < high), 0))
                # Now make the highest table.
                indexlist.append(np.ma.filled(colvalues >= splitvalues[-1], 0))
            else:
                # First make the lowest table.
                indexlist.append(np.ma.filled(colvalues <= splitvalues[0], 0))
                # Then make intermediate tables.
                for i, (low, high) in enumerate(zip(
                        splitvalues[:-1], splitvalues[1:])):
                    indexlist.append(np.ma.filled(np.logical_and(
                        colvalues > low, colvalues <= high), 0))
                # Now make the highest table.
                indexlist.append(np.ma.filled(colvalues > splitvalues[-1], 0))

            # If we have a null_value, gather up all of those.
            if null_value is not None:
                indexlist.append(au.check_null(colvalues, null_value))
                # If null_value can be compared, then remove it from the other
                # indices.
                if null_value == null_value:
                    newindexlist = []
                    for ind in indexlist[:-1]:
                        newindexlist.append(
                            np.logical_and(ind, np.logical_not(indexlist[-1])))
                    newindexlist.append(indexlist[-1])
                    indexlist = newindexlist
        except:
            self._restore_crit(crit, backup_indices)
            raise

        self._setup_indices(splitnames, indexlist, crit)

    def _setup_indices(self, splitnames, indices, crit):
        '''Automatically map splitnames to indices in the internal dict.

        Avoid repetitively setting the values in self.indices to the actual
        indices and setting the crit values.'''
        if len(splitnames) != len(indices):
            raise ValueError("Got {0} names. Expected {1}.".format(
                len(splitnames), len(indices)))
        for name, index in zip(splitnames, indices):
            self.indices[name] = index
        self.splitgroups[crit] = set(splitnames)
        self._check_indices_partition(crit)

    def _setup_complement_index(self, splitnames, index, crit):
        '''Indices where one index is given and the other is complementary

        This is a shorthand for _setup_indices where
        indexarr = [index, np.logical_not(index)]
        
        Don't forget that the index should correspond to the first entry of
        splitnames while the complement should correspond to the second entry.
        '''
        indexarr = [index, np.logical_not(index)]
        self._setup_indices(splitnames, indexarr, crit)

    def _backup_crit(self, crit):
        '''Makes a backup of the indices under crit.

        Return a dictionary containing the indices stored under crit.'''
        pass

    def _restore_crit(self, crit, backups):
        '''Restore the indices in backups under crit.

        Takes a dictionary containing indices, and restores them in the
        indextable under crit.'''
        pass

    def subsample(self, namelist):
        '''Get a specified subsample.

        Subsamples are specified by passing a list of subsample names, and the
        intersection of all of those subsamples will be returned. If a
        particular should be excluded, then this can be indicated with the use
        of a tilde (~). This function will try to raise ValueError if two of
        the same time of names are specified..'''
        indices = self._subsample_indices(namelist)
        subsample = self.data[indices]

        return subsample

    def split_subsample(self, namelist):
        '''Make another splitter on the subsample given in the namelist.

        The return splitter will be a valid splitter of the same type on the
        subsample. The types of splits which were not involved in namelist will
        also be passed down.
        '''
        indices = self._subsample_indices(namelist)
        subsampled_data = self.data[indices]

        subsampled_crits = {}
        nameset = set(namelist)
        # Carry over all criteria which aren't part of the subsample filtering.
        for crit, critset in self.splitgroups.items():
            if nameset.isdisjoint(critset):
                subsampled_crits[crit] = critset

        subsampled_indices = {}
        for critset in subsampled_crits.values():
            for indexname in critset:
                subsampled_indices[indexname] = self.indices[indexname][indices]

        subsample_splitter = type(self)(
            subsampled_data, splitgroups=subsampled_crits,
            indices=subsampled_indices)
        return subsample_splitter

    def split_sample(self, group, otherparms=None):
        '''Split the sample according to the group partition.

        Return a dictionary containing the subgroup Additional
        restrictions on the sample can be specified as a list with otherparms.
        Effectively, this will be like returning subsample() for each subsample
        in group.'''
        samples = tuple([self.subsample(otherparms+[grp]) for grp in
                        self.splitgroups[group]])
        return samples
            

    def _check_namelist_for_conflicts(self, namelist):
        '''Check the namelist to find conflicting entries.'''
        nameset = set([name.lstrip("~") for name in namelist])
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

    def _subsample_indices(self, namelist):
        '''Get the indices corresponding to the given namelist.
    
        Return a boolean array which returns true for all rows corresponding to
        the intersection of requirements in namelist.'''
        # Make sure the namelist doesn't have any conflicting columns.
        self._check_namelist_for_conflicts(namelist)

        index_list = [np.ones(len(self.data))]
        for name in namelist:
            exclusion = False
            if name.startswith("~"):
                name = name[1:]
                exclusion = True
            try:
                index = self.indices[name]
            except KeyError:
                if isinstance(namelist, str):
                    raise ValueError("Please pass a list, not a string")
                else:
                    raise KeyError(
                        "{0} is not a valid subsample name. Run names() to get "
                        "currently available subsample names.".format(name))
            if exclusion:
                index = np.logical_not(index)
            index_list.append(index)
        fullindex = au.multi_logical_and(*index_list)
        return fullindex

    def subsample_len(self, namelist):
        '''Get the size of a subsample.

        This function automatically gets the size of a subsample in a fast way
        as opposed to running subsample() and getting its length.'''
        indices = self._subsample_indices(namelist)
        indexlen = np.count_nonzero(indices)
        return indexlen

    def __len__(self):
        '''Get the size of the full sample.'''
        return self.subsample_len([])

    def names(self):
        '''Print out all the valid subsamples in this dataset.'''
        print(str(self))

    def __repr__(self):
        '''Print a Python representation of the DataSplitter.'''
        reprstr = "{0}({1}, splitgroups={2}, indices={3})".format(
            self.__class__, repr(self.data), repr(self.splitgroups), 
            repr(self.indices))
        return reprstr

    def __str__(self):
        '''Print an informative representation of the DataSplitter.

        This is essentially the names() function.
        '''
        outstr = "\n".join(["{0}: {1}".format(k, v) for k,v in
                            self.splitgroups.items()])
        return outstr

    def __eq__(self, other):
        '''Test for equality between this splitter and others.'''
        # They must be the same type of splitter, otherwise they will not be
        # equal.
        if not isinstance(other, self.__class__):
            return NotImplemented

        data_eq = np.all(self.data == other.data)
        splitgroup_eq = self.splitgroups == other.splitgroups
        # I can't just use self.indices == other.indices
        # Because it tries to compare numpy arrays.
        index_eq = True
        for key, value in self.indices.items():
            try:
                if not np.all(other.indices[key] == value):
                    index_eq = False
                    break
            except KeyError:
                index_eq = False
                break

        return data_eq and splitgroup_eq and index_eq

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

    def delete_crit(self, crit):
        '''Delete the selected criterion from the DataSplitter.

        If a criteron is no longer desired, run this method to delete that
        criterion as well as the corresponding indices.'''
        for index in self.splitgroups[crit]:
            del(self.indices[index])
        del(self.splitgroups[crit])

    def _crit_of_index(self, indname):
        '''Get the criteria corresponding to the given index.'''
        for crit, indices in self.splitgroups:
            if indname in indices:
                return crit
        raise KeyError(
            "{0} is not a valid index name. Run names() to get "
            "currently available index names.".format(indname))


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

def _check_namelist_for_tildes(namelist):
    '''Make sure none of the names in namelist begin with a tilde.'''
    for name in namelist:
        if name.startswith("~"):
            raise ValueError(
                "{0} cannot be a name that starts with a tilde (~)".format(
                    name))


class KeplerSplitter(DataSplitter):
    '''Split dataset with Kepler stellar properties.

    Current stellar properties are: Teff and Log(g).'''

    def __init__(
            self, data, splitgroups=None, indices=None, kic_col="kepid", 
            tm_col="tm_designation"):
        '''Initialize the splitter for a dataset specifying the index columns.

        Set up the Splitter with the given data. The splitter is also indexed
        both by KIC IDs as well as 2MASS IDs. The columns specifying those
        should be given in kic_col and tm_col. They'll be available as
        self.kic_col and self.tm_col.'''
        super().__init__(data, splitgroups=splitgroups, indices=indices)
        self.kic_col = kic_col
        self.tm_col = tm_col

    def split_logg(self, col, splitvalues, splitnames, logg_crit="logg",
                   null_value=None, invert_inequality=False):
        '''Split the data by log(g).

        Since there are many different ways to measure log(g), the desired
        column should be given as col. The values to split about should be
        given in splitvalues, and the names of the classes should be given in
        splitnames. The type of log(g) measurement used should be given in
        logg_crit, for example APOGEE or Huber log(g). This will associate the
        splitnames group with the specific log(g) measurement.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(
            col, splitvalues, splitnames, logg_crit, null_value=null_value, 
            invert_inequality=invert_inequality)

    def split_Ciardi_logg(
        self, loggcol, teffcol, splitnames=("Giant", "Dwarf"), 
        logg_crit="logg"):
        '''Make a Ciardi cut by log(g)

        This is a proposed delineation between dwarfs and giants for Kepler
        targets. The cut is as follows:
                    3.5                   if Teff >= 6000
        log(g) >= { 4.0                   if Teff <= 4250      }
                    5.2 - (2.8e-4 * Teff) if 4250 < Teff < 6000
        '''
        teff = self.data[teffcol]
        logg = self.data[loggcol]
        dwarf_indices = np.logical_or(np.logical_or(
            np.logical_and(teff >= 6000, logg >= 3.5), 
            np.logical_and(teff <= 4250, logg >= 4.0)),
            np.logical_and(
                np.logical_and(teff < 6000, teff > 4250),
                logg >= 5.2 - 2.8e-4 * teff))
        giant_indices = np.logical_not(dwarf_indices)

        self._setup_complement_index(
            splitnames[::-1], dwarf_indices, logg_crit)


    def split_Ciardi_Color(
        self, jcol="jmag", hcol="hmag", 
        splitnames=("Color Giant", "Color Dwarf"), color_crit="J-H sep",
        invert_inequality=False):
        '''Make a Ciardi cut by J-H color.

        This is a proposed delineation between dwarfs and giants in the M
        regime for Kepler targets. In this case, the cut classifies objects
        with J-H > 0.75 as giants. Therefore, columns for the j and h
        magnitudes should be provided in jcol and hcol.
        
        The first element of splitnames should be the label given to giants
        found by this method. The second element of splitnames should be the
        label given to dwarfs found by this method.'''
        jhcolor = self.data[jcol] - self.data[hcol]
        if not invert_inequality:
            dwarf_indices = jhcolor < 0.75
            giant_indices = jhcolor >= 0.75
        else:
            dwarf_indices = jhcolor <= 0.75
            giant_indices = jhcolor > 0.75

        indexlist = [giant_indices, dwarf_indices]
        self._setup_indices(splitnames, indexlist, color_crit)

    def split_teff(self, col, splitvalues, splitnames, teff_crit="teff",
                   null_value=None, invert_inequality=False):
        '''Split the data by Teff.

        Since there are many different ways to measure Teff, the desired
        column should be given as col. The values to split about should be
        given in splitvalues, and the names of the classes should be given in
        splitnames. The type of Teff measurement used should be given in
        teff_crit, for example APOGEE or Huber Teff. This will associate the
        splitnames group with the specific Teff measurement.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(
            col, splitvalues, splitnames, teff_crit, null_value=null_value, 
            invert_inequality=invert_inequality)

    def split_sufficient_quarter_obs(
            self, qneeded=8, qused=range(3, 15), quartercol="st_quarters", 
            splitnames=("Low Quarter Fraction", "OK Quarter Fraction"),
            quarter_crit="Mcq Quarter Fraction"):
        '''Split off objects without enough quarters observed.

        The exact quarters to consider are given as an iterable in the qused 
        parameter; the default is to specify Q3-14. Note that Q0 cannot be 
        specified. The minimum number of quarters needed to be sufficient is 
        given as qneeded. 

        The column which has the quarter flags should be quartercol. This
        function assumes that the column has strings of length 17,
        corresponding to the quarters. And each index, starting with Q1, is
        either a 1 or a 0 depending on whether that quarter was observed or
        not.
        
        Splitnames should be a 2-tuple where the first element is the label for
        objects without sufficient quarters observed. The second element is the
        label for objects with sufficient quarters observed. The quarter_crit
        specifies the labels which are used for this particular split.'''
        rel_quarter = au.slicer_vectorized(self.data[quartercol], qused)
        qobserved = npstr.count(rel_quarter, '1')
        indexarr = [qobserved < 8, qobserved >= 8]
        self._setup_indices(splitnames, indexarr, quarter_crit)

    def split_original_KIC_params(
            self, paramcol="K-Teff", 
            splitnames=("Orig KIC Present", "Orig KIC Not Present"), 
            orig_crit="Orig KIC"):
        '''Split the data based on the presence of original KIC parameters.

        Because the DataSplitter does not handle null values well, It may be useful
        to automatically split by the presence of original KIC params. One useful
        aspect is that the dataset can be split by KIC params after the entries
        with KIC parameters are included.'''
        notindices = self.data[paramcol].mask
        indices = np.logical_not(notindices)

        indexarr = [indices, notindices]
        self._setup_indices(splitnames, indexarr, orig_crit)

    def split_mag(
            self, magcol, mags, splitnames=("Bright", "Faint", "No Mag"), 
            mag_crit="mag", null_value=np.ma.masked, invert_inequality=False):
        '''Split based on a magnitude cut.

        The column which contains magnitudes should be given in magcol. The
        value to be split between should be given as mags.
        Splitnames should have the name for the bright objects as the first
        element, and the name for the faint elements as the second element. The
        mag_crit specifies the labels used for this particular split.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(
            magcol, mags, splitnames, mag_crit, null_value=null_value, 
            invert_inequality=invert_inequality)

    def split_evstate(
            self, teff_col="teff", logg_col="LOGG_FIT", 
            giant_subgiant_points=[(5000, 3.5), (3500, 3.5)],
            subgiant_dwarf_points=[(5690, 4.43), (4640, 3.72)], 
            splitnames=("Giant", "Subgiant", "Dwarf", "NO_EV"), 
            crit="APOGEE Evolutionary State"):
        '''Split the cool dwarf sample by evolutionary state.

        This function splits the sample according to linear cuts. There is a
        linear cut between giants and subgiants, and a linear cut between
        subgiants and dwarfs.

        The default cut between giants and subgiants is a horizontal log(g) 
        cut of 3.5. The default cut between subgiants and dwarfs is a linear
        cut calibrated to an APOGEE HR diagram.
        '''
        topdiv_slope = (
            (giant_subgiant_points[0][1] - giant_subgiant_points[1][1]) /
            (giant_subgiant_points[0][0] - giant_subgiant_points[1][0]))
        topdiv_coord = giant_subgiant_points[0]
        bottomdiv_slope = (
            (subgiant_dwarf_points[0][1] - subgiant_dwarf_points[1][1]) /
            (subgiant_dwarf_points[0][0] - subgiant_dwarf_points[1][0]))
        bottomdiv_coord = subgiant_dwarf_points[0]

        giant_subgiant_div = topdiv_slope * (
            self.data[teff_col] - topdiv_coord[0]) + topdiv_coord[1]
        subgiant_dwarf_div = bottomdiv_slope * (
            self.data[teff_col] - bottomdiv_coord[0]) + bottomdiv_coord[1]

        unclassified_indices = self.data[teff_col] < 0
        giant_indices = np.logical_and(
            self.data[logg_col] < giant_subgiant_div,
            np.logical_not(unclassified_indices))

        subgiant_indices = np.logical_and(
            np.logical_and(
                self.data[logg_col] >= giant_subgiant_div, 
                self.data[logg_col] < subgiant_dwarf_div), 
            np.logical_not(unclassified_indices))

        dwarf_indices = np.logical_and(
            self.data[logg_col] >= subgiant_dwarf_div,
            np.logical_not(unclassified_indices))

        indexarr = [giant_indices, subgiant_indices, dwarf_indices,
                    unclassified_indices]
        self._setup_indices(splitnames, indexarr, crit)

    def split_mk_evstate(
            self, teff_col="TEFF", mk_col="M_K", 
            giant_subgiant_points=[(5000, 0.7), (3500, 0.7)],
            subgiant_dwarf_points=[(5015, 2.4), (5625, 2.4)], 
            splitnames=("Giant", "Subgiant", "Dwarf", "NO_EV"), 
            crit="APOGEE Evolutionary State"):
        '''Split the cool dwarf sample into evolutionary states using M_K.

        Split the catalog according to the K-band absolute magnitude.'''
        self.split_evstate(
            teff_col=teff_col, logg_col=mk_col,
            giant_subgiant_points=giant_subgiant_points,
            subgiant_dwarf_points=subgiant_dwarf_points, splitnames=splitnames,
            crit=crit)

    def split_Berger_EVstate(
            self, class_col="class", splitnames=(
                "Berger Giant", "Berger Subgiant", "Berger Main Sequence", 
                "Berger Cool Binary", "Missing Berger Class"), 
            crit="Berger Evolutionary State"):
        '''Split sample according to the evolutionary state in Berger (2018).

        Berger et al (2018) classified objects in the Teff-Luminosity plane as
        being giants, subgiants, main sequence dwarfs, and binaries on the cool
        end, where the main sequence splits from subgiants.'''
        dwarf_indices = self.data[class_col] == 0
        subgiant_indices = self.data[class_col] == 1
        giant_indices = self.data[class_col] == 2
        binary_indices = self.data[class_col] == 3
        missing_indices = self.data[class_col].mask

        indexarr = [giant_indices, subgiant_indices, dwarf_indices,
                    binary_indices, missing_indices]
        self._setup_indices(splitnames, indexarr, crit)

    def split_photometric_quality(
            self, phot_col, phot_err_col, splitnames=(
                "Detection", "Blend", "Bad photometry"), crit="Photometry Cut",
            null_value=np.ma.masked):
        '''Split sample according to photometry quality.

        Some of the targets do not have good K-band photometry. This function
        separates those with photometry from those without. This function
        basically uses the existence of an error in phot_err_col to flag
        whether the photometry is good. If not, then the corresponding value is
        a upper limit.'''
        det_indices = self.data[phot_err_col] != null_value
        blend_indices = np.logical_and(self.data[phot_col] != null_value,
                                       self.data[phot_err_col] == null_value)
        bad_indices = self.data[phot_col] == null_value

        indexarr = [det_indices, blend_indices, bad_indices]
        self._setup_indices(splitnames, indexarr, crit)

    def split_Gaia(self, parallax_col="parallax", 
                   splitnames=("In Gaia", "Not in Gaia"), gaia_crit="Gaia present"):
        '''Split sample based on the presence of Gaia distances.
        
        Not all of the Kepler sample has Gaia distances. In order to deal with
        that, this function splits the sample into whether there are Gaia
        distances or not.'''
        good_indices = ~self.data[parallax_col].mask
        
        self._setup_complement_index(splitnames, good_indices, gaia_crit)

    def split_provenance(
            self, prov_col, prov_list, splitnames=None, 
            prov_crit="Provenances"):
        '''Split based on provenance for a given column.
        
        The column which holds the provenances should be given in prov_col. A
        list of provenances should be given in prov_list.'''
        if splitnames is None:
            splitnames = prov_list
        indexarr = []
        for prov in prov_list:
            indexarr.append(self.data[prov_col] == prov)
        self._setup_indices(splitnames, indexarr, prov_crit)

class McQuillanSplitter(KeplerSplitter):
    '''Keep an organized database of various cuts on McQuillan data.'''

    def __init__(self, data=None, splitgroups=None, indices=None, 
                 kic_col="kepid", tm_col="tm_designation"):
        '''Initialize a splitter of McQuillan data.'''
        if not data:
            data = catin.mcquillan_with_stelparms()
        super().__init__(
            data, splitgroups=splitgroups, indices=indices, kic_col=kic_col, 
            tm_col=tm_col)

    def split_period(self, splitvalues, splitnames, pcol="Prot", 
                     period_crit="period", invert_inequality=False):
        '''Split the data by period.

        Since there are many different ways to measure period, the desired
        column should be given as col. The values to split about should be
        given in splitvalues, and the names of the classes should be given in
        splitnames. The type of period measurement used should be given in
        period_crit, for example McQuillan period. This will associate the
        splitnames group with the specific period measurement.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(pcol, splitvalues, splitnames, period_crit, 
                          invert_inequality=invert_inequality)

class GaiaSplitter(KeplerSplitter):
    '''Split dataset with Gaia information.'''

    def __init__(self, data, splitgroups=None, indices=None,
                 gaia_index="source_id", kic_col="kepid", tm_col="APOGEE_ID"):
        '''Initialize the splitter for a Gaia-containig dataset.

        Set up the splitter with the given data. The splitter is indexed by the
        gaia_index, which should be specified in the parameter "gaia_index".'''
        super().__init__(data, splitgroups=splitgroups, indices=indices,
                         kic_col=kic_col, tm_col=tm_col)
        self.gaia_index = gaia_index

    def split_parallax_quality(
            self, parallax_col="parallax", parallax_err_col="parallax_error", 
            frac_err=0.05, splitnames=(
                "Good parallax", "Bad parallax", "No parallax"), 
            crit="Gaia DR2 parallax"):
        '''Split the sample based on the quality of the parallax.

        The value and error of parallax must be given in parallax_col and
        parallax_err_col. Splitnames should be a 3-tuple containing the names
        of categories for those objects with good parallax, bad parallax, and
        no parallax.
        
        Only the targets with fractional parallax error will be accepted as
        having good parallax. Targets with fractional parallax error higher
        than frac_error, or those with negative parallaxes, will be removed.
        Objects with no Gaia parallaxes at all are categorized under the no
        parallax name.'''
        parallax_ratio = self.data[parallax_col] / self.data[parallax_err_col]

        no_parallax_indices = self.data[parallax_col].mask
        good_indices = np.logical_and(
            parallax_ratio >= 1/frac_err, np.logical_not(no_parallax_indices))
        bad_indices = np.logical_and(
            parallax_ratio < 1/frac_err, np.logical_not(no_parallax_indices))

        indexarr = [good_indices, bad_indices, no_parallax_indices]
        self._setup_indices(splitnames, indexarr, crit)


class APOGEESplitter(KeplerSplitter):
    '''Keep an organized database of various cuts on APOGEE data.'''

    def __init__(self, data=None, splitgroups=None, indices=None, 
                 kic_col="kepid", tm_col="APOGEE_ID"):
        '''Initialize a splitter of APOGEE data.
        
        If the data parameter is passed, then it will be set to the full data
        sample. If not, then it will be read in manually.'''
        if not data:
            data = catin.dr14_with_KIC_stelparms()
            data["ALPHA_FE"] = data["ALPHA_M"] + data["M_H"] - data["FE_H"]
        super().__init__(data, splitgroups=splitgroups, indices=indices, 
                         kic_col=kic_col, tm_col=tm_col)

    def split_vscatter(self, splitvalues, splitnames, col="VSCATTER",
                       vscatter_crit="VSCATTER", invert_inequality=True):
        '''Split the data by vscatter.

        Split the sample based on the boundaries given in splitvalues. The
        names for the categories should be given in splitnames. If the vscatter
        values are in a column other than VSCATTER, it can be specified with
        the col keyword. If there are other sources of vscatter, then they can
        be specified in vscatter_crit.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(col, splitvalues, splitnames, vscatter_crit, 
                          invert_inequality=invert_inequality)

    def split_vsini(self, splitvalues, splitnames, col="VSINI",
                    vsini_crit="VSINI", null_value=None, invert_inequality=False):
        '''Split the data by vsini.

        Split the sample based on the boundaries given in splitvalues. The
        names for the categories should be given in splitnames. If the vsini
        values are in a column other than VSINI, it can be specified with
        the col keyword. If there are other sources of vsini, then they can
        be labeled separately by vsini_crit.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(
            col, splitvalues, splitnames, vsini_crit, null_value=null_value,
            invert_inequality=invert_inequality)

    def split_metallicity(self, splitvalues, splitnames, col="M_H",
                          met_crit="Metallicity", null_value=None, invert_inequality=False):
        '''Split the data according to metallicity.

        Split the sample based on the boundaries given in splitvalues. The
        names for the categories should be given in splitnames. Those choice of
        metallicity can be specified with the col keyword.

        For more information on invert_inequality, see split_by_col.
        '''
        self.split_by_col(
            col, splitvalues, splitnames, met_crit, null_value=null_value, 
            invert_inequality=invert_inequality)


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
        self.data[tempcol] = rot.vsini_to_period(
            np.maximum(self.data[vsini_col], det_limit), 
            self.data[radius_col])[0]
        self.split_by_col(tempcol, splitperiods, splitnames, rapid_crit, 
                          invert_inequality=invert_inequality)
        del(self.data[tempcol])
        assert tempcol not in self.data.colnames

    def split_logg_evstate(
            self, teff_col="TEFF", logg_col="LOGG_FIT", 
            giant_subgiant_points=[(5000, 3.5), (3500, 3.5)],
            subgiant_dwarf_points=[(5690, 4.43), (4640, 3.72)], 
            splitnames=("Giant", "Subgiant", "Dwarf", "NO_EV"), 
            crit="APOGEE Evolutionary State"):
        '''Split the cool dwarf sample into evolutionary states using M_K.

        Split the catalog according to the K-band absolute magnitude.'''
        self.split_evstate(
            teff_col=teff_col, logg_col=logg_col,
            giant_subgiant_points=giant_subgiant_points,
            subgiant_dwarf_points=subgiant_dwarf_points, splitnames=splitnames,
            crit=crit)

    def split_dlsb(
        self, apid_col="APOGEE_ID", dl_names=("DLSB", "No DLSB", "Unknown DLSB"),
        dlsb_crit="DLSB", dlsb_db=paths.DLSB_DB, nodl_db=paths.NODL_DB):
        '''Split sample based on presence of double-lines.

        Split the sample based on previous observations of double-lined
        spectroscopic binaries. The matching is done via the APOGEE ID of the
        targets. Databases which contain APOGEE IDs of confirmed DLSBs and
        non-DLSBs are in dlsb_db and nodl_db.
        
        Labels for the confirmed DLSB, confirmed non-DLSB, and unconfirmed 
        classes should be given as a tuple in dl_names. Other analyses of 
        double-lined spectroscopic binaries can be specified by dlsb_crit.
        '''
        apids = self.data[apid_col]
        known_dlsbs = catalog.mark_DLSB_indices(
            apids, dlsb_db=dlsb_db)
        known_nondlsbs = catalog.mark_non_DLSB_indices(
            apids, nodl_db=nodl_db)
        unknown_dlsbs = np.logical_not(np.logical_or(
            known_dlsbs, known_nondlsbs))

        indexarr = [known_dlsbs, known_nondlsbs, unknown_dlsbs]
        self._setup_indices(dl_names, indexarr, dlsb_crit)

    def split_asteroseismic_dwarfs(
        self, splitnames=("Asteroseismic", "Non-asteroseismic"), 
        apid_col="APOGEE_ID", astero_crit="astero"):
        '''Separate asteroseismic dwarfs in dataset.

        Separates the asteroseismic dwarfs from the dataset. The asteroseismic
        dwarfs are currently fetched from the APOKASC catalog. Other
        determinations of the asteroseismic dwarfs can be specified in
        astero_crit.'''
        apokasc = catin.read_APOKASC_catalog()[["2MASS_ID", "RADIUS_DW"]]
        ast_dwarf = catalog.filter_invalid_APOGEE_entries(apokasc, "RADIUS_DW")
        astero_indices = au.mark_selections_in_columns(
            self.data[apid_col], ast_dwarf["2MASS_ID"])
        self._setup_complement_index(splitnames, astero_indices, astero_crit)

    def split_McQuillan_periods(
        self, mcq_names=("Mcq", "No Mcq", "Unknown Mcq"), kiccol="KIC", 
        mcq_crit="Mcq"):
        '''Separate detections, nondetections, and undetermined periods.

        Splits objects in three ways: those with McQuillan periods, those
        looked at by McQuillan, but not found to have a detected period, and
        those which haven't been looked at by McQuillan at all. The names for
        the three classes should be given in mcq_names. The two datasets are
        cross-matched by KIC numbers in kiccol.
        '''
        mcq = catin.read_McQuillan_catalog()
        undet = catin.read_McQuillan_nondetections()
        mcq_period = au.mark_selections_in_columns(
            self.data[kiccol], mcq["KIC"])
        mcq_noperiod = au.mark_selections_in_columns(
            self.data[kiccol], undet["KIC"])
        no_mcq = np.logical_not(np.logical_or(
            mcq_period, mcq_noperiod))
        indexarr = [mcq_period, mcq_noperiod, no_mcq]
        self._setup_indices(mcq_names, indexarr, mcq_crit)

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
        bad_indices = catalog.bad_ASPCAP_indices(
            flags, warn=False)
        badwarn_indices = catalog.bad_ASPCAP_indices(flags, warn=True)
        warn_indices = np.logical_and(
            np.logical_not(bad_indices), badwarn_indices)
        vsini_indices = np.logical_and(
            np.logical_not(badwarn_indices), catalog.warn_VSINI_indices(flags))
        good_indices = np.logical_not(np.logical_or(
            badwarn_indices, vsini_indices))
        indexarr = [bad_indices, warn_indices, vsini_indices, good_indices]
        self._setup_indices(qual_names, indexarr, aspcap_crit)

    def split_targeting(
        self, target_label, splitnames=None, target_crit=None):
        '''Select the objects which fall under the targeting flag.

        The target_label should be the label passed to catalog.target_indices.
        Then the indices corresponding to that targeting flag will be placed.
        If a 2-tuple is provided for names, then that will be the label for the
        objects with the targeting flag, and without the targeting flag,
        respectively. Similarly, target_crit can be specified.

        If names or target_crit isn't specified, then they will use the
        targeting flag given. For example, if the targeting flag was
        APOGEE_KEPLER_COOLDWARF, names would be ("APOGEE_KEPLER_COOLDWARF",
        "Not APOGEE_KEPLER_COOLDWARF") and target_crit would just be
        "APOGEE_KEPLER_COOLDWARF".'''

        indices = catalog.target_indices(self.data, target_label)
        if not splitnames:
            splitnames = (target_label, "Not " + target_label)
        if not target_crit:
            target_crit = target_label

        self._setup_complement_index(splitnames, indices, target_crit)

    def split_combined_targeting(
            self, target_list, splitnames, target_crit):
        '''Select objects which fall under any of multiple targeting flags.

        Split off targets which fall in any of the targeting flags contained in
        target_list. Splitnames should be a 2-tuple which contains the names
        for the sample falling under the targeting regime, and the sample not.
        Additionally, target_crit should be a unique explanatory identifier for
        this type of targeting split.
        '''
        indexlist = [
            catalog.target_indices(self.data, targ) for targ in target_list]
        indices = au.multi_logical_or(*indexlist)

        self._setup_complement_index(splitnames, indices, target_crit)
    def split_eclipsing_binaries(
            self, kiccol="kepid", splitnames=("Kepler EB", "Not EB"), 
            eb_crit="Eclipsing Binaries"):
        '''Split off objects which were classified as eclipsing binaries.
        
        The label for objects that were classified as EBs should be the first
        element of splitnames, and the label for objects not EBs should be the
        second one.'''
        eb_indices = ebs.EB_indices(self.data[kiccol])
        self._setup_complement_index(splitnames, eb_indices, eb_crit)

    def split_KOIs(
            self, kiccol="kepid", splitnames=("KOI", "Not KOI"),
            koi_crit="KOIs"):
        '''Split off objects which were classified as KOIs.

        The label for objects classified as KOIs should be the first element of
        splitnames, and the label for objects not KOIs should be the second
        one.'''
        koi_indices = catalog.KOI_indices(self.data[kiccol])
        self._setup_complement_index(splitnames, koi_indices, koi_crit)

    def split_cool_dwarfs(
            self, splitnames=("Cool Sample", "Not Cool Sample"),
            cool_crit="Cool Dwarfs", apogee1_flag="APOGEE_KEPLER_COOLDWARF", 
            apogee2_flag="APOGEE2_APOKASC_DWARF", dwarf_flag="Jen Dwarf",
            hlim_flag="H APOGEE", sdss_cool_flag="Jen Cool", 
            no_sdss_flag="No SDSS Teff", kic_cool_flag="KIC Jen Cool"):
        '''Split off the cool dwarf sample from the rest of the apogee sample.

        This split relies on the following splits to already have occurred on
        this dataset:
        self.split_targeting() for apogee1_flag
        self.split_targeting() for apogee2_flag
        self.split_logg() for dwarf_flag
        self.split_mag() for hlim_flag
        self.split_teff() for sdss_cool_flag and no_sdss_flag
        self.split_teff() for kic_cool_flag

        The first two targeting flags automatically take the
        APOGEE_KEPLER_COOLDWARF targets, and set the APOGEE2_APOKASC_DWARF
        targets as the sample for targets observed in APOGEE2. The dwarf_flag
        should correspond to targets with original KIC log(g) > 4.0. The
        hlim_flag should pick out targets with 7 < H < 11 mag. The
        sdss_cool_flag should pick out targets with Pinsonneault et al (2012)
        SDSS Teff < 5500 K, and the no_sdss_flag should mark targets which
        don't have SDSS Teff values. For those cases, targets with
        kic_cool_flag values, which should be KIC Teff < 5500 K should be
        used.'''
        orig_targets = self.indices[apogee1_flag]
        apogee2_sdss = au.multi_logical_and(
            self.indices[apogee2_flag], self.indices[dwarf_flag],
            self.indices[hlim_flag], self.indices[sdss_cool_flag])
        apogee2_nosdss = au.multi_logical_and(
            self.indices[apogee2_flag], self.indices[dwarf_flag],
            self.indices[hlim_flag], self.indices[no_sdss_flag],
            self.indices[kic_cool_flag])
        full_sample = au.multi_logical_or(
            orig_targets, apogee2_sdss, apogee2_nosdss)

        self._setup_complement_index(splitnames, full_sample, cool_crit)

    def split_modified_Berger_EVstate(
            self, teff_col="TEFF", feh_col="FE_H", alpha_col="ALPHA_FE", 
            MK_col="M_K", class_col="class", cool_limit=5500, splitnames=(
                "Berger Giant", "Berger Subgiant", 
                "Modified Berger Main Sequence", "Modified Berger Cool Binary",
            "No Berger Classification"), 
            crit="Modified Berger Evolutionary State"):
        '''Split sample according to the evolutionary state in Berger (2018).

        Berger et al (2018) classified objects in the Teff-Luminosity plane as
        being giants, subgiants, main sequence dwarfs, and binaries on the cool
        end, where the main sequence splits from subgiants.
        
        This scheme modifies the original Berger classification by performing a
        color-dependent cut on the K-band magnitude excess '''
        fulldwarf_indices = np.logical_or(
            self.data[class_col] == 0, self.data[class_col] == 3)
        subgiant_indices = self.data[class_col] == 1
        giant_indices = self.data[class_col] == 2
        missing_indices = self.data[class_col].mask

        # We can only classify photometric binaries cooler than cool_limit.
        # Also classify bad objects according to the original Berger
        # classification.
        cool_dwarf_indices = np.logical_and(
            fulldwarf_indices, np.logical_and(
                self.data[teff_col] > 0, self.data[teff_col] < 5500))

        magdiff = samp.calc_photometric_excess(
            self.data[teff_col][cool_dwarf_indices],
            self.data[feh_col][cool_dwarf_indices], 
            self.data[alpha_col][cool_dwarf_indices], "Ks",
            self.data[MK_col][cool_dwarf_indices], age=3)
        phot_binary_div_points = [(5427, -0.60), (3946, -0.14)]
        dividing_line = (phot_binary_div_points[0][1] + 
            (phot_binary_div_points[0][1] - phot_binary_div_points[1][1]) /
            (phot_binary_div_points[0][0] - phot_binary_div_points[1][0]) *
            (self.data[teff_col][cool_dwarf_indices] - 
             phot_binary_div_points[0][0]))
        phot_binary_indices = magdiff < dividing_line

        dwarf_indices = np.zeros(len(cool_dwarf_indices))
        binary_indices = np.zeros(len(cool_dwarf_indices))
        dwarf_indices[cool_dwarf_indices] = ~phot_binary_indices
        binary_indices[cool_dwarf_indices] = phot_binary_indices
        # Add in objects with bad APOGEE fits
        dwarf_indices = np.logical_or(dwarf_indices, np.logical_and(
            self.data[teff_col] <= 0, self.data[class_col] == 0))
        binary_indices = np.logical_or(binary_indices, np.logical_and(
            self.data[teff_col] <= 0, self.data[class_col] == 3))
        # Add back hot dwarfs
        dwarf_indices = np.logical_or(dwarf_indices, np.logical_and(
            fulldwarf_indices, self.data[teff_col] > 5500))
        indexarr = [giant_indices, subgiant_indices, dwarf_indices,
                    binary_indices, missing_indices]
        self._setup_indices(splitnames, indexarr, crit)


    def subsample_len(self, namelist):
        '''Get the size of a subsample.

        This function automatically gets the size of a subsample in a fast way
        as opposed to running subsample() and getting its length.
        
        APOGEE data specifically can have duplicate entries because of the
        targeting. This method will make sure to only return number of unique
        entries in the subsample.'''
        indices = self._subsample_indices(namelist)
        names = self.data[self.tm_col][indices]
        indexlen = len(np.unique(names))
        return indexlen

class APOKASCSplitter(APOGEESplitter):
    '''A splitter for the APOKASC dataset.'''

    def __init__(self, data=None, splitgroups=None, indices=None, 
                 kic_col="KEPLER_INT", tm_col="2MASS_ID"):
        '''Initialize a splitter of APOKASC data.
        
        If the data parameter is passed, then it will be set to the full data
        sample. If not, then it will be read in manually.'''
        if not data:
            data = catin.APOKASC_with_KIC_stelparms()
        super().__init__(data, splitgroups=splitgroups, indices=indices, 
                         kic_col=kic_col, tm_col=tm_col)

    def split_asteroseismic_dwarfs(
        self, dwarfcol="RADIUS_DW", splitnames=(
            "Asteroseismic Giants", "Asteroseismic Dwarfs"), 
        apodwarf_crit="Asteroseismic Dwarf"):
        '''Split stars with dwarf pipeline asteroseismic parameters.

        This function will essentially split off targets that don't have valid
        values in the dwarfcol column. The dwarfcol column should have numbers
        for objects that have been run through the dwarf pipeline, and null for
        objects that don't.'''
        invalid = catalog.invalid_indices(self.data, dwarfcol)
        self._setup_complement_index(splitnames, invalid, apodwarf_crit)

    def split_Jen_targets(
            self, jencol="VANSADERS", 
            splitnames=("Jen Targets", "Not Jen Targets"), jen_crit="Jen"):
        '''Split stars which were in Jen's targeting list.

        Targets which have "T" in jencol are considered as being in Jen's list. If
        they weren't, then it should have an "F". The names given to Jen's targets
        should be given in splitnames[0] and those that aren't her targets should
        be splitnames[1].'''
        jentargs = self.data[jencol] == "T"
        notjentargs = self.data[jencol] == "F"

        indexarr = [jentargs, notjentargs]
        self._setup_indices(splitnames, indexarr, jen_crit)

    def split_targeting(
        self, target_label, splitnames=None, target_crit=None,
            aspcapcol="TARGFLAGS"):
        '''Select the objects which fall under the targeting flag.

        The target_label should be the label passed to catalog.target_indices.
        Then the indices corresponding to that targeting flag will be placed.
        If a 2-tuple is provided for names, then that will be the label for the
        objects with the targeting flag, and without the targeting flag,
        respectively. Similarly, target_crit can be specified.

        If names or target_crit isn't specified, then they will use the
        targeting flag given. For example, if the targeting flag was
        APOGEE_KEPLER_COOLDWARF, names would be ("APOGEE_KEPLER_COOLDWARF",
        "Not APOGEE_KEPLER_COOLDWARF") and target_crit would just be
        "APOGEE_KEPLER_COOLDWARF".'''
        strcol = au.byte_to_unicode_cast(self.data[aspcapcol])
        indices = npstr.find(strcol, target_label) >= 0
        if not splitnames:
            splitnames = (target_label, "Not " + target_label)
        if not target_crit:
            target_crit = target_label

        self._setup_complement_index(splitnames, indices, target_crit)

    def subsample_len(self, namelist):
        '''Get the size of a subsample.

        This function automatically gets the size of a subsample in a fast way
        as opposed to running subsample() and getting its length.
        
        APOGEE data specifically can have duplicate entries because of the
        targeting. This method will make sure to only return number of unique
        entries in the subsample.'''
        indices = self._subsample_indices(namelist)
        names = self.data[self.tm_col][indices]
        indexlen = len(np.unique(names))
        return indexlen

def create_combined_rotation_splitter(baseclass):
    '''Create a custom CombinedRotationSplitter.
    
    Return a class which inherits from both the given base class and a
    McQuillanSplitter.'''

    class CombinedRotationSplitter(baseclass,McQuillanSplitter):
        '''A splitter for a dataset containing both APOGEE and McQuillan data.

        This splitter will be useful for unifying parts of the Kepler sample which
        overlap when observed with McQuillan and APOGEE.'''
        def __init__(self, data=None, splitgroups=None, indices=None, 
                     kic_col="kepid", tm_col="tm_designation"):
            '''A class for overlapping spectroscopic and photometric data.

            These objects ought to have both vsinis and rotational periods.'''
            if not data:
                data = catin.dr14_with_KIC_stelparms()
            super().__init__(data, splitgroups=splitgroups, indices=indices, 
                             kic_col=kic_col, tm_col=tm_col)

    return CombinedRotationSplitter

################################################################################
# Initialize Splitters #
###############################################################################

def initialize_clean_APOGEE(aposplit):
    '''Initialize a the full APOGEE Sample.

    Because very little will be done with the full APOGEE sample, this will
    be a pretty rough cut.'''
    aposplit.split_by_ASPCAP_flags()

    aposplit.split_teff(
        "TEFF", [5000], ("Cool Noev", "Hot HighEv", "Bad APOGEE Teff"),
        teff_crit="Teff Evolution", null_value=np.ma.masked)

    aposplit.split_teff(
        "TEFF", [4000, 5000], (
            "APOGEE MetCor Cool", "APOGEE MetCor Teff", "APOGEE MetCor Hot", 
            "No APOGEE MetCor"), null_value=np.ma.masked,
        teff_crit="APOGEE Metallicity Correction Region")

    aposplit.split_teff(
        "TEFF", [4000, 5250], (
            "APOGEE Statistics Cool", "APOGEE Statistics Teff", 
            "APOGEE Statistics Hot", "No APOGEE Statistics"), 
        null_value=np.ma.masked, teff_crit="APOGEE Statistics Region")

    aposplit.split_teff(
        "teff", [4000, 5000], (
            "Huber MetCor Cool", "Huber MetCor Teff", "Huber MetCor Hot", 
            "No Huber MetCor"), null_value=np.ma.masked,
        teff_crit="Huber Metallicity Correction Region")
    aposplit.split_teff(
        "SDSS-Teff", [4000, 5000], (
            "Pinsonneault MetCor Cool", "Pinsonneault MetCor Teff", 
            "Pinsonneault MetCor Hot", "No Pinsonneault MetCor"), 
        null_value=np.ma.masked, 
        teff_crit="Pinsonneault Metallicity Correction Region")

    aposplit.split_vsini(
        [7], ("Vsini nondet", "Vsini det", "No Vsini"),
        null_value=np.ma.masked)

    aposplit.split_metallicity(
        -0.5, ("Low Met", "High Met", "No Met"), col="FE_H", 
        null_value=np.ma.masked)

    aposplit.split_dlsb()

    aposplit.split_photometric_quality(
        "kmag", "kmag_err", splitnames=("K Detection", "Blend", "Bad K"), 
        crit="MK blend")

    aposplit.split_Gaia()


def initialize_full_APOGEE(aposplit):
    '''Get the underlying APOGEE sample we're interested in.'''
    aposplit.split_targeting("APOGEE_KEPLER_COOLDWARF")
    aposplit.split_targeting("APOGEE2_APOKASC_DWARF")
    aposplit.split_targeting("APOGEE2_APOKASC_GIANT")
    aposplit.split_targeting("APOGEE2_APOKASC")
    aposplit.split_targeting("APOGEE_KEPLER_EB")
    aposplit.split_targeting("APOGEE2_KOI")
    aposplit.split_targeting("APOGEE2_KOI_CONTROL")
    aposplit.split_targeting("APOGEE_KEPLER_SEISMO")
    aposplit.split_targeting("APOGEE_RV_MONITOR_KEPLER")
    aposplit.split_targeting("APOGEE2_EB")
    aposplit.split_targeting("APOGEE_KEPLER_HOST")

    aposplit.split_mag(
        "H", [7, 11], ("H Bright", "H APOGEE", "H Faint", "No H"), mag_crit="H",
        null_value=np.ma.masked)
    aposplit.split_combined_targeting(
        ["APOGEE_KEPLER_COOLDWARF", "APOGEE2_APOKASC", "APOGEE_KEPLER_EB",
         "APOGEE2_KOI", "APOGEE2_KOI_CONTROL", "APOGEE_KEPLER_SEISMO",
         "APOGEE_RV_MONITOR_KEPLER", "APOGEE2_EB", "APOGEE_KEPLER_HOST"],
        ("Targeted", "Not Targeted"), "Targeting")

    aposplit.split_McQuillan_periods(kiccol="kepid")

def general_to_hot_kic_sample(apogeesplitter):
    '''Get the subset of the hot sample that has KIC parameters.'''
    hot_kic = apogeesplitter.split_subsample([
        "APOGEE2_APOKASC_DWARF", "Orig KIC Present"])
    return hot_kic

def initialize_general_APOGEE(aposplit):
    '''Initialize the most general and applicable cuts to APOGEE'''
    aposplit.split_teff(
        "TEFF", [5500], ("Cool", "Hot"), teff_crit="APOGEE Teff")
    
    aposplit.split_mag(
        "H", [7, 11], ("H Bright", "H Jen", "H Faint"), mag_crit="H")

    aposplit.split_targeting("APOGEE_KEPLER_COOLDWARF")
    aposplit.split_targeting("APOGEE2_APOKASC_DWARF")
    aposplit.split_by_ASPCAP_flags()

    aposplit.split_original_KIC_params()

    aposplit.split_McQuillan_periods(kiccol=aposplit.kic_col)

    aposplit.split_vsini(
        [0, 7, 10], ("No Vsini", "Vsini nondet", "Vsini marginal", "Vsini det"))

    aposplit.split_vscatter(
        [0, 1], ("Single Visit", "RV Nonvariable", "RV Variable"), 
        invert_inequality=True)

    aposplit.split_dlsb()

    aposplit.split_photometric_quality(
        "K_ERR", splitnames=("Good K", "Blend"), crit="MK blend")

    aposplit.split_modified_Berger_EVstate()

def initialize_cool_KICs(kicsplit):
    '''Initialize cool dwarfs that have KIC values.'''
    kicsplit.split_mk_evstate(crit="Subgiant Split")

def initialize_asteroseismic_sample(aposplit):
    '''Initialize the sample for asteroseismic targets.
    
    This will set aside the asteroseismic dwarfs from the rest of the sample.'''
    # First split the asteroseismic targets
    aposplit.split_asteroseismic_dwarfs()

    # Split between hot and cool
    aposplit.split_teff(
        "TEFF_COR", 5500, ("Cool", "Hot", "No APOGEE Teff"), 
        teff_crit="APOGEE Teff", null_value=np.ma.masked)

    # Now split the spectroscopic targets
    aposplit.split_logg("LOGG_FIT", [3.5, 4.0], (
        "Spectroscopic dwarfs", "Spectroscopic subgiants", 
        "Spectroscopic giants", "No APOGEE logg"), logg_crit="APOGEE logg",
                        null_value=np.ma.masked)

    # Split  by quality.
    aposplit.split_by_ASPCAP_flags()

    # Split by vsini
    aposplit.split_vsini(
        [7], ["Vsini nondet", "Vsini det", "No vsini"],
        null_value=np.ma.masked)

    # Split by DLSB presence.
    aposplit.split_dlsb(apid_col="2MASS_ID")

    # Split by McQuillan Periods
    aposplit.split_McQuillan_periods(kiccol=aposplit.kic_col)

    aposplit.split_photometric_quality(
        "K_MAG_2M", "K_MAG_ERR", splitnames=("Good K", "Blend", "Bad Phot"), 
        crit="MK blend")

    # Absolute K-band magnitude
    aposplit.data["M_K"] = (
        aposplit.data["K_MAG_2M"] - 5 * np.log10(aposplit.data["dis"]/10))
    aposplit.data["M_K_err1"] = np.where(
        aposplit.data["K_MAG_ERR"] > 0, aposplit.data["K_MAG_ERR"]**2 + (
        5 * (aposplit.data["disem"]) / aposplit.data["dis"] / np.log(10))**2,
        np.ma.masked)
    aposplit.data["M_K_err2"] = np.where(
        aposplit.data["K_MAG_ERR"] > 0, aposplit.data["K_MAG_ERR"]**2 + (
        5 * (aposplit.data["disep"]) / aposplit.data["dis"] / np.log(10))**2,
        np.ma.masked)


def initialize_mcquillan_sample(mcqsplit):
    '''Makes a series of cuts related to the rotation period of the targets.'''
    mcqsplit.split_teff(
        "SDSS-Teff", [4000, 5000], (
            "Too Cool MetCor", "Right MetCor Teff", "Too Hot MetCor", 
            "No Pinsonneault Teff"),
        teff_crit="Metallicity Correction", null_value=np.ma.masked)
    mcqsplit.split_teff(
        "SDSS-Teff", [4000, 5000], (
            "Too Cool Statistics", "Right Statistics Teff", 
            "Too Hot Statistics", "No Statistics Teff"), 
        teff_crit="Statistics", null_value=np.ma.masked)
    mcqsplit.split_teff(
        "SDSS-Teff", 7000, splitnames=(
            "Good Isochrone Teff", "Too Hot for Isochrone", 
            "Bad Isochrone Teff"), teff_crit="Isochrone Temperature", 
        null_value=np.ma.masked)
    mcqsplit.split_photometric_quality(
        "kmag", "kmag_err", splitnames=("K Detection", "Blend", "Bad K"), 
        crit="MK blend")

    mcqsplit.split_Gaia()

    
    mcqsplit.split_period([1, 3], ["Too rapid", "Rapid", "Slow"])

def initialize_asteroseismic_periods(aposplit):
    '''Initialize the asteroseismic sample with McQuillan periods.'''
    initialize_asteroseismic_sample(aposplit)
    initialize_mcquillan_sample(aposplit)

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

###############################################################################
# Join period table to DataSplitter #
###############################################################################

def add_periods_to_datasplitter(split, kic_col="KIC"):
    '''Add the McQuillan periods to the given splitter.
    
    This will return a brand new splitter which inherits from the original
    splitter as well as the McQuillanSplitter.'''

    mcq = catin.read_McQuillan_catalog()
    catin.trim_McQuillan_catalog(mcq)
    newdata = au.join_by_id(split.data, mcq, kic_col, "KIC")
    NewRotationClass = create_combined_rotation_splitter(split.__class__)
    combosplit = NewRotationClass(
        data=newdata, kic_col=kic_col, tm_col=split.tm_col)
    return combosplit



################################################################################
# Datasplitter Functions #
################################################################################



def binned_vsini_dist(aposplit, bingroup="Huber Bins", defparams=[
    "Huber dwarf"], normed=False):
    '''Plot cumulative histogram of the vsini distribution for teff bins.
    
    This will plot the empirical distribution function of vsini for each Teff
    bin. It will distinguish between just the good sample and the good/warn
    samples.'''
    for binlabel in aposplit.splitgroups[bingroup]:
        plt.figure()
        # Want to exclude DLSBs, so read in the No and Unknown DLSBs.
        # Also want to exclude bad vsinis. But keep track of how many there
        # are.
        good_nodlsb_highvsini = aposplit.subsample(
            defparams + ["Good", binlabel, "No DLSB", "Vsini det"])
        good_nodlsb_lowvsini = aposplit.subsample(
            defparams + ["Good", binlabel, "No DLSB", "Vsini nondet"])
        good_nodlsb_novsini_len = aposplit.subsample_len(
            defparams + ["Good", binlabel, "No DLSB", "No Vsini"])
        good_udlsb_highvsini = aposplit.subsample(
            defparams + ["Good", binlabel, "Unknown DLSB", "Vsini det"])
        good_udlsb_lowvsini = aposplit.subsample(
            defparams + ["Good", binlabel, "Unknown DLSB", "Vsini nondet"])
        good_udlsb_novsini_len = aposplit.subsample_len(
            defparams + ["Good", binlabel, "Unknown DLSB", "No Vsini"])
        good_vsinis = np.ma.concatenate(
            [good_nodlsb_highvsini["VSINI"], good_nodlsb_lowvsini["VSINI"], 
             good_udlsb_highvsini["VSINI"], good_udlsb_lowvsini["VSINI"]])
        good_novsini_len = good_nodlsb_novsini_len + good_udlsb_novsini_len

        warn_nodlsb_highvsini = aposplit.subsample(
            defparams + ["Warn", binlabel, "No DLSB", "Vsini det"])
        warn_nodlsb_lowvsini = aposplit.subsample(
            defparams + ["Warn", binlabel, "No DLSB", "Vsini nondet"])
        warn_nodlsb_novsini_len = aposplit.subsample_len(
            defparams + ["Warn", binlabel, "No DLSB", "No Vsini"])
        warn_udlsb_highvsini = aposplit.subsample(
            defparams + ["Warn", binlabel, "Unknown DLSB", "Vsini det"])
        warn_udlsb_lowvsini = aposplit.subsample(
            defparams + ["Warn", binlabel, "Unknown DLSB", "Vsini nondet"])
        warn_udlsb_novsini_len = aposplit.subsample_len(
            defparams + ["Warn", binlabel, "Unknown DLSB", "No Vsini"])
        goodwarn_vsinis = np.ma.concatenate(
            [warn_nodlsb_highvsini["VSINI"], warn_nodlsb_lowvsini["VSINI"], 
             warn_udlsb_highvsini["VSINI"], warn_udlsb_lowvsini["VSINI"], 
             good_vsinis])
        goodwarn_novsini_len = (
            warn_nodlsb_novsini_len + warn_udlsb_novsini_len + good_novsini_len)

        sorted_good_vsinis = np.sort(good_vsinis)
        sorted_goodwarn_vsinis = np.sort(goodwarn_vsinis)

        good_stepx = np.ma.concatenate(
            [sorted_good_vsinis, sorted_good_vsinis[[-1]]])
        goodwarn_stepx = np.ma.concatenate(
            [sorted_goodwarn_vsinis, sorted_goodwarn_vsinis[[-1]]])
        if normed:
            good_stepy = np.linspace(0, 1, len(sorted_good_vsinis)+1)
            goodwarn_stepy = np.linspace(0, 1, len(sorted_goodwarn_vsinis)+1)
        else:
            good_stepy = np.arange(len(sorted_good_vsinis)+1)
            goodwarn_stepy = np.arange(len(sorted_goodwarn_vsinis)+1)

        goodline = plt.step(good_stepx, good_stepy, label="Good")
        goodwarnline = plt.step(goodwarn_stepx, goodwarn_stepy, 
                                label="Good+Warn")
        ax = plt.gca()
        plt.text(0.79, 0.2, "{0:d} missing vsini".format(good_novsini_len),
                 horizontalalignment="center", verticalalignment="center",
                 transform=ax.transAxes, color=goodline[0].get_color())
        plt.text(0.79, 0.25, "{0:d} missing vsini".format(goodwarn_novsini_len), 
                 horizontalalignment="center", verticalalignment="center",
                 transform=ax.transAxes, color=goodwarnline[0].get_color())
        plt.xlabel("Vsini")
        plt.ylabel("N(<vsini)")
        plt.title("{0} CDF".format(binlabel))
        plt.xlim(0, 70)

        plt.legend(loc="lower right")

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

################################################################################
# Detection limit table #
################################################################################

def rapid_rotator_det_test_split(
    aposplit, det_limits=range(7, 11), det_crit_format="Det limit {0:d}",
    det_format="Det >{0:d}", nondet_format="Nondet <{0:d}", 
    rapid_crit_format="Spec Rapid Det={0:d}", 
    very_rapid_format="Very Rapid Det={0:d}", rapid_format="Rapid Det={0:d}",
    slow_format="Slow Det={0:d}"):
    '''Create categories for getting rapid rotator fractions.

    The limits to be probed should be given as an iterable in det_limits. The
    default are [7, 8, 9, 10] km/s. For each limit, the dataset will be split
    into detections and nondetections, labeled as det_format and nondet_format.
    The detections will be under the group name det_crit_format. Additionally,
    the classifications will be made under the group name rapid_crit_format.
    The very rapid, rapid, and slow rotators will be accessible under the
    labels very_rapid_format, rapid_format, and slow_format.

    Note that all of the keywords ending in _format will have the detection
    limit value in det_limits passed to it via the format() method. This is
    because the current DataSplitter implementation requires unique category
    names.'''

    for i in det_limits:
        splitnames = [rotclass.format(i) for rotclass in [
            very_rapid_format, rapid_format, slow_format]]
        aposplit.split_vsini(
            i, [nondet_format.format(i), det_format.format(i)],
            vsini_crit=det_format.format(i))
        aposplit.split_spectroscopic_rapid_rotators(
            [1, 5], splitnames, det_limit=i, 
            rapid_crit=rapid_crit_format.format(i))

def rapid_rotator_det_test(
    aposplit, det_limits=range(7, 11), det_format="Det >{0:d}",
    rapid_format="Rapid Det={0:d}", othercrit=["Huber dwarf"],
    tempgroup="Huber Teff"):
    '''Determine the rapid rotator fraction as function of detection limit.

    In each Teff bin, calculate the rapid rotator fraction using the given
    vsini detection limits. This assumes that the necessary categories were
    already made by rapid_rotator_det_test_split().'''
    for teff in aposplit.splitgroups[tempgroup]:
        for i in det_limits:
             num, denom = rapid_rotator_fraction(
                aposplit, rrcrit=rapid_format.format(i),
                 detcrit=det_format.format(i), othercrit=othercrit+[teff])
             print(("For {0} and vsini>={1:d}, {2:d} out of {3:d} ({4:d}%) "
                    "are rapid.").format(
                        teff, i, num, denom, num*100//denom))

###############################################################################
# APOGEE Radii #
###############################################################################

def add_radius_column_to_splitter(
    split, teff_rad, teff_col="TEFF", radius_col="APOGEE RADIUS"):
    '''Add a radius column to the APOGEESplitter.
    
    The radius is derived by fitting the Huber radii of dwarfs to a cubic 
    function over Teff. Note that this does not take any scatter due to
    metallicity into account.'''
    rad_fit = samp.huber_dwarf_radius_relation()
    samp.generate_radius_column(
        split.data, rad_fit, teffcol="TEFF", radcol="APOGEE MS radius")

    
################################################################################
# Making Narrow-purpose datasplitters #
################################################################################

def jen_cool_splitter():
    '''Create a Datasplitter consisting only of Jen's cool dwarf sample.
    
    These are mostly targets with the APOGEE_KEPLER_COOLDWARF and around 100 of
    the APOGEE2_APOKASC_DWARF targets. These targets mostly have Teff < 5500
    and H < 11.'''
    
    fullapogee = APOGEESplitter()
    initialize_general_APOGEE(fullapogee)
    kicsplit = fullapogee.split_subsample(["Orig KIC Present"])
    initialize_cool_KICs(kicsplit)
    return kicsplit.split_subsample(["Cool Sample"])

def general_to_cool_sample(apogeesplitter):
    '''Separate the cool sample from a generalized APOGEE splitter.'''
    kicsplit = apogeesplitter.split_subsample(["Orig KIC Present"])
    initialize_cool_KICs(kicsplit)
    cooldwarfs = kicsplit.split_subsample(["Cool Sample"])
    return cooldwarfs

def general_to_hot_kic_sample(apogeesplitter):
    '''Get the subset of the hot sample that has KIC parameters.'''
    hot_kic = apogeesplitter.split_subsample([
        "APOGEE2_APOKASC_DWARF", "Orig KIC Present"])
    return hot_kic

def general_to_hot_nonkic_sample(apogeesplitter):
    '''Get the subset of the hot sample without KIC parameters.'''
    hot_kic = apogeesplitter.split_subsample([
        "APOGEE2_APOKASC_DWARF", "Orig KIC Not Present"])
    return hot_kic

def jen_cool_apodwarf_splitter():
    '''Create a Datasplitter with the good dwarf subset of Jen's sample.

    From within Jen's cool dwarf sample, create a datasplitter which constains
    the subset with APOGEE parameters log(g) < 4.0 and Teff > 4200 K. This
    dataset will also have radii calibrated from the Huber parameters, but
    using the APOGEE Teffs. Possibly even the metallicities at some point.

    Because the radii are present, this sample will also have the distinction
    between slow and rapid rotators.'''
    jensplit = jen_cool_splitter()
    # Calibrate the relationship between Teff and radius.
    radius_calib = jensplit.subsample(["Good Teff", "Huber dwarf"])
    radius_fitter = samp.fit_teff_radius_relation(
        radius_calib["teff"], radius_calib["radius"])
    
    jen_dwarfs = jensplit.subsample(["APOGEE dwarf", "Good Teff"])
    samp.generate_radius_column(jen_dwarfs, radius_fitter)
    dwarfsplitter = APOGEESplitter(jen_dwarfs)
    initialize_apogee_dwarf_rotation_sample(dwarfsplitter)
    return dwarfsplitter

def combo_from_APOGEE_Splitter(aposplit):
    '''Add the McQuillan periods to an existing APOGEE splitter.

    This function will take the existing APOGEE splitter, and then take the
    subset with McQuillan periods to yield a DataSplitter with McQuillan
    information.'''
    targs_with_mcq = aposplit.subsample(["Mcq"])
    mcq = catin.read_McQuillan_catalog().copy()
    mcq.remove_columns(["Teff", "log_g_", "Mass", "_RA", "_DE", "Ref"])
    combined_table = au.join_by_id(targs_with_mcq, mcq, "kepid", "KIC")
    assert len(targs_with_mcq) == len(combined_table)

    combosplitter = CombinedRotationSplitter(combined_table)
    return combosplitter
