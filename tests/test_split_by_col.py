import numpy as np

from astropy.table import Table
import pytest

import catalog
from APOGEE_spectroscopy import DataSplitter

tablestring = '''
|  a  |  b  |  c  |
| 12.0| 1.2 | 1.6 |
| 4.6 | 2.4 | 8.9 |
| 2.3 | 4.5 | 5.8 |
| 1.4 | 6.0 | 9.3 |
| 0.3 | 6.6 | 1.3 |
'''

class CustomSplitter(DataSplitter):
    '''Simple splitter for testing purposes'''
    def split_a(self, splitvalues, splitnames, col="a", a_crit="a",
                invert_inequality=False):
        '''Split by a'''
        self.split_by_col(col, splitvalues, splitnames, invert_inequality)
        self.splitgroups[a_crit] = set(splitnames)
    def split_b(self, splitvalues, splitnames, col="b", b_crit="b",
                invert_inequality=False):
        '''Split by b'''
        self.split_by_col(col, splitvalues, splitnames, invert_inequality)
        self.splitgroups[b_crit] = set(splitnames)
    def split_c(self, splitvalues, splitnames, col="c", c_crit="c",
                invert_inequality=False):
        '''Split by c'''
        self.split_by_col(col, splitvalues, splitnames, invert_inequality)
        self.splitgroups[c_crit] = set(splitnames)

@pytest.fixture
def typical_splitter():
    '''Generate a general table.
    
    This has a straightforward split in column a. And a boundary split in
    column b.'''
    a = Table.read(tablestring, format="ascii.fixed_width")
    b = CustomSplitter(a)
    b.split_a([3], ["low_a", "high_a"])
    b.split_b([6.0], ["low_b", "high_b"])
    b.split_b([6.0], ["inv_low_b", "inv_high_b"], invert_inequality=True)
    b.split_c([1, 6], ["low_c", "mid_c", "high_c"])
    return b

def test_single_split(typical_splitter):
    '''Split the table according to one value with default keywords.'''
    typical_splitter.split_a([3], ["low_a", "high_a"])
    assert np.all(
        typical_splitter.indices["low_a"] == [False, False, True, True, True])
    assert np.all(
        typical_splitter.indices["high_a"] == [True, True, False, False,
                                               False])
    assert typical_splitter.splitgroups["a"] == {"low_a", "high_a"}

def test_single_split_without_list(typical_splitter):
    '''Split the table according to one value given without a list.'''
    typical_splitter.split_a(3, ["low_a", "high_a"])
    assert np.all(
        typical_splitter.indices["low_a"] == [False, False, True, True, True])
    assert np.all(
        typical_splitter.indices["high_a"] == [True, True, False, False,
                                               False])
    assert typical_splitter.splitgroups["a"] == {"low_a", "high_a"}

def test_split_outside_range(typical_splitter):
    '''Test a split that occurs outside of column range.'''
    typical_splitter.split_a([-1], ["low_a", "high_a"])
    assert np.all(
        typical_splitter.indices["low_a"] == [
            False, False, False, False, False])
    assert np.all(
        typical_splitter.indices["high_a"] == [True, True, True, True, True])
    assert typical_splitter.splitgroups["a"] == {"low_a", "high_a"}

def test_multiple_split(typical_splitter):
    '''Split the table according to multiple values.'''
    typical_splitter.split_c([2, 6], ["low_c", "mid_c", "high_c"])
    assert np.all(
        typical_splitter.indices["low_c"] == [True, False, False, False, True])
    assert np.all(
        typical_splitter.indices["mid_c"] == [False, False, True, False,
                                              False])
    assert np.all(
        typical_splitter.indices["high_c"] == [False, True, False, True,
                                               False])
    assert typical_splitter.splitgroups["c"] == {"low_c", "mid_c", "high_c"}

def test_custom_group_name(typical_splitter):
    '''Split the table with a custom group name'''
    typical_splitter.split_a([3], ["low_a", "high_a"], a_crit="custom_a")
    assert np.all(
        typical_splitter.indices["low_a"] == [False, False, True, True, True])
    assert np.all(
        typical_splitter.indices["high_a"] == [True, True, False, False,
                                               False])
    assert typical_splitter.splitgroups["custom_a"] == {"low_a", "high_a"}
    with pytest.raises(KeyError):
        a_group = typical_splitter.subsample(["a"])

def test_group_replacement(typical_splitter):
    '''Replace a group classification.'''
    typical_splitter.split_a([3], ["low_a", "high_a"])
    typical_splitter.split_a([4], ["low_a", "high_a"])
    assert np.all(
        typical_splitter.indices["low_a"] == [False, False, True, True, True])
    assert np.all(
        typical_splitter.indices["high_a"] == [True, True, False, False,
                                               False])
    assert typical_splitter.splitgroups["a"] == {"low_a", "high_a"}

#TODO
def test_tilde_splitname(typical_splitter):
    '''Make sure function doesn't allow a split name to start with tilde.'''
    with pytest.raises(ValueError):
        typical_splitter.split_a([3], ["low_a", "~high_a"])
