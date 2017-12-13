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
        self.split_by_col(col, splitvalues, splitnames, a_crit, 
                          invert_inequality)
    def split_b(self, splitvalues, splitnames, col="b", b_crit="b",
                invert_inequality=False):
        '''Split by b'''
        self.split_by_col(col, splitvalues, splitnames, b_crit, 
                          invert_inequality)
    def split_c(self, splitvalues, splitnames, col="c", c_crit="c",
                invert_inequality=False):
        '''Split by c'''
        self.split_by_col(col, splitvalues, splitnames, c_crit, 
                          invert_inequality)

@pytest.fixture
def typical_splitter():
    '''Generate a general table.
    
    This has a straightforward split in column a. And a boundary split in
    column b.'''
    a = Table.read(tablestring, format="ascii.fixed_width")
    b = CustomSplitter(a)
    return b

def test_empty_split(typical_splitter):
    '''Make sure empty split raises an error.'''
    with pytest.raises(ValueError):
        typical_splitter.split_a([], ["a"])

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

def test_split_on_value(typical_splitter):
    '''Test that functions splits correctly on exact matches.'''
    typical_splitter.split_b([6.0], ["low_b", "high_b"])
    assert np.all(typical_splitter.indices["low_b"]  == [
        True, True, True, False, False])
    assert np.all(typical_splitter.indices["high_b"]  == [
        False, False, False, True, True])
    assert typical_splitter.splitgroups["b"] == {"low_b", "high_b"}

def test_inverted_inequality(typical_splitter):
    '''Test that functions splits correctly on exact matches.'''
    typical_splitter.split_b([6.0], ["low_b", "high_b"], invert_inequality=True)
    assert np.all(typical_splitter.indices["low_b"]  == [
        True, True, True, True, False])
    assert np.all(typical_splitter.indices["high_b"]  == [
        False, False, False, False, True])
    assert typical_splitter.splitgroups["b"] == {"low_b", "high_b"}

def test_multiple_inverted_inequality(typical_splitter):
    '''Test that the function inverts multiple values.

    This is to test for a syntax error that occurs when invert_inequality is
    used with multiple objects.'''
    typical_splitter.split_b([3.0, 6.0], ["low_b", "mid_b", "high_b"], 
                             invert_inequality=True)
    assert np.all(typical_splitter.indices["low_b"]  == [
        True, True, False, False, False])
    assert np.all(typical_splitter.indices["mid_b"]  == [
        False, False, True, True, False])
    assert np.all(typical_splitter.indices["high_b"]  == [
        False, False, False, False, True])
    assert typical_splitter.splitgroups["b"] == {"low_b", "mid_b", "high_b"}

def test_custom_group_name(typical_splitter):
    '''Split the table with a custom group name'''
    typical_splitter.split_a([3], ["low_custom_a", "high_custom_a"], 
                             a_crit="custom_a")
    assert np.all(
        typical_splitter.indices["low_custom_a"] == [
            False, False, True, True, True])
    assert np.all(
        typical_splitter.indices["high_custom_a"] == [
            True, True, False, False, False])
    assert typical_splitter.splitgroups["custom_a"] == {
        "low_custom_a", "high_custom_a"}
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

def test_tilde_splitname(typical_splitter):
    '''Make sure function doesn't allow a split name to start with tilde.'''
    with pytest.raises(ValueError):
        typical_splitter.split_a([3], ["low_a", "~high_a"])

def test_split_replacement(typical_splitter):
    '''Test that replacing a split groups cleans up correctly.'''
    typical_splitter.split_a([3], ["low_a", "high_a"])
    typical_splitter.split_a([1, 4], ["low_a", "mid_a", "higher_a"])
    assert np.all(
        typical_splitter.indices["low_a"] == [
            False, False, False, False, True])
    assert np.all(
        typical_splitter.indices["mid_a"] == [False, False, True, True, False])
    assert np.all(
        typical_splitter.indices["higher_a"] == [
            True, True, False, False, False])
    assert typical_splitter.splitgroups["a"] == {"low_a", "mid_a", "higher_a"}
    assert "high_a" not in typical_splitter.indices

def test_split_name_conflict(typical_splitter):
    '''Test that conflicting names for different groups throw an error.'''
    typical_splitter.split_a([3], ["low_a", "high_a"])
    with pytest.raises(ValueError):
        typical_splitter.split_b([3], ["low_b", "high_a"])

