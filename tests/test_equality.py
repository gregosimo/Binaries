import numpy as np

from astropy.table import Table
import pytest

from data_splitting import DataSplitter

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
                null_value=None, invert_inequality=False):
        '''Split by a'''
        self.split_by_col(
            col, splitvalues, splitnames, a_crit, null_value, invert_inequality)
    def split_b(self, splitvalues, splitnames, col="b", b_crit="b",
                null_value=None, invert_inequality=False):
        '''Split by b'''
        self.split_by_col(
            col, splitvalues, splitnames, b_crit, null_value, invert_inequality)
    def split_c(self, splitvalues, splitnames, col="c", c_crit="c",
                null_value=None, invert_inequality=False):
        '''Split by c'''
        self.split_by_col(
            col, splitvalues, splitnames, c_crit, null_value, invert_inequality)

@pytest.fixture
def typical_splitter():
    '''Generate a general table.
    
    This has a straightforward split in column a. And a boundary split in
    column b.'''
    a = Table.read(tablestring, format="ascii.fixed_width")
    b = CustomSplitter(a)
    b.split_a([3], ["low_a", "high_a"])
    b.split_b([6.0], ["low_b", "high_b"])
    b.split_b([6.0], ["inv_low_b", "inv_high_b"], b_crit="inv b", invert_inequality=True)
    b.split_c([2, 6], ["low_c", "mid_c", "high_c"])
    return b

@pytest.fixture
def empty_splitter():
    '''Generate an empty table.
    
    This splitter just has data in it without any splits.'''
    a = Table.read(tablestring, format="ascii.fixed_width")
    b = CustomSplitter(a)
    return b

def test_empty_equality():
    '''Test for equality for empty splitters.'''
    empty_table = Table()
    a = CustomSplitter(empty_table)
    b = CustomSplitter(empty_table)
    assert a == b

def test_simple_equality(empty_splitter):
    '''Test for equality with one split.'''
    newtable = Table.read(tablestring, format="ascii.fixed_width")
    copy_splitter = CustomSplitter(newtable)

    empty_splitter.split_b([6.0], ["low_b", "high_b"])
    copy_splitter.split_b([6.0], ["low_b", "high_b"])
    assert typical_splitter != copy_splitter


def test_different_data(empty_splitter):
    '''Ensure splitters with different data are different.'''
    newtable = Table.read(tablestring, format="ascii.fixed_width")
    newtable["b"][2] = 20
    copy_splitter = CustomSplitter(newtable)
    assert empty_splitter != copy_splitter

def test_different_split_value(empty_splitter):
    '''Different split values are different.'''
    newtable = Table.read(tablestring, format="ascii.fixed_width")
    copy_splitter = CustomSplitter(newtable)

    empty_splitter.split_a([3], ["low_a", "high_a"])
    copy_splitter.split_a([6.0], ["low_a", "high_a"])
    assert empty_splitter != copy_splitter

def test_different_crit(empty_splitter):
    '''Different crit names are different.'''
    newtable = Table.read(tablestring, format="ascii.fixed_width")
    copy_splitter = CustomSplitter(newtable)

    empty_splitter.split_c([3], ["low_c", "high_c"])
    copy_splitter.split_c([3], ["low_c", "high_c"], c_crit="DiffC")
    assert empty_splitter != copy_splitter

def test_different_index_name(empty_splitter):
    '''Check if one index name is different that there is inequality.'''
    newtable = Table.read(tablestring, format="ascii.fixed_width")
    copy_splitter = CustomSplitter(newtable)

    empty_splitter.split_c([3], ["low_c", "high_c"])
    copy_splitter.split_c([3], ["low_c", "high_c2"])
    assert empty_splitter != copy_splitter

def test_full_equality(typical_splitter):
    '''Ensure splitters with the same data, criteria, and indices are equal.'''
    newtable = Table.read(tablestring, format="ascii.fixed_width")
    copy_splitter = CustomSplitter(newtable)
    copy_splitter.split_a([3], ["low_a", "high_a"])
    copy_splitter.split_b([6.0], ["low_b", "high_b"])
    copy_splitter.split_b([6.0], ["inv_low_b", "inv_high_b"], b_crit="inv b", 
                          invert_inequality=True)
    copy_splitter.split_c([2, 6], ["low_c", "mid_c", "high_c"])
    assert typical_splitter == copy_splitter
