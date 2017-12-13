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
        self.split_by_col(col, splitvalues, splitnames, a_crit, invert_inequality)
    def split_b(self, splitvalues, splitnames, col="b", b_crit="b",
                invert_inequality=False):
        '''Split by b'''
        self.split_by_col(col, splitvalues, splitnames, b_crit, invert_inequality)
    def split_c(self, splitvalues, splitnames, col="c", c_crit="c",
                invert_inequality=False):
        '''Split by c'''
        self.split_by_col(col, splitvalues, splitnames, c_crit, invert_inequality)

@pytest.fixture
def typical_splitter():
    '''Generate a general table.
    
    This has a straightforward split in column a. And a boundary split in
    column b.'''
    a = Table.read(tablestring, format="ascii.fixed_width")
    b = CustomSplitter(a)
    b.split_a([3], ["low_a", "high_a"])
    b.split_a([2], ["custom_low_a", "custom_low_b"], a_crit="custom a")
    b.split_b([6.0], ["low_b", "high_b"])
    b.split_b([6.0], ["inv_low_b", "inv_high_b"], invert_inequality=True)
    b.split_c([1, 6], ["low_c", "mid_c", "high_c"])
    return b

def test_default_delete(typical_splitter):
    '''Check that the simplest type of delete works.'''
    typical_splitter.delete_crit("a")
    assert "a" not in typical_splitter.splitgroups
    assert "low_a" not in typical_splitter.indices
    assert "high_a" not in typical_splitter.indices

def test_custom_delete(typical_splitter):
    '''Check that a custom delete will work.'''
    typical_splitter.delete_crit("custom a")
    assert "custom a" not in typical_splitter.splitgroups
    assert "custom_low_a" not in typical_splitter.indices
    assert "custom_high_a" not in typical_splitter.indices

def test_delete_nonexistant(typical_splitter):
    '''Check that deleting a criterion that doesn't work throws an error.'''
    with pytest.raises(KeyError):
        typical_splitter.delete_crit("zzz")
