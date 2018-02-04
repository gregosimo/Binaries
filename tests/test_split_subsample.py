import numpy as np

from astropy.table import Table
import pytest

import catalog
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
def empty_splitter():
    '''Generate an empty splitter
    
    No splits would have been made yet on this splitter.'''
    a = Table.read(tablestring, format="ascii.fixed_width")
    b = CustomSplitter(a)
    return b

@pytest.fixture
def single_split():
    '''Make a table with only one split.'''
    a = Table.read(tablestring, format="ascii.fixed_width")
    b = CustomSplitter(a)
    b.split_a([3.0], ["low_a", "high_a"])
    return b

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

def test_single_split(single_split):
    '''Test that the splitter works correctly in the simplest case.'''
    subsplitter = single_split.split_subsample(["low_a"])
    subsample = single_split.subsample(["low_a"])
    testsplitter = CustomSplitter(subsample)
    assert testsplitter == subsplitter

def test_full_split(typical_splitter):
    '''Test a full split.'''
    subsplitter = typical_splitter.split_subsample(["inv_high_b"])
    subsample = typical_splitter.subsample(["inv_high_b"])
    testsplitter = CustomSplitter(subsample)
    testsplitter.split_a([3], ["low_a", "high_a"])
    testsplitter.split_b([6.0], ["low_b", "high_b"])
    testsplitter.split_c([2, 6], ["low_c", "mid_c", "high_c"])
    assert testsplitter == subsplitter
