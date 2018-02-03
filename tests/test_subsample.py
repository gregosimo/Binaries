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

def test_easy_split(typical_splitter):
    '''Test that the splitter works correctly in the simplest case.'''
    low_a = typical_splitter.subsample(["low_a"])
    high_a = typical_splitter.subsample(["high_a"])
    assert np.all(low_a == typical_splitter.data[2:])
    assert np.all(high_a == typical_splitter.data[:2])

def test_boundary(typical_splitter):
    '''Test that the splitter handles boundary values correctly.'''
    high_b = typical_splitter.subsample(["high_b"])
    assert np.all(high_b == typical_splitter.data[-2:])

def test_multi_split(typical_splitter):
    '''Test getting the intersection of two subsamples.'''
    mixed = typical_splitter.subsample(["low_a", "high_b"])
    assert np.all(mixed == typical_splitter.data[-2:])

def test_no_subsample(typical_splitter):
    '''Test that specifying no subsamples returns the same table.'''
    samesample = typical_splitter.subsample([])
    assert np.all(samesample == typical_splitter.data)

def test_invert_inequality(typical_splitter):
    '''Test that invert_inequality works correctly.'''
    inv_high_b = typical_splitter.subsample(["inv_high_b"])
    assert np.all(inv_high_b == typical_splitter.data[-1:])

def test_conflict_raises_exception(typical_splitter):
    '''Test that including conflicting keywords will raise an exception.'''
    with pytest.raises(ValueError):
        a_combo = typical_splitter.subsample(["high_a", "low_a"])

def test_exclusion(typical_splitter):
    '''Tests an exclusion category.'''
    low_a = typical_splitter.subsample(["~high_a"])
    assert np.all(low_a == typical_splitter.data[2:])

def test_multiple_exclusion(typical_splitter):
    '''Test one of more than two exclusion categories.'''
    higher_c = typical_splitter.subsample(["~low_c"])
    assert np.all(higher_c == typical_splitter.data[1:4])

def test_exclusion_conflict(typical_splitter):
    '''Test that exclusion conflicts are detected.'''
    with pytest.raises(ValueError):
        c_combo = typical_splitter.subsample(["low_c", "~high_c"])
