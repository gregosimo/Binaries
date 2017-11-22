import pytest
import numpy as np

from astropy.table import Table
import catalog

@pytest.fixture
def empty_table():
    '''Generate an empty table for testing purposes.'''
    emp = Table([[], [], []], names=("a", "b", "c"), 
                dtype=(np.float, np.float, np.float))
    return emp

@pytest.fixture
def gen_table():
    '''Generate a populated table for testing purposes.'''
    tbl = Table([[1, 2, 3], [4, 5, 6], [7, 8, 9]], names=("a", "b", "c"))
    return tbl

def test_split_empty(empty_table):
    '''Test that split doesn't freak out when it is given an empty table.'''
    splittables = catalog.split(empty_table, "b", [5])
    assert len(splittables) == 2
    assert np.all(splittables[0] == empty_table)
    assert np.all(splittables[1] == empty_table)



# Add a test for inverted relation.

def test_split(gen_table):
    '''Test that split works correctly with a typical table.'''
    # Remember that low <= val < high
    # Get a column in the middle.
    splittables = catalog.split(gen_table, 'b', [5])
    assert len(splittables) == 2
    assert np.all(splittables[0] == gen_table[:1])
    assert np.all(splittables[1] == gen_table[1:])

    # Get column at right.
    splittables = catalog.split(gen_table, 'a', [1])
    assert len(splittables) == 2
    assert np.all(splittables[0] == gen_table[:0])
    assert np.all(splittables[1] == gen_table)

    # Get column at left.
    splittables = catalog.split(gen_table, 'c', [9])
    assert len(splittables) == 2
    assert np.all(splittables[0] == gen_table[:2])
    assert np.all(splittables[1] == gen_table[2:])
    
    # Try underestimate
    splittables = catalog.split(gen_table, "a", [-2])
    assert len(splittables) == 2
    assert np.all(splittables[0] == gen_table[0:0])
    assert np.all(splittables[1] == gen_table)

    # Try overestimate
    splittables = catalog.split(gen_table, "c", [10])
    assert len(splittables) == 2
    assert np.all(splittables[0] == gen_table)
    assert np.all(splittables[1] == gen_table[0:0])

# Add a test for a non-list splitvalue.
def test_inverted(gen_table):
    '''Test with invert_inequality.'''
    # Now low < val <= high
    # Get a column in the middle.
    splittables = catalog.split(gen_table, 'b', [5])
    assert len(splittables) == 2
    assert np.all(splittables[0] == gen_table[:2])
    assert np.all(splittables[1] == gen_table[2:])

    # Get column at right.
    splittables = catalog.split(gen_table, 'a', [1])
    assert len(splittables) == 2
    assert np.all(splittables[0] == gen_table[:1])
    assert np.all(splittables[1] == gen_table[1:])

    # Get column at left.
    splittables = catalog.split(gen_table, 'c', [9])
    assert len(splittables) == 2
    assert np.all(splittables[0] == gen_table)
    assert np.all(splittables[1] == gen_table[:0])
    
    # Try underestimate
    splittables = catalog.split(gen_table, "a", [-2])
    assert len(splittables) == 2
    assert np.all(splittables[0] == gen_table[0:0])
    assert np.all(splittables[1] == gen_table)

    # Try overestimate
    splittables = catalog.split(gen_table, "c", [10])
    assert len(splittables) == 2
    assert np.all(splittables[0] == gen_table)
    assert np.all(splittables[1] == gen_table[0:0])

# Add a test for a non-list splitvalue.
def test_inverted(gen_table):
    '''Test with invert_inequality.'''
# Add a test for multiple splitvalues.
def test_multi_split(gen_table):
    '''Test that multiple split values work correctly with a typical table'''
    # Remember that low <= val < high
    splittables = catalog.split(gen_table, 'b', [4, 5])
    assert len(splittables) == 3
    assert np.all(splittables[0] == gen_table[:0])
    assert np.all(splittables[1] == gen_table[0:1])
    assert np.all(splittables[2] == gen_table[1:])

    # Make the low value out of bounds
    splittables = catalog.split(gen_table, 'a', [-1, 3])
    assert len(splittables) == 3
    assert np.all(splittables[0] == gen_table[:0])
    assert np.all(splittables[1] == gen_table[0:2])
    assert np.all(splittables[2] == gen_table[2:])

    # Make the high value out of bounds
    splittables = catalog.split(gen_table, 'c', [9, 15])
    assert len(splittables) == 3
    assert np.all(splittables[0] == gen_table[:2])
    assert np.all(splittables[1] == gen_table[2:])
    assert np.all(splittables[2] == gen_table[:0])
