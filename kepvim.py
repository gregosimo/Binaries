'''
Module having to do with functions related to Variability Induced Motions.
'''

def filter_by_quarters(kepvimtable, vimquarters, kic_col="KIC"):
    '''Remove objects with fewer than the specified quarters of VIM detections.

    Return a table with only the entries which have at least vimquarters number
    of observations.'''
    filtered_objects = []
    # Using groups takes a REALLY long time. Is there a way to avoid this?
    vimgroups = kepvimtable.group_by(kic_col)
    for kicgroup in vimgroups.groups:
        if len(kicgroup) >= vimquarters:
            filtered_objects.append(kicgroup)
    filtered_table = vstack(filtered_objects)
    return filtered_table

def KICs_with_VIM_quarters(kepvimtable, minquarters, kic_col="KIC"):
    '''Get KIC IDs for objects with VIM detections over minquarters.

    Gets the KIC IDs for all of the KIC objects which have at least minquarters
    VIM detections over the campaign.
    '''
    newvim = filter_by_quarters(kepvimtable, minquarters, kic_col=kic_col)
    kepvimgroup = newvim.group_by(kic_col)
    kictable = au.first_row_in_group(kepvimgroup)
    return kictable[kic_col]
# Also want function that returns just KIC numbers that have VIM in at least n
# quarters.

def compress_kepVIM(kepvimtable, kic_col="KIC"):
    '''Compress each KIC to a unique row.

    The raw organization of the KepVIM catalog has a row for each unique
    combination of KIC and quarter. This function moves the quarter information
    from rows to columns; that way TBD'''
    fullcolnames = kepvimtable.colnames
    vimvariable_colnames = {
        'F50', 'Xpix', 'Ypix', 'r', 'AX', 'AY', 'ds_dF', 'PA', 'Ch', 'Q'}
    remaining_colnames = [fixedcol for fixedcol in fullcolnames if fixedcol not 
                          in vimvariable_colnames]
    unique_table = unique(kepvimtable[remaining_colnames], keys=kic_col)

    for quarter in range(1, 18):
        quarter_colname = "Q{0:d}".format(quarter)
        unique_table[quarter_colname] = np.zeros(len(unique_table))

        # Find all KIC values with VIM detections in a given quarter, and set
        # the corresponding entries in quarter_column to true.
        kic_detection_in_quarter = kepvimtable[kic_col][
            au.astropy_table_index(kepvimtable, "Q", quarter)]
        unique_kic_indices = au.astropy_table_indices(
            unique_table, kic_col, kic_detection_in_quarter)
        unique_table[quarter_colname][unique_kic_indices] = 1

    return unique_table

def kepVIM_quarter_table(kepvimtable, kic_col="KIC"):
    '''Create a table indicating which in quarters each KIC object had VIM.
    
    For the sake of making things sane again, this table has one row for each
    KIC object, and columns for each quarter, indicating in which quarter the 
    KIC object had VIM observations.'''
    quarter_table = Table([np.unique(kepvimtable[kic_col])])
    colcount = np.zeros(len(quarter_table))

    for quarter in range(1, 18):
        quarter_colname = "Q{0:d}".format(quarter)
        quarter_table[quarter_colname] = np.zeros(len(quarter_table))

        # Find all KIC values with VIM detections in a given quarter, and set
        # the corresponding entries in quarter_column to true.
        kic_detection_in_quarter = kepvimtable[kic_col][
            au.astropy_table_index(kepvimtable, "Q", quarter)]
        unique_kic_indices = au.astropy_table_indices(
            quarter_table, kic_col, kic_detection_in_quarter)
        quarter_table[quarter_colname][unique_kic_indices] = 1
        colcount += quarter_table[quarter_colname]

    quarter_table["Num_Q"] = colcount
    return quarter_table

def kepVIM_blending_statistics(magdiffs, offsets, quarters):
    '''Plots how contaminants affect various quarters of VIM.

    Generates a plot that shows how the number of quarters that an object
    experiences VIM is related to the magnitude and distance of the
    contaminant. The magnitude difference will be shown on the y-axis, the
    distance of the contaminant on the x-axis, and the color will reflect how
    many quarters of VIM it has.
    '''
    colormap = cm.viridis
    floatquarters = np.array(quarters, dtype=np.float)
    plt.scatter(offsets, magdiffs, c=floatquarters, cmap=colormap, s=8,
                edgecolors="face")
    cbar = plt.colorbar()
    cbar.set_label("Quarters")
    plt.xlabel("Distance from source")
    plt.ylabel("J_Target - J_Contam")
