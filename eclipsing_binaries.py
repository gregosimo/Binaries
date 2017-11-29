'''
Functions using the Kepler Eclipsing Binary population
'''

###############################################################################
# EBs and Rafa #
###############################################################################

def EBs_missed_by_Rafa(rafa_ebs, missed_ebs, xcol, ycol, xlabel="", ylabel=""):
    '''Compares the objects which were found by Rafa's code to those missed.

    This function will plot two columns which are in both rafa_ebs and
    missed_ebs against each other.'''
    plt.plot(rafa_ebs[xcol], rafa_ebs[ycol], 'ko', label="Detected")
    plt.plot(missed_ebs[xcol], missed_ebs[ycol], 'gx', label="Missed")
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(xcol)
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(ycol)

def missed_EBs_logg_teff(rafa_ebs, missed_ebs, loggcol="logg", teffcol="teff"):
    '''Plot the missed EBs in an HR diagram.'''
    EBs_missed_by_Rafa(rafa_ebs, missed_ebs, teffcol, loggcol, "teff", "log g")
    hr.invert_x_axis()
    hr.invert_y_axis()
    plt.legend(loc="upper right")

def missed_EBs_period_depth(rafa_ebs, missed_ebs, periodcol="period",
                            depthcol="pdepth"):
    '''Plot the missed EBs with period vs. eclipse depth.'''
    EBs_missed_by_Rafa(rafa_ebs, missed_ebs, periodcol, depthcol, 
                       "Period (day)", "Primary Eclipse Depth")

def missed_EBs_teff_depth(rafa_ebs, missed_ebs, teffcol="teff", 
                          depthcol="pdepth"):
    '''Plot the missed EBs with Teff vs. eclipse depth.'''
    EBs_missed_by_Rafa(rafa_ebs, missed_ebs, teffcol, depthcol, 
                       "Teff (K)", "Primary Eclipse Depth")
    hr.invert_x_axis()

def missed_EBs_period_width(rafa_ebs, missed_ebs, periodcol="period",
                            widthcol="pwidth"):
    '''Plot the missed EBs with period vs. eclipse width.'''
    EBs_missed_by_Rafa(rafa_ebs, missed_ebs, periodcol, widthcol, 
                       "Period (day)", "Primary Eclipse Width")

def missed_EBs_teff_width(rafa_ebs, missed_ebs, teffcol="teff",
                          widthcol="pwidth"):
    '''Plot the missed EBs with Teff vs. eclipse width.'''
    EBs_missed_by_Rafa(rafa_ebs, missed_ebs, teffcol, widthcol, 
                       "Period (day)", "Primary Eclipse Width")
    hr.invert_x_axis()

def missed_EBs_compare_histogram(rafa_ebs, missed_ebs, histcol, binrange,
                                 bins=50, xlabel=""):
    '''Create a histogram of missed vs detected EBs.'''
    plt.hist(rafa_ebs[histcol], range=binrange, bins=bins, label="Missing")
    plt.hist(missed_ebs[histcol], range=binrange, bins=bins, label="Observed")
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(histcol)
    plt.ylabel("Number")

def EB_plot(eb_periods, eb_vs, eb_flags):
    """Plots the eclipsing binary period vs velocity.

    This function plots the period as determined from Kepler light curves vs
    the vscatter calculated from APOGEE. In general, sin i ~ 1 for this case. 
    This plot functions to note whether VSCATTER is a useful determinant of
    whether an object is a tidally-synchronized binary.

    VSCATTER by itself is generally not a good measure. However, I would now
    like to see whether VMAX = VSCATTER * NOBS would be.
    """
    bad_indices = bad_ASPCAP_indices(eb_flags)
    good_indices = np.logical_not(bad_indices)

    # Plot the data.
    plt.plot(eb_periods[bad_indices], eb_vs[bad_indices], 'ro')
    plt.plot(eb_periods[good_indices], eb_vs[good_indices], 'ko')

    # Plot the lines.
    periodrange = np.linspace(0.01, 12, 100) * u.day
    standard_vel = bc.calc_velocity_of_binary(1*u.solMass, periodrange, 0.5)
    highmass_vel = bc.calc_velocity_of_binary(2*u.solMass, periodrange, 0.5)
    lowmass_vel = bc.calc_velocity_of_binary(0.5*u.solMass, periodrange, 0.5)
    highratio_vel = bc.calc_velocity_of_binary(1*u.solMass, periodrange, 1.0)
    lowratio_vel = bc.calc_velocity_of_binary(1*u.solMass, periodrange, 0.1)
    plt.plot(
        periodrange.to(u.day).value, standard_vel.to(u.km/u.s).value, 'k--',
        label="M=1, q=0.5")
    plt.plot(
        periodrange.to(u.day).value, highmass_vel.to(u.km/u.s).value, 'k--')
    plt.plot(
        periodrange.to(u.day).value, lowmass_vel.to(u.km/u.s).value, 'k--')
    plt.plot(
        periodrange.to(u.day).value, lowratio_vel.to(u.km/u.s).value, 'b--',
        label="M=1, q=0.1")
    plt.plot(
        periodrange.to(u.day).value, highratio_vel.to(u.km/u.s).value, 'r--',
        label="M=1, q=1.0")

    plt.xlabel("Period (day)")
    plt.ylabel("VMAX (km/s)")

    plt.xlim(0, 12)

###############################################################################
# Kepler EB tools #
###############################################################################

def remove_Kepler_EBs(maincat, McQuillan=True, mainkiccol="KIC"):
    '''Filters out Kepler Eclipsing Binaries.

    Removes the KIC values corresponding to the eclipsing binaries in the
    version of the EB catalog given in paths.EB_PATH.
    '''
    if McQuillan:
        ebcat = catin.mcquillan_ebs()
    else:
        ebcat = catin.read_villanova_EBs()
    filtered_maincat = au.filter_column_from_subtable(
            maincat, mainkiccol, ebcat["KIC"])
    return filtered_maincat

def read_Kirk_geometric_correction_spline(
    splinepath="/home/regulus/simonian/Binaries/Kirk_geometric_correction_spline.csv"):
    '''Read the spline that represents the geometric correction for EBs.

    The correction was taken from Fig. 11 in Kirk et al (2016).'''
    splinepoints = Table.read(
        splinepath, format="ascii.csv", data_start=0, 
        names=("period", "efficiency"))
    periods = np.log10(splinepoints["period"])
    corrections = splinepoints["efficiency"]
    correction_interpolator = interp1d(periods, corrections)
    return correction_interpolator

def num_missing_binaries(periods):
    '''Infer the number of noneclipsing binaries from eclipsing ones.
    
    Given a distribution of orbital periods of a sample of stars, this function
    uses the geometric correction to infer the number of binaries which should
    also exist in the sample.
    '''
    logperiods = np.log10(periods)
    # This should take the log of periods as input, and the efficiency of
    # detection based on the corrections.
    geofrac_spline = read_Kirk_geometric_correction_spline()
    efficiencies = geofrac_spline(logperiods)
    # These should be the number of non-eclipsing binaries for each eclipsing
    # binary.
    inferred_missing = 1.0 / efficiencies - 1
    num_inferred_missing = np.sum(inferred_missing)
    return num_inferred_missing

def EB_histogram_to_rotation_histogram(binvalues, bins):
    '''Convert EB histogram to one of expected rotatational modulation.

    The bin values should be the number of EBs in each bin. The bins should be
    the edges of each bin. This should basically take arguments of plt.hist().
    '''
    midbins = (bins[:-1] + bins[1:]) / 2
    logmidbins = np.log10(midbins)
    geofrac_spline = read_Kirk_geometric_correction_spline()
    efficiencies = geofrac_spline(logmidbins)
    missing_fraction = 1.0 / efficiencies - 1
    missing_histogram = binvalues * missing_fraction
    print(missing_fraction)
    rotation_fraction = rotation_modulation_fraction()
    rotation_histogram = rotation_fraction * missing_histogram
    return rotation_histogram
