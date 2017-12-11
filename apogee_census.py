

import APOGEE_spectroscopy as apo

def display_fraction_census(aposplit, basic_crit=[]):
    '''Display a table showing the various levels of rapid rotation'''
    othercrits = basic_crit
    printfunc = print_not_bad_totals
    print()
    print_evstate_fractions(aposplit, othercrits, printfunc=printfunc)
    print()
    for state in aposplit.splitgroups["APOGEE logg"]:
        print("For {0}s".format(state))
        print()
        statecrits = othercrits + [state]
        print_vsini_detection_types(aposplit, statecrits, printfunc=printfunc)
        print()
        print_DLSB_fractions(aposplit, statecrits, printfunc=printfunc)
        print()
        statecrits.append("~DLSB")
        print_rapid_rotation_fractions(aposplit, statecrits, printfunc=printfunc)
        print()
        print_mcq_analysis_overlap(aposplit, statecrits, printfunc=printfunc)
        print()
        print_mcq_detections(aposplit, statecrits, printfunc=printfunc)
        print()

###############################################################################
# Functions for choosing quality
###############################################################################
    

def print_good_warn_totals(
        aposplit, fracfunc, othercrits, dispstring):
    '''Print totals for objects using both good and warn flags.
    
    This is a pretty generic function that should be used to display various
    lines of information that require knowing various fractions of objects that
    fulfill some criteria.
    
    Aposplit should be the data file that should be used to calculate the
    functions. Fracfunc should be a function that takes two arguments. One
    positional argument which should be aposplit, and a keyword argument
    "othercrits", which will take the necessary criteria that go into
    calculating the desired fractions.
    
    Dispstring should be a string which goes before the fractions.'''

    good_frac = fracfunc(aposplit, othercrit=othercrits+["Good"])
    warn_frac = fracfunc(aposplit, othercrit=othercrits+["Warn"])
    numerator = good_frac[0] + warn_frac[0]
    denominator = good_frac[1] + warn_frac[1]
    ratio = numerator / denominator
    fracstring = "{0:d}/{1:d} = {2:.1f}%".format(numerator, denominator,
                                                ratio*100)
    print(dispstring + fracstring)

def print_good_totals(
        aposplit, fracfunc, othercrits, dispstring):
    '''Print totals for objects using just good flags.
    
    This is a pretty generic function that should be used to display various
    lines of information that require knowing various fractions of objects that
    fulfill some criteria.
    
    Aposplit should be the data file that should be used to calculate the
    functions. Fracfunc should be a function that takes two arguments. One
    positional argument which should be aposplit, and a keyword argument
    "othercrits", which will take the necessary criteria that go into
    calculating the desired fractions.
    
    Dispstring should be a string which goes before the fractions.'''

    good_frac = fracfunc(aposplit, othercrit=othercrits+["Good"])
    numerator = good_frac[0]
    denominator = good_frac[1]
    ratio = numerator / denominator
    fracstring = "{0:d}/{1:d} = {2:.1f}%".format(numerator, denominator,
                                                ratio*100)
    print(dispstring + fracstring)

def print_not_bad_totals(
    aposplit, fracfunc, othercrits, dispstring):
    '''Print totals for objects without the bad flag.

    This is a pretty generic function that should be used to display various
    lines of information that require knowing various fractions of objects that
    fulfill some criteria.
    
    Aposplit should be the data file that should be used to calculate the
    functions. Fracfunc should be a function that takes two arguments. One
    positional argument which should be aposplit, and a keyword argument
    "othercrits", which will take the necessary criteria that go into
    calculating the desired fractions.
    
    Dispstring should be a string which goes before the fractions.'''

    notbad_frac = fracfunc(aposplit, othercrit=othercrits+["~Bad"])
    numerator = notbad_frac[0]
    denominator = notbad_frac[1]
    ratio = numerator / denominator
    fracstring = "{0:d}/{1:d} = {2:.1f}%".format(numerator, denominator,
                                                ratio*100)
    print(dispstring + fracstring)

###############################################################################
# Reusable functions for printing #
###############################################################################

def print_vsini_detection_types(
    aposplit, othercrits, printfunc=print_not_bad_totals):
    '''Print the types of vsini detections.

    These are: robust, marginal, and nondetection.'''

    nondet_string = "Fraction of nondetections (vsini < 7 km/s): "
    printfunc(
        aposplit, nondetection_fraction, othercrits, nondet_string)

    marginal_string = ("Fraction of marginal detections (7 km/s <= vsini " 
                       "< 10 km/s): ")
    printfunc(
        aposplit, marginal_detection_fraction, othercrits, 
        marginal_string)

    robust_string = "Fraction of robust detections (vsini >= 10 km/s): "
    printfunc(
        aposplit, robust_detection_fraction, othercrits, 
        robust_string)

def print_DLSB_fractions(aposplit, othercrits, printfunc=print_not_bad_totals):
    '''Print the fraction of DLSBs in the robust and marginal detections.'''

    robust_dlsb_string = "Fraction of robust non-DLSBs: "
    try:
        printfunc(
            aposplit, not_dlsb_fraction, othercrits+["Vsini det"], 
            robust_dlsb_string)
    except ZeroDivisionError:
        pass

    marginal_dlsb_string = "Fraction of marginal non-DLSBs: "
    try:
        printfunc(
            aposplit, not_dlsb_fraction, othercrits+["Vsini marginal"], 
            marginal_dlsb_string)
    except ZeroDivisionError:
        pass

def print_evstate_fractions(
    aposplit, othercrits, printfunc=print_not_bad_totals):
    '''Print the fraction of objects of evolutionary states in the sample.'''

    robust_giant_string = (
        "Fraction of robust detections that are giants: ")
    printfunc(
        aposplit, giant_fraction, othercrits+["Vsini det"], 
        robust_giant_string)

    robust_subgiant_string = (
        "Fraction of robust detections that are subgiants: ")
    printfunc(
        aposplit, subgiant_fraction, othercrits+["Vsini det"], 
        robust_subgiant_string)

    robust_dwarf_string = ("Fraction of robust detections that are dwarfs: ")
    printfunc(
        aposplit, dwarf_fraction, othercrits+["Vsini det"], 
        robust_dwarf_string)

    marginal_giant_string = (
        "Fraction of marginal detections that are giants: ")
    printfunc(
        aposplit, giant_fraction, othercrits+["Vsini marginal"], 
        marginal_giant_string)

    marginal_subgiant_string = (
        "Fraction of marginal detections that are subgiants: ")
    printfunc(
        aposplit, subgiant_fraction, othercrits+["Vsini marginal"],
        marginal_subgiant_string)

    marginal_dwarf_string = (
        "Fraction of marginal detections that are dwarfs: ")
    printfunc(
        aposplit, dwarf_fraction, othercrits+["Vsini marginal"], 
        marginal_dwarf_string)

def print_rapid_rotation_fractions(
    aposplit, othercrits, printfunc=print_not_bad_totals):
    '''Print fraction of rotators for robust and marginal detections.'''

    robust_very_rapid_fraction_string = (
        "Fraction of robust detections that are very rapid rotators "
        "(vsini >= 2piR/(1 day)): ")
    def robust_very_rapid(x, othercrit=[]):
        return rapid_rotator_fraction(
            x, othercrit=othercrit, detcrit="Vsini det", 
            rrcrit="Very rapid rotators")
    try:
        printfunc(
            aposplit, robust_very_rapid, othercrits, 
            robust_very_rapid_fraction_string)
    except ZeroDivisionError:
        pass

    robust_rapid_fraction_string = (
        "Fraction of robust detections that are rapid rotators "
        "(2piR / (1 day) > vsini >= 2piR/(5 day)): ")
    def robust_rapid(x, othercrit=[]):
        return rapid_rotator_fraction(
            x, othercrit=othercrit, detcrit="Vsini det")
    try:
        printfunc(
            aposplit, robust_rapid, othercrits, 
            robust_rapid_fraction_string)
    except ZeroDivisionError:
        pass

    robust_slow_fraction_string = (
        "Fraction of robust detections that are slow rotators "
        "(vsini < 2piR/(5 day)): ")
    def robust_slow(x, othercrit=[]):
        return slow_rotator_fraction(
            x, othercrit=othercrit, detcrit="Vsini det")
    try:
        printfunc(
            aposplit, robust_slow, othercrits, 
            robust_slow_fraction_string)
    except ZeroDivisionError:
        pass

    marginal_very_rapid_fraction_string = (
        "Fraction of marginal detections that are very rapid rotators "
        "(vsini >= 2piR / (1 day)): ")
    def marginal_very_rapid(x, othercrit=[]):
        return rapid_rotator_fraction(
            x, othercrit=othercrit, detcrit="Vsini marginal", 
            rrcrit="Very rapid rotators")
    try:
        printfunc(
            aposplit, marginal_very_rapid, othercrits, 
            marginal_very_rapid_fraction_string)
    except ZeroDivisionError:
        pass

    marginal_rapid_fraction_string = (
        "Fraction of marginal detections that are rapid rotators "
        "(vsini >= 2piR/(5 day)): ")
    def marginal_rapid(x, othercrit=[]):
        return rapid_rotator_fraction(
            x, othercrit=othercrit, detcrit="Vsini marginal")
    try:
        printfunc(
            aposplit, marginal_rapid, othercrits, 
            marginal_rapid_fraction_string)
    except ZeroDivisionError:
        pass

    marginal_slow_fraction_string = (
        "Fraction of marginal detections that are slow rotators "
        "(vsini < 2piR/(5 day)): ")
    def marginal_slow(x, othercrit=[]):
        return slow_rotator_fraction(
            x, othercrit=othercrit, detcrit="Vsini marginal")
    try:
        printfunc(
            aposplit, marginal_slow, othercrits, 
            marginal_slow_fraction_string)
    except ZeroDivisionError:
        pass

def print_mcq_analysis_overlap(
    aposplit, othercrits, printfunc=print_not_bad_totals):
    '''Print how much of the various groups were analyzed by McQuillan'''

    robust_very_rapid_mcq_analysis_string = (
        "Fraction of robust very rapid rotators analyzed by McQuillan : ")
    try:
        printfunc(
            aposplit, mcquillan_analysis_fraction, 
            othercrits+["Vsini det", "Very rapid rotators"], 
            robust_very_rapid_mcq_analysis_string)
    except ZeroDivisionError:
        pass

    robust_rapid_mcq_analysis_string = (
        "Fraction of robust rapid rotators analyzed by McQuillan: ")
    try:
        printfunc(
            aposplit, mcquillan_analysis_fraction, 
            othercrits+["Vsini det", "Rapid rotators"], 
            robust_rapid_mcq_analysis_string)
    except ZeroDivisionError:
        pass

    robust_slow_mcq_analysis_string = (
        "Fraction of robust slow rotators analyzed by McQuillan: ")
    try:
        printfunc(
            aposplit, mcquillan_analysis_fraction, 
            othercrits+["Vsini det", "Slow rotators"], 
            robust_slow_mcq_analysis_string)
    except ZeroDivisionError:
        pass

    marginal_very_rapid_mcq_analysis_string = (
        "Fraction of marginal very rapid rotators analyzed by McQuillan: ")
    try:
        printfunc(
            aposplit, mcquillan_analysis_fraction, 
            othercrits+["Vsini marginal", "Very rapid rotators"], 
            marginal_very_rapid_mcq_analysis_string)
    except ZeroDivisionError:
        pass

    marginal_rapid_mcq_string = (
        "Fraction of marginal rapid rotators analyzed by McQuillan: ")
    try:
        printfunc(
            aposplit, mcquillan_analysis_fraction, 
            othercrits+["Vsini marginal", "Rapid rotators"],
            marginal_rapid_mcq_string)
    except ZeroDivisionError:
        pass

    marginal_slow_mcq_string = (
        "Fraction of marginal slow rotators analyzed by McQuillan: ")
    try:
        printfunc(
            aposplit, mcquillan_analysis_fraction, 
            othercrits+["Vsini marginal", "Slow rotators"],
            marginal_slow_mcq_string)
    except ZeroDivisionError:
        pass

    nondet_mcq_string = (
        "Fraction of nondetections analyzed by McQuillan: ")
    def nondet_mcq(x, othercrit=[]):
        nondets = mcquillan_analysis_fraction(
            aposplit, othercrit=othercrit+["Vsini nondet"])
        bads = mcquillan_analysis_fraction(
            aposplit, othercrit=othercrit+["No Vsini"])
        return (nondets[0]+bads[0], nondets[1]+bads[1])
    try:
        printfunc(aposplit, nondet_mcq, othercrits, nondet_mcq_string)
    except ZeroDivisionError:
        pass

def print_mcq_detections(aposplit, othercrits, printfunc=print_not_bad_totals):
    '''Print the number of rotators with McQuillan detections.'''

    robust_very_rapid_mcq_string = (
        "Fraction of robust very rapid rotators with "
        "McQuillan detections: ")
    try:
        printfunc(
            aposplit, mcquillan_detection_fraction, 
            othercrits+["Vsini det", "Very rapid rotators"], 
            robust_very_rapid_mcq_string)
    except ZeroDivisionError:
        pass

    robust_rapid_mcq_string = ("Fraction of robust rapid rotators with "
                               "McQuillan detections: ")
    try:
        printfunc(
            aposplit, mcquillan_detection_fraction, 
            othercrits+["Vsini det", "Rapid rotators"], 
            robust_rapid_mcq_string)
    except ZeroDivisionError:
        pass

    robust_slow_mcq_string = ("Fraction of robust slow rotators with "
                              "McQuillan detections: ")
    try:
        printfunc(
            aposplit, mcquillan_detection_fraction, 
            othercrits+["Vsini det", "Slow rotators"], 
            robust_slow_mcq_string)
    except ZeroDivisionError:
        pass

    marginal_very_rapid_mcq_string = (
        "Fraction of marginal very rapid rotators with "
        "McQuillan detections: ")
    try:
        printfunc(
            aposplit, mcquillan_detection_fraction, 
            othercrits+["Vsini marginal", "Very rapid rotators"], 
            marginal_very_rapid_mcq_string)
    except ZeroDivisionError:
        pass

    marginal_rapid_mcq_string = ("Fraction of marginal rapid rotators with "
                                 "McQuillan detections: ")
    try:
        printfunc(
            aposplit, mcquillan_detection_fraction, 
            othercrits+["Vsini marginal", "Rapid rotators"],
            marginal_rapid_mcq_string)
    except ZeroDivisionError:
        pass

    marginal_slow_mcq_string = ("Fraction of marginal slow rotators with "
                                "McQuillan detections: ")
    try:
        printfunc(
            aposplit, mcquillan_detection_fraction, 
            othercrits+["Vsini marginal", "Slow rotators"],
            marginal_slow_mcq_string)
    except ZeroDivisionError:
        pass

    nondet_mcq_string = "Fraction of nondetections with McQuillan detections: "
    def nondet_mcq(x, othercrit=[]):
        nondets = mcquillan_detection_fraction(
            aposplit, othercrit=othercrit+["Vsini nondet"])
        bads = mcquillan_detection_fraction(
            aposplit, othercrit=othercrit+["No Vsini"])
        return (nondets[0]+bads[0], nondets[1]+bads[1])
    try:
        printfunc(
        aposplit, nondet_mcq, othercrits, nondet_mcq_string)
    except ZeroDivisionError:
        pass


###############################################################################
# Functions for calculating fractions #
###############################################################################

def nondetection_fraction(
    aposplit, detcrit="Vsini nondet", novsinicrit="No Vsini", othercrit=[]):
    '''Return the number of vsini nondetections in the sample.

    The nondetections are given in the criterion of detcrit. Usually this is due
    to some threshold in vsini. Other cuts on categories can be specified in
    othercrit.'''
    nondetlen = (aposplit.subsample_len([detcrit] + othercrit) +
                 aposplit.subsample_len([novsinicrit] + othercrit))
    fullsamp = aposplit.subsample_len(othercrit)
    return (nondetlen, fullsamp)

def marginal_detection_fraction(
        aposplit, marginalcrit="Vsini marginal", othercrit=[]):
    '''Return the number of vsini marginal detections in the sample.

    The marginal detections are given in the criterion of marginalcrit. This is
    usually some vsini cut between 7-10 km/s or so. Other cuts on categories
    can be specified in othercrit.'''
    marginal_len = aposplit.subsample_len([marginalcrit] + othercrit)
    fullsamp = aposplit.subsample_len(othercrit)
    return (marginal_len, fullsamp)

def robust_detection_fraction(
        aposplit, rapidcrit="Vsini det", othercrit=[]):
    '''Return the number of robust vsini detections in the sample.

    The robust detections are given in the criterion of rapidcrit. Usually this
    is above some cutoff. Other cuts on categories can be specified in
    othercrit.'''
    robust_len = aposplit.subsample_len([rapidcrit] + othercrit)
    fullsamp = aposplit.subsample_len(othercrit)
    return (robust_len, fullsamp)

def slow_rotator_fraction(
    aposplit, slcrit="Slow rotators", detcrit="Vsini det", othercrit=[]):
    '''Return the fraction of slow rotators in a sample.

    The slow rotators are those with a vsini detection, but not a large enough
    vsini to place them into the rapid rotation regime.'''
    srlen = aposplit.subsample_len([detcrit, slcrit] + othercrit) 
    fullsamp = aposplit.subsample_len([detcrit] + othercrit)
    return (srlen, fullsamp)

def rapid_rotator_fraction(
    aposplit, rrcrit="Rapid rotators", detcrit="Vsini det", othercrit=[]):
    '''Return the number of rapid rotators and total sample.

    The rapid rotators are those classified under rrcrit, and those with
    matching vsini detections are under detcrit. Other cuts on categories can
    be specified in othercrit.'''

    rrlen = aposplit.subsample_len([rrcrit, detcrit] + othercrit)
    fullsamp = aposplit.subsample_len([detcrit] + othercrit)
    return (rrlen, fullsamp)

def mcquillan_analysis_fraction(
    aposplit, mcqcrit="Unknown Mcq", othercrit=[]):
    '''Get the fraction of the data that were analyzed by McQuillan.

    Returns a 2-tuple containing the number of objects in the subsample
    analyzed by McQuillan, along with the total number of objects in the
    subsample.'''
    mcqlen = aposplit.subsample_len([mcqcrit]+othercrit)
    fullsamp = aposplit.subsample_len(othercrit)
    return (fullsamp - mcqlen, fullsamp)

def mcquillan_detection_fraction(
    aposplit, mcqcrit="Mcq", unknowncrit="~Unknown Mcq", othercrit=[]):
    '''Get the fraction of the data that has a McQuillan detection.

    Returns a 2-tuple containing the number of objects in the subsample with
    McQuillan detections, along with the total number of objects in the
    subsample.'''
    mcqlen = aposplit.subsample_len([mcqcrit]+othercrit)
    fullsamp = aposplit.subsample_len([unknowncrit]+othercrit)
    return (mcqlen, fullsamp)

def not_dlsb_fraction(
    aposplit, nodlcrit="~DLSB", othercrit=[]):
    '''Get the fraction of data that are not known to be DLSBs.

    Return a 2-tuple containing the number of objects in the subsample not
    known to be DLSBs, along with the total number of objects in the
    subsample.'''
    nodllen = aposplit.subsample_len([nodlcrit]+othercrit)
    fullsamp = aposplit.subsample_len(othercrit)
    return nodllen, fullsamp

def giant_fraction(
    aposplit, giantcrit="Giant", othercrit=[]):
    '''Get the fraction of objects that are giants.

    Return a 2-tuple containing the number of objects in the subsample which
    have been classified as giants, along with the total number of objects
    in the subsample.'''
    giantlen = aposplit.subsample_len([giantcrit]+othercrit)
    fullsamp = aposplit.subsample_len(othercrit)
    return giantlen, fullsamp

def subgiant_fraction(
    aposplit, subgiantcrit="Subgiant", othercrit=[]):
    '''Get the fraction of objects that are subgiants.

    Return a 2-tuple containing the number of objects in the subsample which
    have been classified as subgiants, along with the total number of objects
    in the subsample.'''
    subgiantlen = aposplit.subsample_len([subgiantcrit]+othercrit)
    fullsamp = aposplit.subsample_len(othercrit)
    return subgiantlen, fullsamp

def dwarf_fraction(
    aposplit, dwarfcrit="Dwarf", othercrit=[]):
    '''Get the fraction of objects that are dwarfs.

    Return a 2-tuple containing the number of objects in the subsample which
    have been classified as dwarfs, along with the total number of objects
    in the subsample.'''
    dwarflen = aposplit.subsample_len([dwarfcrit]+othercrit)
    fullsamp = aposplit.subsample_len(othercrit)
    return dwarflen, fullsamp
