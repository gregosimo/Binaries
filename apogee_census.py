

import APOGEE_spectroscopy as apo

def display_fraction_census(
        aposplit, nondet="Vsini nondet", badvsini="Bad Vsini",
        robust_vsini="Vsini det", marginal_vsini="Vsini marginal"):
    '''Display a table showing the various levels of rapid rotation'''
    for tefflabel in ["4100-4700", "4700-5300", "5300-5900"]:
        if tefflabel == "5300-5900":
            printfunc = print_good_totals
        else:
            printfunc = print_good_warn_totals
        print(tefflabel)
        print()
        loggtype = "~Huber giant"
        othercrits = ["~DLSB", tefflabel, loggtype]
        nondet_string = "Fraction of nondetections (vsini < 7 km/s): "
        printfunc(
            aposplit, apo.nondetection_fraction, othercrits, nondet_string)

        marginal_string = ("Fraction of marginal detections (7 km/s <= vsini " 
                           "< 10 km/s): ")
        printfunc(
            aposplit, apo.marginal_detection_fraction, othercrits, 
            marginal_string)

        robust_string = "Fraction of robust detections (vsini >= 10 km/s): "
        printfunc(
            aposplit, apo.robust_detection_fraction, othercrits, 
            robust_string)

        print()

        robust_very_rapid_fraction_string = (
            "Fraction of robust detections that are very rapid rotators "
            "(vsini >= 2piR/(1 day)): ")
        def robust_very_rapid(x, othercrit=[]):
            return apo.rapid_rotator_fraction(
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
            return apo.rapid_rotator_fraction(
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
            return apo.slow_rotator_fraction(
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
            return apo.rapid_rotator_fraction(
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
            return apo.rapid_rotator_fraction(
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
            return apo.slow_rotator_fraction(
                x, othercrit=othercrit, detcrit="Vsini marginal")
        try:
            printfunc(
                aposplit, marginal_slow, othercrits, 
                marginal_slow_fraction_string)
        except ZeroDivisionError:
            pass

        
        print()

        robust_very_rapid_mcq_string = (
            "Fraction of robust very rapid rotators with "
            "McQuillan detections: ")
        try:
            printfunc(
                aposplit, apo.mcquillan_detection_fraction, 
                othercrits+["Vsini det", "Very rapid rotators"], 
                robust_very_rapid_mcq_string)
        except ZeroDivisionError:
            pass

        robust_rapid_mcq_string = ("Fraction of robust rapid rotators with "
                                   "McQuillan detections: ")
        try:
            printfunc(
                aposplit, apo.mcquillan_detection_fraction, 
                othercrits+["Vsini det", "Rapid rotators"], 
                robust_rapid_mcq_string)
        except ZeroDivisionError:
            pass

        robust_slow_mcq_string = ("Fraction of robust slow rotators with "
                                  "McQuillan detections: ")
        try:
            printfunc(
                aposplit, apo.mcquillan_detection_fraction, 
                othercrits+["Vsini det", "Slow rotators"], 
                robust_slow_mcq_string)
        except ZeroDivisionError:
            pass

        marginal_very_rapid_mcq_string = (
            "Fraction of marginal very rapid rotators with "
            "McQuillan detections: ")
        try:
            printfunc(
                aposplit, apo.mcquillan_detection_fraction, 
                othercrits+["Vsini marginal", "Very rapid rotators"], 
                marginal_very_rapid_mcq_string)
        except ZeroDivisionError:
            pass

        marginal_rapid_mcq_string = ("Fraction of marginal rapid rotators with "
                                     "McQuillan detections: ")
        try:
            printfunc(
                aposplit, apo.mcquillan_detection_fraction, 
                othercrits+["Vsini marginal", "Rapid rotators"],
                marginal_rapid_mcq_string)
        except ZeroDivisionError:
            pass

        marginal_slow_mcq_string = ("Fraction of marginal slow rotators with "
                                    "McQuillan detections: ")
        try:
            printfunc(
                aposplit, apo.mcquillan_detection_fraction, 
                othercrits+["Vsini marginal", "Slow rotators"],
                marginal_slow_mcq_string)
        except ZeroDivisionError:
            pass

        nondet_mcq_string = "Fraction of nondetections with McQuillan detections: "
        def nondet_mcq(x, othercrit=[]):
            nondets = apo.mcquillan_detection_fraction(
                aposplit, othercrit=othercrit+["Vsini nondet"])
            bads = apo.mcquillan_detection_fraction(
                aposplit, othercrit=othercrit+["No Vsini"])
            return (nondets[0]+bads[0], nondets[1]+bads[1])
        try:
            printfunc(
            aposplit, nondet_mcq, othercrits, nondet_mcq_string)
        except ZeroDivisionError:
            pass
        print()

def print_good_warn_totals(
        aposplit, fracfunc, othercrits, dispstring):
    '''Print totals for objects using both good and warn flags.
    
    This is a pretty generic function that should be used to display various
    lines of information taht require knowing various fractions of objects that
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
    lines of information taht require knowing various fractions of objects that
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
