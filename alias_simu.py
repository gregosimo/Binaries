import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def Raghavan_periods_for_rapid_rotators(est_rrs, rr_periodrange=(1, 5)):
    '''Create an array of periods with the estimated number of rapid rotators.

    Return an array of periods following the prescription in Raghavan et al
    (2010). This function creates an array large enough to have the estimated
    number of rapid rotators (as specified by periodrange).'''
    RAGHAVAN_MEAN = 5.03
    RAGHAVAN_DISP = 2.28
    e = np.exp(1)
    RAGHAVAN_LN_MEAN = RAGHAVAN_MEAN / np.log10(e)
    RAGHAVAN_LN_DISP = RAGHAVAN_DISP / np.log10(e)
    Raghavan_norm = stats.norm(loc=RAGHAVAN_LN_MEAN, scale=RAGHAVAN_LN_DISP)
    # The fraction of the total population estimated to be in rr_periodrange.
    # We divide by 2 because we will be using the lowend handle.
    range_frac = (Raghavan_norm.cdf(np.log(rr_periodrange[1])) -
                  Raghavan_norm.cdf(np.log(rr_periodrange[0]))) * 2
    print(range_frac)
    array_size = int(est_rrs / range_frac)
    logps = Raghavan_period_distribution(array_size, lowend=True)
    return logps

def Raghavan_period_distribution(array_size, lowend=False):
    '''Create an array of periods following Raghavan period distribution.

    The Raghavan et al (2010) period distribution is lognormal with a peak at
    :math:`\log P = 5.03` and standard deviation of :math:`\sigma_{\log P} =
    2.28`. This function will return an array of given array_size which
    contains periods derived from this distribution.'''
    RAGHAVAN_MEAN = 5.03
    RAGHAVAN_DISP = 2.28
    e = np.exp(1)
    RAGHAVAN_LN_MEAN = RAGHAVAN_MEAN / np.log10(e)
    RAGHAVAN_LN_DISP = RAGHAVAN_DISP / np.log10(e)
    logpdist = stats.norm.rvs(
        loc=RAGHAVAN_LN_MEAN, scale=RAGHAVAN_LN_DISP, size=array_size)
    # Since the period distribution is symmetric, and I only care about periods
    # really up to 10 days or so, I can get twice the number of points by
    # reassigning the points above the mean to below the mean.
    if lowend:
        highps = np.where(logpdist > RAGHAVAN_LN_MEAN)
        logpdist[highps] = RAGHAVAN_LN_MEAN - (logpdist[highps] - 
                                               RAGHAVAN_LN_MEAN)
    pdist = 10**(logpdist / np.log(10))
    return pdist

def effects_of_aliasing(pdist, upscatterprobs, downscatterprobs, 
                        rr_periodrange=(1,5), nbins=10):
    '''Show the effects of aliasing on a period distribution. 
    
    The underlying distribution should be given as pdist. Upscatterprobs and
    downscatterprobs should be given as two similarly-sized lists which contain
    different combinations of probabilities for upscattering and downscattering
    of the values.'''
    # Plot the underlying distribution.
    plt.hist(pdist, bins=nbins, range=rr_periodrange, histtype='bar',
             label="Raghavan")
    print("Total RR: {0:d}".format(len(pdist[np.logical_and(
        pdist > rr_periodrange[0], pdist <= rr_periodrange[1])])))
    for usprob, dsprob in zip(upscatterprobs, downscatterprobs):
        upscatter_indices = np.where(
            stats.binom.rvs(n=1, p=usprob, size=len(pdist)) == 1)
        downscatter_indices = np.where(
            stats.binom.rvs(n=1, p=dsprob, size=len(pdist)) == 1)
        scattered_array = np.array(pdist)
        scattered_array[upscatter_indices] = (
            scattered_array[upscatter_indices] * 2)
        scattered_array[downscatter_indices] = (
            scattered_array[downscatter_indices] / 2)

        scatterlabel = "up = {0:.2f}; down = {1:.2f}".format(usprob, dsprob)
        plt.hist(scattered_array, bins=nbins, range=rr_periodrange,
                 histtype='step', lw=2, label=scatterlabel)
        print("{0} total: {1:d}".format(scatterlabel, len(scattered_array[
            np.logical_and(scattered_array > rr_periodrange[0], 
                  scattered_array <= rr_periodrange[1])])))
    plt.xlabel("Period (day)")
    plt.ylabel("Counts")
    plt.title("Total Rapid Rotators: {0:d}".format(len(pdist[np.logical_and(
        pdist > rr_periodrange[0], pdist <= rr_periodrange[1])])))
