import numpy as np
import astropy.units as u
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy.constants import c

import sed

@u.quantity_input
def sample_RV_curve(point_array: u.day, amplitude: u.km / u.s, period: u.day,
                    phase: u.rad=0*u.rad):
    '''Sample an RV curve at the given points.

    The RV curve is characterized by an amplitude (generally V sini), a period
    (given in units of the point_array), as well as a phase. The phase will be
    defined as starting from when the object is at the furthest point from the
    Earth in its orbit.
    '''
    sample_RVs = amplitude * np.cos(
        2*np.pi * (point_array / period).to(
            u.rad, equivalencies=u.dimensionless_angles()) - np.pi/2*u.rad + 
        phase)
    return sample_RVs

@u.quantity_input
def rv_interval_sample(interval: u.hour, amplitude: u.km/u.s, period:u.day,
                       phase: u.rad=0*u.rad):
    '''Randomly samples an RV curve for a given number of points.

    The RV curve is characterized by an amplitude and period. The sampling 
    points will be randomly chosen.
    '''
    sample = np.array([0, interval.value])*interval.unit
    RV_results = sample_RV_curve(
        sample[:, np.newaxis], amplitude, period, phase)
    return RV_results

@u.quantity_input
def random_sample_RV_curves(
    npoints, nsamples, amplitude: u.km/u.s, period:u.day):
    '''Randomly samples an RV curve for a given number of points.

    The RV curve is characterized by an amplitude and period. The sampling 
    points will be randomly chosen.
    '''
    random_samples = np.random.rand(npoints, nsamples) * period
    RV_results = sample_RV_curve(random_samples, amplitude, period)
    return RV_results

@u.quantity_input
def MDM_sample_RV_curves(
        obs_per_day, obs_interval: u.hour, period: u.day, num_nights,
        nschedules=1):
    '''Creates a light curve simulating nocturnal observations.

    This will simulate some number of observations per day. While the
    observations are evenly spaced through the night, there is a stochastic
    offset. The observations will be taken in a period of time given by
    obs_interval, which should be the length of time for which the Kepler field
    is observable. The light curve will also be given a random offset in phase,
    which is calculated by randomly selecting a number less than the period.
    The total number of nights will be specified by num_nights.

    The number of potential observations to be realized is given by nschedules.
    If nschedules=100, then 100 different potential curves will be made.

    This does not take irregularities due to weather into account.
    '''
    # First make the backbond of observations. This should the grid where
    # regularly-spaced observations will be done. They will then be scattered
    # by adding in a random array.
    obs_steps, dobs = np.linspace(0, obs_interval.to(u.hour).value,
                                  obs_per_day, retstep=True)
    obs_intervals = np.tile(obs_steps, num_nights)
    night_count = np.repeat(np.arange(0, num_nights), obs_per_day)
    obs_grid = (obs_intervals + 24*night_count)*u.hour
    # Now add in the delay
    delay = np.random.rand(nschedules) * period
    random_fluctuations = (np.random.random_sample([nschedules, len(obs_grid)]) * 
                           dobs * obs_grid.unit)
    observing_times = delay[:, np.newaxis] + obs_grid + random_fluctuations
    return observing_times

@u.quantity_input
def check_randomly_distributed_RV_efficiency(
    npoints, amplitude: u.km/u.s, period: u.day, uncertainty: u.km/u.s, 
    probsigma=5, nsamplings=1000):
    '''Check the probability of mischaracterizing an RV curve as nonvariable.

    This will essentially run a suite of samplings (nsamplings times) to see 
    how often a given RV-variable curve will be mistaken for a non-variable 
    object. The samplings will have a number of points given as npoints. The 
    points will be assumed to be distributed normally, and the number of 
    sigma away from flat to be calculated is given in probsigma.
    '''
    # This holds the number of RV curves falsely classified as RV-nonvariable.
    RV_curves = random_sample_RV_curves(npoints, nsamplings, amplitude, period)
    flatcurves = consistent_with_flat(RV_curves, uncertainty, probsigma)
    numflats = np.count_nonzero(flatcurves == True)

    return numflats / nsamplings

@u.quantity_input
def check_MDM_sampling(
        obs_per_day, obs_interval: u.hour, period: u.day, num_nights,
        nschedules=1):
    '''Check probability of getting RV variability from daily observations.'''



# In order to generalize this, there needs to be a very firm understanding on
# what the shape of npoints is and what it represents. I don't think I'm
# capable of doing that yet.

@u.quantity_input
def check_single_sampling_interval(
    interval: u.hour, amplitude: u.km/u.s, period: u.day, 
    uncertainty: u.km/u.s, probsigma=5, nsamplings=1000):
    '''Probability of mischaracterizing a two-sample RV curve as nonvariable.
   
    This will sample an RV curve with two observations at a specified 
    interval with random phases.'''
    phases = np.random.rand(nsamplings) * 2*np.pi*u.rad
    RV_curves = rv_interval_sample(interval, amplitude, period, phase=phases)
    flatcurves = consistent_with_flat(RV_curves, uncertainty, probsigma)
    numflats = np.count_nonzero(flatcurves == True)

    return numflats / nsamplings

@u.quantity_input
def consistent_with_flat(RVpoints: u.km/u.s, uncertainty: u.km/u.s, probsigma): 
    '''Returns whether an RV curve is consistent with nonvariable.

    This function assumes that each of the points in RVpoints are
    Gaussian-distributed with a given uncertainty. A curve will be considered
    consistent with flat if the probability of having all points consistent
    with zero is greater than probsigma.
    
    This assumes all of the data points are independent, which likely isn't a
    bad assumption if the sampling truly is random.'''
    meanRV = np.mean(RVpoints, axis=0)
    chisq = np.sum((RVpoints - meanRV)**2, axis=0) / uncertainty**2
    flatsigma = sed.chi_squared_to_sigma(chisq, dof=RVpoints.shape[0])
    return flatsigma < probsigma

def plot_exclusion_probability(
    samplepoints, amplitude: u.km/u.s, period:u.day, uncertainties: u.km/u.s,
    probsigma=5, nsamplings=10000):
    '''Plot RV variability mischaracterization probability for RV errors.

    This makes a plot of how the probability of excluding the nonvariable model
    changes with amplitude for a given set of amplitudes.
    '''
    probvalues = np.empty(uncertainties.shape)
    for i in range(len(probvalues)):
        probvalues[i] = check_randomly_distributed_RV_efficiency(
            samplepoints, amplitude, period, uncertainties[i], probsigma,
            nsamplings)
    plt.plot(uncertainties.value, probvalues, 
             label="{0:d} points".format(samplepoints))
    plt.plot([0.3, 0.3], [0.0, 1.0], 'k--')
    plt.xlabel("RV uncertainty ({0})".format(uncertainties.unit))
    plt.ylabel("Probability")
    plt.title("Mischaracterization prob. for randomly sampled {0:.1f} "
              "RV curve".format(amplitude))

def plot_minimum_sampling_probability(
    amplitude: u.km/u.s, period: u.day, uncertainty: u.km/u.s, intervals:
    u.hour, probsigma=5, nsamplings=10000):
    '''Plot RV mischaracterization probability for minimum sampling intervals.

    Shows the probability of mischaracterizing a given target as nonvariable
    with two observations at a given interval apart.
    '''
    probvalues = np.empty(intervals.shape)
    for i in range(len(probvalues)):
        probvalues[i] = check_single_sampling_interval(
            intervals[i], amplitude, period, uncertainty, probsigma,
            nsamplings)
    plt.plot(intervals.value, probvalues)
    plt.xlabel("Sampling interval ({0})".format(intervals.unit))
    plt.ylabel("Probability")
    plt.title("Mischaracterization prob for RV curve sampled twice.")

def plot_velocity_smearing(
    velocity_amplitude: u.km/u.s, period: u.day, exposure_time: u.minute,
    numtimesteps=10000, phase: u.rad=np.pi/2*u.rad, line_center: u.um=1.57*u.um, 
    linewidth: u.km/u.s=6e-5 * u.um, specpixels=2048, 
    specrange=(1.51*u.um, 1.70*u.um)):
    '''Show how smearing over an observation affects a line.

    If a line in a RV-variable source is observed over a finite amount of time,
    its location will be smeared throughout the exposure. This function will
    plot the line shape of the smeared line compared to the instantaneous line 
    shape.'''
    sampled_times = np.linspace(-exposure_time/2, exposure_time/2, numtimesteps)
    sampled_velocities = sample_RV_curve(
        sampled_times, velocity_amplitude, period, phase=phase)
    central_wavelengths = line_center * (1 + sampled_velocities / c)
    wavelengths = np.linspace(specrange[0], specrange[1], specpixels)
    # I want this to produce array of len(wavelengths) x
    # len(central_wavelengths)
    line_profiles = gaussian_line_profile(
        wavelengths, central_wavelengths, linewidth)
    print(wavelengths)
    print(central_wavelengths.to(u.um))
    smeared_profile = np.average(line_profiles, axis=1)
    plt.plot(wavelengths, smeared_profile, 'b-', lw=3)
    plt.plot(wavelengths, line_profiles[:, numtimesteps//2], 'r-')

def gaussian_line_profile(
    wavelengths: u.um, central_wavelength: u.um, disp: u.um):
    '''Sample a gaussian line profile given a central wavelength and
    dispersion.'''
    gaussian = np.exp(-(wavelengths[:,np.newaxis] - central_wavelength)**2 /
                      (2 * disp**2)) / np.sqrt(2 * np.pi * disp**2)
    return gaussian
