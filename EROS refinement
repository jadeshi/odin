#!/usr/bin/env python
import os
import numpy as np
import random
from msmbuilder import arglib

# Scripts to manipulate the input data
def get_intensities(input):
    '''Extracts intensity data from a crysol .int file and returns a numpy array containing the intensities.'''
    return np.loadtxt(input, skiprows=1, usecols=(0, 1))[:,1]


def get_q(input):
    '''Extracts the q range from a crysol .int file and returns a numpy array containing the range of q used.'''
    return np.loadtxt(input, skiprows=1, usecols=(0, -1))[:,0]


def interpolate(data, axis):
    '''Takes a 2D numpy array "data" and a 1D numpy array "axis", and maps the values of the 2nd column of "data" to "axis" by doing a linear interpolation of the original data.'''
    output = np.zeros(len(axis))
    y0 = data[:,1]
    x0 = data[:,0]
    for i in range(len(axis)):
        start = np.where(x0 < axis[i])[0][-1]
        end = np.where(x0 > axis[i])[0][0]
        slope = (y0[end] - y0[start]) / (x0[end] - x0[start])
        interpolated = y0[start] + slope * (axis[i] - x0[start])
        output[i] = interpolated
    return output


def opt_normalize(data, ref, ref_error):
    '''Optimizes the fit of a 1D numpy array "data" and a reference array "ref" by multiplying "data" by a constant normalization factor. The standard error of "data" should also be provided in order to avoid over-fitting to noise. The cu
toff is the RMS value under which the optimization should stop.'''
    c = max(ref) / max(data)
    current_obj = sum((c * data - ref) ** 2 / ref_error ** 2) * 1 / len(data)
    perturb = 10 ** -2
    perturb_cutoff = 10 ** -5
    mode = 'increase'
    tolerance = 0
    absolute_tolerance = 2
    while current_obj > 0:
        if mode == 'increase':
            new_c = (1 + perturb) * c
        elif mode == 'decrease':
            new_c = (1 - perturb) * c
        new_obj = sum((new_c * data - ref) ** 2 / ref_error ** 2) * 1 / len(data)
        if new_obj >= current_obj and mode == 'increase':
            mode = 'decrease'
        elif new_obj >= current_obj and mode == 'decrease':
            mode = 'increase'
        elif new_obj < current_obj:
            c = new_c
            current_obj = new_obj
            #print 'New objective function is %s. Accepting new normalization factor c = %s.'%(str(new_obj), str(c))
        # Ensure nothing gets stuck oscillating around the minimum
        if current_obj != new_obj:
            tolerance += 1
        if tolerance >= absolute_tolerance:
            perturb *= 0.5
            tolerance = 0
        if perturb < perturb_cutoff: # Prevents perturbation from decreasing forever without doing something productive.
            break
    return c * data


# EROS
def prepare_crysol_data(input):
    '''Given a directory of crysol intensities, returns a 2D numpy array of all the intensities, ordered in increasing numerical order by their filename.
    The intended naming scheme of the of crysol intensities files is i00.int, where i is the microstate number. So microstate 0 would correspond to 000.int, state 1 would be 100.int, and so on.'''
    files = []
    for i in os.listdir(input):
        if 'int' in i:
            files.append(int(i[:i.index('00.int')]))
    files.sort()
    intensities = np.array([get_intensities('%s/%d00.int'%(input, files[0]))])
    for i in files[1:]:
        intensity = np.array([get_intensities('%s/%d00.int'%(input, i))])
        intensities = np.vstack((intensities, intensity))
    return intensities


def generate_average_profile(weights, intensities):
    '''Returns an average intensity profile for an ensemble.'''
    avg_profile = weights[0] * intensities[0]
    for i in range(1, len(intensities)):
        avg_profile += weights[i] * intensities[i]
    return avg_profile


def perturb_weights(weights, mag=0.25):
    # Prepare new weights vector...
    new_weights = np.zeros(len(weights))
    for i in range(len(weights)):
        new_weights[i] = weights[i]
    # Check for outstanding weights, and if so, bias the algorithm to drain probability from it with probability corresponding to its weight
    bias = max(weights) / sum(weights)
    seed = random.random()
    if seed < bias:
        from_weight = np.where(weights == max(weights))[0][0]
    else:
        from_weight = random.choice(range(len(new_weights)))
    to_weight = random.choice(range(len(new_weights)))
    perturbation = new_weights[from_weight] * mag 
    new_weights[from_weight] -= perturbation
    new_weights[to_weight] += perturbation
    # Renormalize
    return new_weights / sum(new_weights)


def calculate_regularization_penalty(initial_weights, current_weights, theta):
    S = -1 * sum(current_weights * np.log(current_weights / initial_weights))
    return theta * S


def calculate_RMS_error(current_weights, intensities, ref_intensities, ref_error):
    profile = generate_average_profile(current_weights, intensities)
    return sum((ref_intensities - opt_normalize(profile, ref_intensities, ref_error)) ** 2 / (ref_error ** 2)) / len(profile)


def calculate_objective_function(initial_weights, current_weights, intensities, ref_intensities, ref_error, theta):
    gamma_sq = calculate_RMS_error(current_weights, intensities, ref_intensities, ref_error)
    theta_S = calculate_regularization_penalty(initial_weights, current_weights, theta)
    return  gamma_sq - theta_S


def simulated_annealing(initial_weights, current_weights, intensities, ref_intensities, ref_error, T, theta):
    # Print statements aren't necssary but good for keeping track of where you are. Running the script too long can make excessively big text files though if you print to stdout.
    current_obj = calculate_objective_function(initial_weights, current_weights, intensities, ref_intensities, ref_error, theta)
    print 'Current objective function: %s'%(str(current_obj))
    current_RMS = calculate_RMS_error(current_weights, intensities, ref_intensities, ref_error)
    print 'Current RMS: %s'%(str(current_RMS))
    new_weights = perturb_weights(current_weights)
    new_obj = calculate_objective_function(initial_weights, new_weights, intensities, ref_intensities, ref_error, theta)
    print 'Objective function step: %s'%(str(new_obj - current_obj))
    if new_obj <= current_obj:
        print 'Accepting step'
        return new_weights, new_obj
    elif new_obj > current_obj:
        accept = random.random()
        if accept <= np.e ** (-1 * (new_obj - current_obj) / T):
            print 'Accepting step'
            return new_weights, new_obj
        elif accept > np.e ** (-1 * (new_obj - current_obj) / T):
            print 'Rejecting step'
            return current_weights, current_obj    
    
    
def optimize_weights(initial_weights, intensities, ref_intensities, ref_error, cutoff, T, theta, restart_weights):
    if restart_weights != '-1':
        weights = restart_weights
    else:
        weights = np.copy(initial_weights)
    current_obj = calculate_objective_function(initial_weights, weights, intensities, ref_intensities, ref_error, theta)
    # Record best objective function
    best_obj = float('Inf')
    while current_obj > cutoff: # Cutoff is normally set to 0 so that the script gives the best possible answer
        weights, current_obj = simulated_annealing(initial_weights, weights, intensities, ref_intensities, ref_error, T, theta)
        # Write weights and profile if objective function is a global minimum so far
        if current_obj < best_obj:
            best_obj = current_obj
            np.savetxt('prior_weights', weights)
            profile = generate_average_profile(weights, intensities)
            profile = opt_normalize(profile, ref_intensities, ref_error)
            np.savetxt('prior_profile', profile)
    return 0


if __name__ == '__main__':
    parser = arglib.ArgumentParser(description='''Fit a set of CRYSOL scattering profiles to a reference experimental profile using the EROS algorithm. The output consists of two files which are updated in real time during the optimizatio
n process: 
    - prior_profile: currently optimal scattering profile I(q). This is to be plotted using the EXPERIMENTAL q-range (i.e. the first column in "exp_ints") 
    - prior_weights: currently optimal weights.
    Also, will print a log of the optimization process. Recommend writing to standard output.''')
    parser.add_argument('exp_ints', type=str, help='''Experimental scattering profile to which to fit the simulated data. Must be in the form of a 3-column text file with the first column being q, second being I(q), and third being the ex
perimental uncertainty.''')
    parser.add_argument('simulated_ints', type=str, help='''A directory of simulated scattering intensities generated by CRYSOL. The files should follow a specific naming scheme: i00.int, where "i" is the microstate number.''')
    parser.add_argument('initial_weights', type=str, help='''A single column text file with the initial weights from which you want to begin the optimization.''')
    parser.add_argument('restart_weights', type=str, help='''A set of weights from which to begin the optimization, but using initial_weights to calculate the penalty function. Useful if you want to resume an optimization you've stopped.'
'', default='-1')
    parser.add_argument('temp', type=str, help='''Monte Carlo temperature parameter. Setting this higher will mean more unfavorable steps in the simulated annealing process will be accepted. A higher temperature allows easier escape from 
local minima at the expense of speed of convergence.''', default=10 ** -5)
    parser.add_argument('theta', type=str, help='''Regularization parameter. The higher this value is, the more the system will resist moving from away from "initial_weights". Setting to 0 will give best possible fit to experiment but at 
risk of overfitting.''')
    parser.add_argument('output', type=str, help='''Where to output the files "prior_profile" and "prior_weights".''', default='.')
    args = parser.parse_args()
    
    intensities = prepare_crysol_data(args.simulated_ints) # Intensities of all crysol profiles in 1 2D array
    q = get_q('%s/000.int'%(args.simulated_ints))
    exp = np.loadtxt(args.exp_ints) # Reference spectrum
    exp_ints = exp[:,1]
    exp_axis = exp[:,0]
    exp_error = exp[:,2]
    data = np.vstack((q, intensities[0])).transpose()
    used_intensities = np.array([interpolate(data, exp_axis)])
    for profile in intensities[1:]:
        data = np.vstack((q, profile)).transpose()
        used_intensities = np.vstack((used_intensities, [interpolate(data, exp_axis)]))
    # Pick initial weights
    initial_weights = np.loadtxt(args.initial_weights)
    # Pick restart weights, if desired.
    if args.restart_weights != '-1':
        restart_weights = np.loadtxt(args.restart_weights)
    elif args.restart_weights == '-1':
        restart_weights = '-1'
    # Execution of the actual code
    try:
        os.mkdir(args.output)
    except OSError:
        pass
    os.chdir(args.output)
    optimize_weights(initial_weights, used_intensities, exp_ints, exp_error, 0, float(args.temp), float(args.theta), restart_weights)
