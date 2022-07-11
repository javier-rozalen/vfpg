# -*- coding: utf-8 -*-
####################### IMPORTANT INFORMATION #######################
"""
This program computes the GS density of a 1D quantum harmonic oscillator (HO).
It uses the MCMC approach to path integrals. 
The following parameters can be adjusted manually:
    # GENERAL PARAMETERS
    seed --> int, Random seed to be used.
    N --> int, Number of points of each path.
    M --> int, Total number of paths that we want to generate.
    mu --> float, Mean of the step-proposing PDF in Metropolis.
    sigma --> float, Standard deviation of the step-proposing PDF in Metropolis.
    d --> float, Step size.
    m --> float, Mass of the particle.
    w --> float, Frequency of the HO.
    n_faulty --> int, Number of paths which are discarded at the beginning.
    T --> float, Final time of the path.
    hbar --> float, Value for hbar.
    action --> function, Action to use. Default is that of the HO.
    
    PLOTTING PARAMETERS
    leap --> Number of accepted paths after which to plot the current progress.
    
    SAVES/BOOLEANS
    metropolis --> Boolean, Whether to run the calculation or not.
    save_wf --> Boolean, Whether to save the wave function data or not.
    write_data --> Boolean, Whether to write the paths and actions to a file or 
                    not.
    save_plot --> Boolean, Whether to save the final plot or not.
                        
"""

################################# IMPORTS #####################################
# Change to the directory of this script
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

# General modules
import numpy as np
from tqdm import tqdm
import scipy.integrate as integrate

# My Modules
from modules.actions import S_HO, S_double_well
from modules.plotters import nice_plot
from modules.dir_support import dir_support
from modules.aux_functions import histogram

########################### GENERAL PARAMETERS ##########################
# General parameters
seed = 1
N = 20
M = 10000
mu = 0
sigma = 1/6
d = 1.
m = 1
w = 1
n_faulty = 300
T = 100
hbar = 1.
action = S_HO

# Plotting parameters
leap = M/5

# Saves/Booleans
save_wf = True
write_data = True
save_plot = True

# Not adjustable
dx = 0.1
paths_file = f'saved_data/paths/paths_N{N}_M{M}.txt'
actions_file = f'saved_data/actions/actions_N{N}_M{M}.txt'
wf_file = f'saved_data/wave_functions/wf_N{N}_M{M}.txt'
plot_file = f'saved_data/plots/plot_N{N}_M{M}.pdf'
    
##################### AUXILIARY STUFF #####################
x_axis = np.linspace(-4.95, 4.95, 100)
y_target = (((1/(np.pi*1**2))**(1/4))*np.exp(-x_axis**2/(2*1**2)))**2

np.random.seed(seed)
h = T/N

x0 = [0.]*N
x0 = np.random.normal(0, 1, N).tolist()
paths = [x0]
S_paths = [action(x0, h, m, w)]
wf = np.array([0.]*100)
    
################## METROPOLIS ####################
k = 0
n_accepted = 1
pbar = tqdm(total=M)
if write_data: 
    dir_support(['saved_data','paths'])
    dir_support(['saved_data','actions'])
    with open(paths_file, 'w') as file:
        with open(actions_file, 'w') as file2:
            while n_accepted < M:
                chi = np.random.normal(mu, sigma, N)
                path_old = paths[-1]
                path_new = path_old + d*chi
                path_new[-1] = path_new[0]
                S_new = action(path_new, h, m, w)
                S_old = action(path_old, h, m, w)
                delta_S = S_new - S_old
                
                if delta_S <= 0:
                    accepted = True
                else:
                    r = np.random.rand(1)
                    if r<np.exp(-delta_S):
                        accepted = True
                    else:
                        accepted = False
                  
                if accepted:
                    n_accepted += 1
                    paths.append(path_new)
                    S_paths.append(S_new)
                    if write_data:
                        file.write(' '.join([str(x) for x in path_new]) + '\n')
                        file.write(' '.join([str(x) for x in -path_new]) + '\n')
                        file2.write(str(S_new) + '\n')
                        file2.write(str(S_new) + '\n')
                    pbar.update(1)
                    if n_accepted > n_faulty:
                        wf += histogram(path_new, dx) + histogram(-path_new, dx)
                        wf_norm = integrate.simpson(y=wf, x=x_axis)
                        counts = histogram(path_new, dx)
                        if n_accepted % leap == 0:
                            nice_plot(x_axis,wf/wf_norm, S_paths, n_accepted,
                                   path_new)
                        if n_accepted == M-1:
                            nice_plot(x_axis, wf/wf_norm, S_paths, n_accepted,
                                   path_new, plot_file, save=False)
                            x = []
                            for i in range(100):
                                for j in range(counts[i]):
                                    x.append(x_axis[i])
                                    
              
                k += 1
        pbar.close()
        file.close()
        file2.close()
else:
    while n_accepted < M:
        chi = np.random.normal(mu, sigma, N)
        path_old = paths[-1]
        path_new = path_old + d*chi
        path_new[-1] = path_new[0]
        S_new = action(path_new, h, m, w)
        S_old = action(path_old, h, m, w)
        delta_S = S_new - S_old
        
        if delta_S <= 0:
            accepted = True
        else:
            r = np.random.rand(1)
            if r<np.exp(-delta_S):
                accepted = True
            else:
                accepted = False
          
        if accepted:
            n_accepted += 1
            paths.append(path_new)
            S_paths.append(S_new)
            pbar.update(1)
            if n_accepted > n_faulty:
                wf += histogram(path_new, dx) + histogram(-path_new, dx)
                wf_norm = integrate.simpson(y=wf, x=x_axis)
                counts = histogram(path_new, dx)
                if n_accepted % leap == 0:
                    nice_plot(x_axis,wf/wf_norm, S_paths, n_accepted,
                           path_new)
                if n_accepted == M-1:
                    nice_plot(x_axis, wf/wf_norm, S_paths, n_accepted,
                           path_new, plot_file, save=False)                                
      
        k += 1
pbar.close()

if save_wf:
    dir_support(['saved_data', 'wave_functions'])
    with open(wf_file, 'w') as file:
        for x, wf2 in zip(x_axis, wf/wf_norm):
            file.write(f'{x} {wf2}\n')
        file.close()
    print('Wave function data correctly saved.')
    
if save_plot:
    dir_support(['saved_data','plots'])
    nice_plot(x_axis, wf/wf_norm, S_paths, n_accepted,
           path_new, plot_file, save=True)


#%% ############ <X> #############
"""
x,y = [],[]
with open('saved_data/wf_N{N}_M{M}.txt', 'r') as file:
    for line in file.readlines():
        line = line.split(' ')
        x.append(float(line[0]))
        y.append(float(line[1]))
    file.close()
x, wf = np.array(x), np.array(y)
wf_norm = integrate.simpson(y=wf, x=x)

E_X = integrate.simpson(y=wf*x/wf_norm, x=x)
E_X2 = integrate.simpson(y=wf*x**2/wf_norm, x=x)
E_X3 = integrate.simpson(y=wf*x**3/wf_norm, x=x)
E_X4 = integrate.simpson(y=wf*x**4/wf_norm, x=x)
E = m*w**2*E_X2
print('\n')
print(f'<X> = {E_X}')
print(f'<X²> = {E_X2}')
print(f'<X³> = {E_X3}')
print(f'<X⁴> = {E_X4}')
print(f'<E> = {E}')
"""









    


