# A Generative-Model Approach to path integrals

## Requirements
The machine learning part of the code in the files above is written in PyTorch. It does not come with the default Python 3 installation; to install it, go to [Official PyTorch page](https://pytorch.org/get-started/locally/) or type:

`pip3 install torch`

Also, the progress bar `tqdm` is used. To install it:

`pip3 install tqdm` 

Finally, the `numpy` library:

`pip3 install numpy`

## Usage guide
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Step 1. Generating paths with MCMC.
We open the file MonteCarlo/main_mcmc.py, set the desired initial parameters and run the file. An explanation of the adjustable parameters can be found at the beggining of the file. If the saving parameters were set to `True`, the program will save the data under the MonteCarlo/saved_data/ folder (created automatically). 

Example of the results:
![plot](./example_plots/MCMC_N20_M10000.png?raw=true)

### Step 2. Training the VAE.
We repeat the process of Step 1, but this time with the file vae/main_vae.py. This will train a VAE using the paths generated in Step 1 and, if desired, save the model for posterior experiments.

### Step 3. Generating paths with VAE.
Once we have some generated data, we go to the vae/sampling_from_vae.py file, set the desired initial parameters and run the file. Again, an explanation of the adjustable parameters can be found at the beggining of the file. This will plot a ground-state wave function computed with VAE-generated paths, along with some of these paths. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Sample plots of the trained VAE can be found under the vae/saved_data/ folder. 
