This repository contains code for calculating the anharmonic infrared (IR) spectrum of polycyclic aromatic hydrocarbons (PAHs) using a machine learning-based molecular dynamics (MLMD) approach. The method employs two distinct machine learning (ML) models:
Neural Network Force Field (NNFF) [doi:10.1021/acs.jcim.1c01380]: Used to construct the potential energy surface.
Electron Passing Neural Network (EPNN) [doi:10.1021/acs.jcim.0c01071]: Used to predict the molecular dipole moment.

Overview of the Process:
To compute the anharmonic IR spectrum, molecular dynamics (MD) simulations are performed to obtain atomic configurations (trajectories) during molecular vibrations. These configurations are generated using atomic forces predicted by the NNFF model. The resulting atomic trajectories are then used by the EPNN model to calculate the dipole moments at each time step. The dipole time-autocorrelation function is subjected to a Fourier transform to derive the IR intensity.

Both the NNFF and EPNN models are pre-trained and ready for immediate use, so no additional machine learning training is required. 

For more information on the training of these models and the MD simulations, please refer to the following article:

DOI link to article

Authors: Xinghong Mai, Zhao Wang, Lijun Pan, XX, XX, Jesus Carrete, and Georg K. H. Madsen

The code is available at: https://github.com/zwAstroChem/MLMDirPAH

Citation: 
If you use this code to generate data, please cite the article and acknowledge the code in your publications. For inquiries, you can contact Prof. Zhao Wang at zw[at]gxu.edu.cn.


---------------------------
Instructions
---------------------------

I. Installation with Anaconda

1. Create and activate a virtual environment:

conda create -n MLMD_env python=3.10.13 # Create a virtual environment

conda activate MLMD_env # Activate the virtual environment

2. Install the required packages:
a) Navigate to the lib/learned_optimization directory and install the optimizer:

pip install -e .

b) Install NNFF:

cd lib/bessel-nn-potentials-velo

pip install -e . 

c) Install EPNN:

cd lib/epnn-main

pip install -e . 

3. Install additional libraries:

pip install flax==0.7.4

pip install optax==0.1.7

pip install orbax-checkpoint==0.4.1

pip install numpy==1.26.1

pip install scipy==1.11.3

pip install chex==0.1.84

pip install oryx==0.2.7

Note: You may encounter version incompatibility warnings, which can be safely ignored.

4. Install JAX and JAXLib (CPU version):

pip install jax[cpu]==0.4.19 -f https://storage.googleapis.com/jax-releases/jax_releases.html

pip install jaxlib==0.4.19 -f https://storage.googleapis.com/jax-releases/jax_releases.html

Note: Note: If version incompatibilities occur, you can ignore them. For specific library versions, refer to the environment.yaml file. After installation, use pip list to confirm JAX and JAXLib are both version 0.4.19. If not, repeat the installation Step 4.

II. Running the Program

1. Data Preparation:
Place the .xyz files for the molecules you wish to compute in the ./inputs/XYZ/ directory.
Example: C10H8_330.xyz (from the original dataset).
The .xyz file format consists of:
Line 1: Total number of atoms.
Line 2: Comment line.
Lines 3 and beyond: Each line contains the atom type (e.g., C for carbon) and its Cartesian coordinates (x, y, z) in Angstrom.

2. MLMD Calculations
On Linux:

Modify the TEMPERATURE variable on line 4 of code/run_calc.sh to set the desired simulation temperature.

Run the script: 
./code/run_calc.sh

This script performs the following: 

a) MD Simulations: The script runs code/1_MD_calc_position.py to perform MD simulations, generating atomic trajectories at the specified temperature. Progress bars will indicate the current molecule being processed and the simulation phase (NVT and NVE). The simulation for molecules like C10H8 takes around 16 hours.

b) Dipole Moment Calculation: The script runs code/2_calc_dipole.py to calculate time-evolved dipole moments based on atomic trajectories. Progress bars will display the progress for 400,000 time steps. For molecules like C10H8, this step takes approximately 32 hours.

c) IR Spectrum Calculation: The script runs code/3_calc_IR.py to compute the IR spectrum using a Fourier transform of the dipole time-autocorrelation function. Each molecule's IR spectrum calculation typically completes in a few seconds.

On Windows:
Navigate to the code/ directory and run the following commands:

python 1_MD_calc_position.py 50  # Set temperature to 50 K

python 2_calc_dipole.py 50 

python 3_calc_IR.py 50 

Each python program will process each .xyz file in the ./inputs/XYZ/ directory one by one.

III. Output Files

1. IR Spectrum: 
The computed IR spectrum will be saved in the ./outputs/IR_txt/ folder.

Example: ./outputs/IR_txt/C10H8_330.txt contains two columns:
Column 1: Wavenumber (Freq_MD) in cm^-1
Column 2: IR absorption intensity (Inten_MD) in 10^5 cm^2 mol^-1.
A visualization of the IR spectrum is saved as C10H8_330.jpg, with the X-axis representing the wavenumber (cm^-1 and the Y-axis representing the IR absorption intensity (10^5 cm^2 mol^-1).

2. Intermediate Results
   
Atomic Trajectories: Saved in intermediate_results/equilibration/ (for NVT) and intermediate_results/VelocityVerlet/ (for NVE). Each line includes: Step Number: The timestep number. Total Energy: The total energy of the molecule at this timestep, in electron volts (eV). Temperature: The temperature at this timestep, in Kelvin (K). Atomic positions (X, Y, Z) for each atom.

Dipole Moments: Saved in intermediate_results/Dipole_txt_mass/, with each row containing the dipole moment components (dipole_x, dipole_y, dipole_z) at a specific time step. The dipole moment components are given in units of e*Angstrom.


---------------------------
Limitations
---------------------------

The model is currently trained only on neutral PAHs and does not support charged molecules or isotopologues. Predictions for molecules that significantly differ from the training dataset may have increased uncertainty. We are working to address these limitations and will continue to update the code on GitHub.

---------------------------
Additional Contents
---------------------------
This repository also includes spectral data for 2,025 theoretically calculated and 49 experimentally tested PAHs.

---------------------------
License
---------------------------
This code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. You may use, share, and adapt the code for educational, research, and non-commercial purposes, provided appropriate credit is given, a link to the license is provided, and any changes are indicated. Contributions must be shared under the same license.
