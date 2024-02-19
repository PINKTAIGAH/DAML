To run the script that leads to the answer to question 1: 

WHILE IN THE ./GeantExample2Part1 DIRECTORY, run the following command:
    python bin/ViewEnergy.py

From the histogram we see what we expected, with less energy deposits for the EM showers in the deeper layers 
of the detector. Tis is die to the fact that hadronic showers tend to penetrate deeper in meduium than EM showers

#########################################################################

To run the script that leads to the answer to question 2: 

WHILE IN THE ./GeantExample2Part1 DIRECTORY, run the following command:
    python bin/CalibrateEnergy.py

We see from the historgram of the detector energy percentage difference distributuion a clear resolution of 0.031

#########################################################################

To run the script that leads to the answer to question 3: 

WHILE IN THE ./GeantExample2Part1 DIRECTORY, run the following command:
    python bin/Plot2dHist.py

From the 2d historgram plot, we see a relativly consistant energy resolution for all MC trith energies simulated.
We see the best calibration for the simulation at higher energies compared to the lower ones, where the mean is
not centered at 0. Additionally,. we see a relativly equal energy resolution at all energies.

#########################################################################

To run the script that leads to the answer to question 4: 

WHILE IN THE ./GeantExample2Part2 DIRECTORY, run the following command:
    ./build/MyProgram

The number of positrons are printed out into the console

#########################################################################

To run the script that leads to the answer to question 5: 

WHILE IN THE ./GeantExample2Part2 DIRECTORY, run the following command:
    python bin/ComputeEnergyFraction.py

We see that approximately 3/4 of the energy deposited within the detector is due to electrons, which is to be
expected from incoming electorns

#########################################################################

PNG versions of all the figures for each excersice can be found in the ./images directory

The fraction for E_electon/E_tot_deposited was found to be 0.750