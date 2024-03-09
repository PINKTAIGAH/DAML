## **Exercise 1**

The ```run.mac``` file was then changed to simulate 1000 events and the GUI was turned off to speed up simulations.

The csv data files can be found in the ```data/``` directory. 

## **Exercise 2**

To find pairs of hits for each event, run the following command:

```bash
python bin/find_hit_pairs.py
```

As is to be expected, a majority of the events have 2 pairs, corresponding to the two products from the e+e- interaction. A small number of events have more pairs, corresponding to additional particles that were created prior to crossing the first tracker. Lastly, there is a small number of events that do not conain any pairs, this is likely a result of particles only crossing 1 or no tracking units of the detector.

## **Exercise 3**

To calculate the energy of each reconstructed track in the detector, run the following command:

```bash
python bin/calculate_track_momentum.py
```

## **Exercise 4**

To create a momentum resolution plot, run the following command:

```bash
python bin/plot_figures.py
```

The momentum resolution plot is found in the ```./images/momentum_resolution_hist.png```.

## **Exersice 5**

To create the mass distibution plots, run the following command:

```bash
python bin/plot_figures.py
```

From the mass distribution, we see that the mean initial energy of the interaction inside the detector was centered at $\sim 90 GeV$, which coinsides with the rest mass of the Z-boson. Hence the interaction that was simulated was likely the decay of a Z boson via NC channel int a mu+mu-.