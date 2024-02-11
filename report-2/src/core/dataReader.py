import os
import numpy as np
import pandas as pd
import time

class DataReader(object):
    """
    This is a class used to read raw csv files and return a well formatted numpy array containing all the numerical data to be used 
    for training. For more info, read main jupyter notebook.
    """
    def __init__(self, filename, cwd, numParticles, loadProcessed=False, saveProcessed=False, delimiter=";",):

        # Define class parameters
        self.filename       = filename
        self.cwd            = cwd 
        self.delimiter      = delimiter
        self.numParticles   = numParticles 
        self.dataset        = None 

        # Define class constants
        self.saveProcessedDirectory = self.cwd + "processed_datasets/"
        self.initKeys               = ["event_id", "process_id", "event_weight", "met", "met_phi",]
        self.dropMetadataKeys       = ["event_id", "process_id", "event_weight",]
        self.numParticleKeys        = ["n_photons", "n_bjets", "n_jets", "n_muon", "n_ele"]
        self.particleObjectTypes    = ["g", "b", "j", "m", "e"]
        self.kinematicParamsKeys    = ["object", "energy", "trans_momentum", "pseudo_rapidity", "phi"]

        # Read preprocessed dataset from csv if required
        if loadProcessed:
            self.loadProcessedDataset()
        else:
            # Parse csv and process dataset
            self.buildProcessedDataset()
            # Save dataset
            if saveProcessed:
                self.saveProcessedDataset()

    def _flattenList(self, listName):
        """
        Returns a flattened version of an inputed 2D list
        """
        return [x for xs in listName for x in xs]

    def _buildInitialDataset(self, datasetEventStrings):
        """
        Iterates over raw csv string, parses event metadata and event kinematic string.
        Function builds up dataset dataframe to contain metadata and event kinematic string.
        """
        # Define empty string to containg kinematic strings of csv
        kinematicStringList = []

        # Iterate over each event in array and start to build up dictionary containing data
        for event in datasetEventStrings:
            
            # Join all particle kinematic data of event into string with no spacing inbetween elements
            kinematicString = self.delimiter.join(event[len(self.initKeys):])
            # Append kinematic string to list
            kinematicStringList.append(kinematicString)

            # Fill in first five keys of dataset
            for idx, key in enumerate(self.dataset.keys()):
                # Append kinematic string to last key of dictionary
                if "met" in key:
                    # Make sure momentum metadata is saved as a float
                    self.dataset[key].append(float(event[idx]))
                else:
                    self.dataset[key].append(event[idx])

        # Convert dictionary to dataframe, add kinematioc strings as a column and return it
        self.dataset = pd.DataFrame(self.dataset)
        self.dataset["kinematics_string"] = kinematicStringList

    def _sortEventEnergies(self, kinematicStringsSplitAll):
        """
        Sort KinematicStringsSplitAll object in order of descending energies
        """
        # Create array with only energies
        particleEnergies = np.array(kinematicStringsSplitAll[1::5], dtype=float)
        
        # Find indexes of energies in decending order
        sortedEnergyIndex = np.argsort(particleEnergies)[::-1]
        
        # Make arrey containing all indexes for sorted particles
        sortedKinematicIndex = self._flattenList([list(range(idx*5, idx*5+5)) for idx in sortedEnergyIndex])
        
        # Rearrange kinematic_string_slit all in terms of energy
        kinematicStringsSplitAll = kinematicStringsSplitAll[sortedKinematicIndex]

        return kinematicStringsSplitAll

    def _parseEventKinematics(self, kinematicStringsSplit, totalKinematicParamsKeys, maximumNumParticles):
        """
        Parse the kinematic string for each event in the datase. 
        Build up and return dictionary containing each individual kinematic parameter for each event 
        """
        # Define empty array structure inside dictionary which is of size (max_n_particles * num_kinematic_params, max_n_particles) which we will build 
        particleKinematics = {key : [] for key in totalKinematicParamsKeys}

        # Create string which will be used to fill padded elements 
        padding = ( "0," * len(self.kinematicParamsKeys) )[:-1]         # indexing is used at the end to get rid of last ',' char

        # Pad out kinematic stirngs by adding additional particle kinematic elements padded by 0 to a total of n_tot_particle elements in array
        # In order to convert to numpy array (rather than an akward array)
        for event in kinematicStringsSplit:

            # Due to way csv is set up the last element in this array is an empty string, so we remove it
            event = event[:-1]   
            # Compute how many elements we need to pad to achieve max_n_particles
            numPaddingElements = maximumNumParticles - len(event)

            # Pad element
            for _ in range(numPaddingElements): event.append(padding)

            # Split kinematic stirings further such that each elements corresponds to each indfividual kinematic parameter for each particle
            kinematicStringsSplitAll = np.array( self._flattenList( np.char.split( np.array(event, dtype=str), ",") ) )
            # kinematicStringsSplitAll = np.array(self._flattenList(kinematicStringsSplitAll))

            # Sort kinematic parameters of event based on energy of paritlce
            kinematicStringsSplitAll = self._sortEventEnergies(kinematicStringsSplitAll)

            # Append individual kinematic parameter to dictionary
            for idx, key in enumerate(totalKinematicParamsKeys):
                if "object" in key:
                    particleKinematics[key].append(str(kinematicStringsSplitAll[idx]))
                else:
                    particleKinematics[key].append(float(kinematicStringsSplitAll[idx]))
            
        return particleKinematics

    def _takeColumnLogs(self, keys):
        """
        Take log of dataset columns specified in list of keys passed as parameter 
        """

        # Iterate over all momentum keys and take natural log
        for key in keys:
            # Create new object , Take log and replace -infs with 0
            # This is done to work around pandas shenanigans
            column = np.log(self.dataset[key])
            column[np.isneginf(column)] = 0.0
            self.dataset[key] = column
    
    def _computeNumObjects(self):
        """
        Compute the number of objects in each event of dataset and add to dictionary
        """
        # Make list containing all keys for object names in dataset
        objectKeys = [key for key in self.dataset.columns if "object" in key]
        # Create a dataframe containing all particles in event in the form of a numpy char array 
        numObjects = self.dataset[objectKeys].to_numpy().astype(str)
        # Iterate over each object we want to count and add to dataset
        for objectType, key in zip(self.particleObjectTypes, self.numParticleKeys):
            # Find number of specific object for each event in dataset
            numObject = np.count_nonzero( np.char.find(numObjects, objectType)==0 , axis=1)
            # Insert array into dataset 
            self.dataset.insert(0, key, numObject)

        # Drop columns containg object names
        self.dataset = self.dataset.drop(columns=objectKeys)
        
    def readCsv(self):
        """
        Read and parse the raw data csv, create a pandas dataframe containing the metadata of the dataset
        """

        filepath = self.cwd + self.filename
        
        # Check csv path exists
        if not os.path.isfile(filepath):
            raise Exception(f"File {filepath} not found.")
        
        # Define the name for the first five keys of the dataframe
        self.dataset = {key:[] for key in self.initKeys}
        
        # Parse individual lines of csv and return them as a numpy array of strings
        datasetEventStrings = np.loadtxt(filepath, dtype=str)

        # Split datapoint in the event of the array delimited by ';'
        datasetEventStrings = np.char.split(datasetEventStrings, self.delimiter)

        # Create dataset containing metadata + unparsed kinematic string 
        self._buildInitialDataset(datasetEventStrings)

    def processKinematics(self):
        """
        Parse the raw kinematic string for each event in the dataset and include kinematic data to dataset dataframe
        """

        # Create series containg the kinematic strings of the dataset
        kinematicStrings = self.dataset["kinematics_string"]

        # Split elements in kinematic string s.t each particle's kinematics is contained to one element
        kinematicStringsSplit = np.char.split(np.array(kinematicStrings, dtype=str), self.delimiter)

        # Find maximum number of particles in an event for the entire dataset
        maximumNumParticles = max(map(len, kinematicStringsSplit))
        # Create kinematic parameter keys for each partice (up to maximum_n_particles)
        totalKinematicParamsKeys = self._flattenList( [ [self.kinematicParamsKey + str(idx+1) for self.kinematicParamsKey in self.kinematicParamsKeys]  for idx in range(maximumNumParticles) ] )

        # Get dictionaries containing parsed kinematic parameters for all events
        particleKinematics = self._parseEventKinematics(kinematicStringsSplit, totalKinematicParamsKeys, maximumNumParticles)

        # Append the existing dataset with the particle kinematics entries
        ##### IS THERE A BETTER WAY OF DOING THIS? PROBABLY. BUT EVERY PANDAS FUNCTION I TRIED TO USE WOULD NOT WORK, SO HERE WE ARE BRUTE FORCING IT #####
        for key in particleKinematics:
            if str(self.numParticles+1) in key:
                break
            self.dataset[key] = particleKinematics[key]
        
        # Drop the kinematic string column of the dataframe as it is no longer needed
        self.dataset = self.dataset.drop(columns="kinematics_string")

        # Drop any undesirable metadata which may be a part of the dataset
        for key in self.dropMetadataKeys:
            # Check if key is in dataset
            if key in list(self.dataset.columns):
                self.dataset = self.dataset.drop(columns=key)

    def buildProcessedDataset(self):
        """
        Read in raw CSV and create a dataframe containig the processed dataset
        """
        # Read and parse CSV
        self.readCsv()
        # Parse and process event kinematics
        self.processKinematics()
        # Build list of keys from dataset which will have logs calculated 
        logKeys = []
        logKeys.extend([energy_key for energy_key in self.dataset.columns if "energy" in energy_key])
        logKeys.extend([momentum_key for momentum_key in self.dataset.columns if "momentum" in momentum_key])
        logKeys.extend([met_key for met_key in self.dataset.columns if "met" == met_key])
        # Take log of logKeys columns
        self._takeColumnLogs(logKeys)
        # Compute number of objects in each event
        self._computeNumObjects()

    def loadProcessedDataset(self):
        """
        Load processed dataset csv file for easier dataset loading
        """
        filepath = self.saveProcessedDirectory + self.filename.split("/")[-1]

        # Check csv path exists
        if not os.path.isfile(filepath):
            raise Exception(f"File {filepath} not found.")

        # Load processed dataset
        self.dataset = pd.read_csv(filepath, sep=",", header=None, index_col=0, dtype=str)

        # Set column names
        self.dataset.columns = self.dataset.iloc[0]
        self.dataset = self.dataset.loc[1:]

    def saveProcessedDataset(self):
        """
        Save processed dataset as a csv file (AGAINST MY WILL) for easier dataset loading in the future
        """

        # Check if processed dataset directory exists
        if not os.path.exists(self.saveProcessedDirectory):
            # Make save directory is it doesnt already exist
            os.mkdirs(self.saveProcessedDirectory)
        
        filename = self.filename.split("/")[-1]

        # Save dataset
        self.dataset.to_csv(self.saveProcessedDirectory+filename, sep=",", index=True, header=True)

    def getDataset(self, asNumpy=True):
        """
        Return dataset object
        """
        return self.dataset.to_numpy(dtype=np.float32) if asNumpy else self.dataset


def test():

    cwd = "/home/giorgio/DAML/report-2/"
    dataloader = DataReader("raw_datasets/background_chan2b_7.8.csv", cwd, 8,)
    dataset = dataloader.getDataset()

if __name__ == "__main__":
    test()