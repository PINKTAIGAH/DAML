import os
from collections import OrderedDict
import tensorflow as tf
from tqdm import tqdm
from . import DenseVAE, DenseAE
from tensorflow.keras.metrics import mean_squared_error as MSE, kl_divergence as KL

# Trun of tensorflow warnings
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class TensorflowTrainer(object):
    """
    Class used to train and validate a given model. Currently designed to train VAEs
    """

    def __init__(self, config, trainData, valData, saveNetwork=True):

        # Define class configuration dictionary 
        self.config = config["trainer"]
        self.epochs = self.config["epochs"]
        self.beta   = self.config["beta"]

        # Initialise iterable datasets
        self.trainData  = trainData
        self.valData    = valData

        # Initialise network
        self.network = None
        if self.config["model_name"] == "denseVAE":
            self.network    = DenseVAE(config["network"]["input_dims"], config["network"]["latent_dims"]) 
        if self.config["model_name"] == "denseAE":
            self.network    = DenseAE(config["network"]["input_dims"], config["network"]["latent_dims"]) 
        if self.network is None:
            raise Exception("YOU FUCKED UP")

        # Initialise optimiser
        if self.config["optimiser"]["name"] == "adam":
            self.optimiser  = tf.keras.optimizers.Adam(
                learning_rate=self.config["learning_rate"], beta_1=self.config["optimiser"]["moments"]
            )
        else: 
            tf.keras.optimizers.SGD(
                learning_rate=self.config["learning_rate"]
            )
        
        # Initialise loss 
        self.lossFunction = self.denseVAELoss if self.config["model_name"] == "denseVAE" else None

        # # Initialise tensorboard writer
        # trainLogDir                 = self.config["output_dir"] + self.config["run_id"] + '/logs/train'
        # valLogDir                   = self.config["output_dir"] + self.config["run_id"] + '/logs/val'
        # self.train_summary_writer   = tf.summary.create_file_writer(trainLogDir)
        # self.val_summary_writer     = tf.summary.create_file_writer(valLogDir)

        # Initialise metric dictionary
        self.metrics = {
            "val" : {
                "loss" : []
            },
            "train": {
                "loss" : []
            }
        }

    def denseVAELoss(self, xTruth, xGenerated):
        """
        Loss function consists of (1-beta)*MSE + beta*KL 
        """
        mseLoss = MSE(xTruth, xGenerated)
        return mseLoss*100 

    def trainStep(self, input):
        """
        Execute a single training step
        """

        # Perform forward step
        with tf.GradientTape() as tape:
            # Compute network output
            output = self.network(input)

            # Compute loss function
            loss = self.lossFunction(input, output)
        
        # Preform backwards step
        gradients = tape.gradient(loss, self.network.trainable_variables) 
        # Apply gradients to model
        self.optimiser.apply_gradients(zip(gradients, self.network.trainable_variables))
        # print(loss)
        # Return loss value for step
        return loss

    def validationStep(self, input):
        """
        Execute a single training step
        """
        # Perform validation step
        # Compute network output
        output = self.network(input)

        # Compute loss function
        loss = self.lossFunction(input, output)
        # print(loss)
        return loss
    
    def printResults(self, epoch, loss, mode):
        """
        Print out results to console through tqdm
        """

        print(f"EPOCH: {epoch} {'#'*5} MODE: {mode} {'#'*5} LOSS: {loss}")    

    def logMetrics(self, trainLoss, valLoss):
        """
        Log metrics to metric dictionary
        """
        # Append Metrics  
        self.metrics["train"]["loss"].append(trainLoss)      
        self.metrics["val"]["loss"].append(valLoss)      
    
    def runBatchProcess(self,):
        """
        Execute the main training loop
        """

        # Iterate per epoch
        for epoch in range(self.epochs):
            
            trainIter   = iter(self.trainData)
            valIter     = iter(self.valData)
            
            print(f"##### EPOCH {epoch+1} #####")

            # Define variable to hold running loss
            runningTrainLoss    = 0.0
            runningValLoss      = 0.0
            
            # Run train step
            for input in trainIter:
                # Compute iteration metrics
                runningTrainLoss += self.trainStep(input).numpy().mean()

            # Compute overall training loss
            trainLoss = runningTrainLoss/self.trainData.cardinality().numpy()
            # Print Training step results
            self.printResults(epoch+1, trainLoss, "Train")

            # Run Validation step
            for input in valIter:
                # Compute iteration metics
                runningValLoss += self.validationStep(input).numpy().mean()

            # Compute overall validation loss
            valLoss = runningValLoss/self.valData.cardinality().numpy()
            # Print validation step results
            self.printResults(epoch+1, valLoss, "Validation")
            # Log metrics
            self.logMetrics(trainLoss, valLoss)