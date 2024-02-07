import os
import tensorflow as tf

# Trun of tensorflow warnings
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class DenseAE(tf.keras.Model):
    """
    Class for a 1D AE using fully connected sense layers for auto-encoding implemented in tensorflow.
    Class is adapted from tensorflow example given in https://www.tensorflow.org/tutorials/generative/cvae.
    """

    def __init__(self, inputDims, latentDims, featureDims=32, activation="leaky"):
        
        # Initialise parent class
        super(DenseAE, self).__init__()
        
        # Define class parameters
        self.latentDims = latentDims
        self.inputDims  = inputDims

        # Define encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.inputDims,)),
            # First dense block
            tf.keras.layers.Dense(featureDims),
            tf.keras.layers.LeakyReLU(0.2) if activation=="leaky" else tf.keras.layers.ReLU(),
            # Second dense block
            tf.keras.layers.Dense(featureDims/2),
            tf.keras.layers.LeakyReLU(0.2) if activation=="leaky" else tf.keras.layers.ReLU(),
            # Third dense layer w/ output being latent dims. This block will have no activation func
            tf.keras.layers.Dense(self.latentDims)
        ])

        # Define decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latentDims,)),
            # Second dense block
            tf.keras.layers.Dense(featureDims/2),
            tf.keras.layers.LeakyReLU(0.2) if activation=="leaky" else tf.keras.layers.ReLU(),
            # Second dense block
            tf.keras.layers.Dense(featureDims),
            tf.keras.layers.LeakyReLU(0.2) if activation=="leaky" else tf.keras.layers.ReLU(),
            # Third dense layer w/ output being input dims. This block will have no activation func
            tf.keras.layers.Dense(self.inputDims)
        ])

    def encode(self, x):
        """
        Return meana nd variance of the encoded latent vector
        """
        latentVector = self.encoder(x)
        return latentVector

    def decode(self, z, apply_sigmoid=False):
        """
        Decode the encoded latent vector
        """
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, input):
        """
        Run block of code when object is called 
        """
        latentVector    = self.encode(input)
        output          = self.decode(latentVector)

        return output


def test():
    # Test class
    input_dims = 39
    latent_dims = 4
    batch_size = 7
    input = tf.random.normal(shape=(batch_size, input_dims))

    model = DenseAE(input_dims, latent_dims)

    output = model(input)

    print(input.shape)
    print(output.shape)

if __name__ == "__main__":
    test()
