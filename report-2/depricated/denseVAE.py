import os
import tensorflow as tf

# Trun of tensorflow warnings
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class DenseVAE(tf.keras.Model):
    """
    Class for a 1D VAE using fully connected sense layers for auto-encoding implemented in tensorflow.
    Class is adapted from tensorflow example given in https://www.tensorflow.org/tutorials/generative/cvae.
    """

    def __init__(self, inputDims, latentDims, featureDims=32, activation="leaky"):
        
        # Initialise parent class
        super(DenseVAE, self).__init__()
        
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
            tf.keras.layers.Dense(self.latentDims+self.latentDims)
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

    @tf.function
    def generateSample(self, epsilon=None):
        """
        Generate a sample using the decoder. Allows for the use of the reparametrasation trick by defining an epsilon to 
        create random latent vector while not breaking backpropogation chain.
        """
        # Randomly generate epsilon if not provided
        if epsilon is None:
            epsilon = tf.random.normal(shape=(100, self.latent_dim))
        
        # Generate output 
        output = self.decode(epsilon, applySigmoid=True)
        return output

    def encode(self, x):
        """
        Return mean and variance of the encoded latent vector
        """
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterise(self, mean, logvar):
        """
        Use the mean and variance of the encoded latent vector to create a reparametrised latent vector
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

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
        mean, logvar = self.encode(input)
        z = self.reparameterise(mean, logvar)
        output = self.decode(z)       
        return output  


def test():
    # Test class
    input_dims = 39
    latent_dims = 3
    batch_size = 7
    input = tf.random.normal(shape=(batch_size, input_dims))

    model = DenseVAE(input_dims, latent_dims)

    mean, logvar = model.encode(input)
    z = model.reparameterise(mean, logvar)
    output = model.decode(z)

    output = model(input)

    print(input.shape)
    print(mean.shape)
    print(z.shape)
    print(output.shape)

if __name__ == "__main__":
    test()
