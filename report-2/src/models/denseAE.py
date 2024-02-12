import torch 
import torch.nn as nn

class DenseAE(nn.Module):
    """
    Class for a 1D VAE using fully connected sense layers for auto-encoding implemented in tensorflow.
    Class is adapted from tensorflow example given in https://www.tensorflow.org/tutorials/generative/cvae.
    """

    def __init__(self, inputDims, latentDims, featureDims=32, activation="leaky"):
        
        # Initialise parent class
        super(DenseAE, self).__init__()
        # Define class parameters
        self.latentDims = latentDims
        self.inputDims  = inputDims
        # Define encoder
        self.encoder = nn.Sequential(
            # First dense block
            nn.Linear(self.inputDims, featureDims),
            nn.LeakyReLU(0.2) if activation=="leaky" else nn.ReLU(),
            # Second dense block
            nn.Linear(featureDims, int(featureDims/2)),
            nn.LeakyReLU(0.2) if activation=="leaky" else nn.ReLU(),
            # Third dense layer w/ output being latent dims. This block will have no activation func
            nn.Linear(int(featureDims/2), self.latentDims),
        )

        # Define decoder
        self.decoder = nn.Sequential(
            # First dense block
            nn.Linear(self.latentDims, int(featureDims/2),),
            nn.LeakyReLU(0.2) if activation=="leaky" else nn.ReLU(),
            # Second dense block
            nn.Linear(int(featureDims/2), featureDims,),
            nn.LeakyReLU(0.2) if activation=="leaky" else nn.ReLU(),
            # Third dense layer w/ output being latent dims. This block will have no activation func
            nn.Linear(featureDims, self.inputDims,),
        )

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
            probs = nn.Sigmoid(logits)
            return probs
        return logits

    def forward(self, input):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.rand(size=(batch_size, input_dims)).to(device)

    model = DenseAE(input_dims, latent_dims).to(device)

    output = model(input).to(device)

    print(input.shape)
    print(output.shape)

if __name__ == "__main__":
    test()
