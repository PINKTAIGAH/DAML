import torch 
import torch.nn as nn

class DenseVAE(nn.Module):
    """
    Class for a 1D AE using fully connected sense layers for auto-encoding implemented in tensorflow.
    Class is adapted from tensorflow example given in https://www.tensorflow.org/tutorials/generative/cvae.
    """

    def __init__(self, inputDims, latentDims, device, featureDims=32, activation="leaky"):
        
        # Initialise parent class
        super(DenseVAE, self).__init__()
        # Define device 
        self.device = device
        # Define class parameters
        self.latentDims = latentDims
        self.inputDims  = inputDims
        # Define encoder
        self.encoder = nn.Sequential(
            # First dense block
            nn.Linear(self.inputDims, featureDims),
            nn.LeakyReLU(0.2) if activation=="leaky" else nn.GELU(),
            # Second dense block
            nn.Linear(featureDims, int(featureDims/2)),
            nn.LeakyReLU(0.2) if activation=="leaky" else nn.GELU(),
            # Third dense layer w/ output being latent dims. This block will have no activation func
            nn.Linear(int(featureDims/2), self.latentDims),
        )

        # Define mu and logvar layers
        self.meanLayer = nn.Linear(self.latentDims, 2)
        self.logvarLayer = nn.Linear(self.latentDims, 2)

        # Define decoder
        self.decoder = nn.Sequential(
            # Take z to latent dim
            nn.Linear(2, self.latentDims),
            nn.LeakyReLU(0.2) if activation=="leaky" else nn.GELU(),
            # First dense block
            nn.Linear(self.latentDims, int(featureDims/2),),
            nn.LeakyReLU(0.2) if activation=="leaky" else nn.GELU(),
            # Second dense block
            nn.Linear(int(featureDims/2), featureDims,),
            nn.LeakyReLU(0.2) if activation=="leaky" else nn.GELU(),
            # Third dense layer w/ output being latent dims. This block will have no activation func
            nn.Linear(featureDims, self.inputDims,),
        )

    def encode(self, x):
        """
        Return meana nd variance of the encoded latent vector
        """
        latentVector = self.encoder(x)
        mu, logvar = self.meanLayer(latentVector), self.logvarLayer(latentVector)
        return mu, logvar

    
    def reparameterise(self, mean, logvar):
        """
        Use the mean and variance of the encoded latent vector to create a reparametrised latent vector
        """
        epsilon = torch.randn_like(logvar).to(self.device)
        z = epsilon * logvar  + mean
        return z

    def decode(self, z, apply_sigmoid=False):
        """
        Decode the encoded latent vector
        """
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = nn.functional.sigmoid(logits)
            return probs
        return logits

    def forward(self, input):
        """
        Run block of code when object is called 
        """
        mu, logvar  = self.encode(input)
        z           = self.reparameterise(mu, logvar)
        output      = self.decode(z)

        return output, mu, logvar


def test():
    # Test class
    input_dims = 39
    latent_dims = 4
    batch_size = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.rand(size=(batch_size, input_dims)).to(device)

    model = DenseVAE(input_dims, latent_dims, device).to(device)

    output, mu, logvar = model(input)

    print(input.shape)
    print(output.shape)
    print(mu.shape)
    print(logvar.shape)

if __name__ == "__main__":
    test()
