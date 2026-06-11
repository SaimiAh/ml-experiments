import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Generate synthetic data (two moons)
def generate_data():
    # Create data
    X, _ = make_moons(n_samples=100, noise=0.05)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

# Simple Variational Autoencoder (VAE) class
class VAE:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Initialize weights
        self.weights_mean = np.random.rand(input_dim, latent_dim)
        self.weights_log_var = np.random.rand(input_dim, latent_dim)
        
    def encode(self, x):
        # Compute mean and log variance
        z_mean = np.dot(x, self.weights_mean)
        z_log_var = np.dot(x, self.weights_log_var)
        
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        # Sample from standard normal
        eps = np.random.randn(*z_mean.shape)
        
        # Compute latent variable
        z = z_mean + np.exp(z_log_var / 2) * eps
        
        return z
    
    def decode(self, z):
        # Compute reconstructed input
        x_recon = np.dot(z, self.weights_mean.T)
        
        return x_recon

# Main function
if __name__ == "__main__":
    # Generate and scale data
    X = generate_data()
    
    # Define VAE model
    vae = VAE(input_dim=2, latent_dim=1)
    
    # Encode and reparameterize
    z_mean, z_log_var = vae.encode(X)
    z = vae.reparameterize(z_mean, z_log_var)
    
    # Decode
    X_recon = vae.decode(z)
    
    # Print reconstruction error
    print("Reconstruction Error:", np.mean((X - X_recon) ** 2))
    
    # Plot original and reconstructed data
    plt.scatter(X[:, 0], X[:, 1], label="Original")
    plt.scatter(X_recon[:, 0], X_recon[:, 1], label="Reconstructed")
    plt.legend()
    plt.show()