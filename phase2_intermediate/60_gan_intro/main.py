# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate synthetic data using make_moons
# WHY: To demonstrate GANs on a simple, non-linearly separable dataset
X, y = make_moons(n_samples=1000, noise=0.05)

# Plot the data
# WHY: To visualize the dataset and understand its structure
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Moons Dataset")
plt.show()

# Define the generator and discriminator functions
# WHY: These are the core components of a GAN
def generator(z):
    # Simple linear transformation
    return 2 * z + 1

def discriminator(x):
    # Simple linear transformation
    return x / 2 - 0.5

# Train the generator and discriminator
# WHY: To demonstrate the basic idea of GANs
# NOTE: This is a highly simplified example and not a real GAN implementation
z = np.random.uniform(-1, 1, size=(100, 1))
generated_data = generator(z)
discriminator_output = discriminator(generated_data)

# Print the discriminator output
# WHY: To verify the discriminator's behavior
print("Discriminator output:", discriminator_output[:5])

# Plot the generated data
# WHY: To visualize the generated samples
plt.scatter(generated_data[:, 0], generated_data[:, 0], c='r')
plt.title("Generated Data")
plt.show()

if __name__ == "__main__":
    print("GAN intro demo completed.")
    print("Discriminator output (first 5 samples):", discriminator_output[:5])