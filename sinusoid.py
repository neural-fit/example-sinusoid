import neuralfit as nf
import matplotlib.pyplot as plt
import numpy as np

# Define the dataset
x = np.linspace(0, 2*np.pi, 1000).reshape(-1,1)
y = np.sin(x)

# Create and compile the model
model = nf.Model(inputs=1, outputs=1, size=4)
model.compile(optimizer='alpha', loss='mse', monitors=['size'])

# Evolve the model
model.evolve(x, y, epochs=1000)

# Get model predictions
y_hat = model.predict(x)

# Plot results
plt.plot(x, y, label='True', color='k', linestyle='--')
plt.plot(x, y_hat, label='Predicted',color='#52C560', linewidth=2)
plt.show()
