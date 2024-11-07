import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense 

# Generator Model
def define_generator(latent_dim, output_dim=1):
  model = Sequential()
  model.add(InputLayer(input_shape=(latent_dim,)))  # Use `shape` argument for input definition
  model.add(Dense(10, activation='relu'))
  model.add(Dense(output_dim, activation='linear'))
  return model

# discriminator Model
def define_discriminator(input_dim=1):
    model=Sequential()
    model.add(InputLayer(input_shape=(input_dim,)))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

# GAN Model:
def define_gan(generator,discriminator):
    discriminator.trainiable=False
    model=Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss="binary_crossentropy",optimizer='adam')
    return model

# Generate real samples
def generate_real_samples(n):
    x=np.random.randn(n)
    y=np.ones((n,1))
    return x,y

# Generate Latent points as input for generator
def generate_latent_points(latent_dim, n):
    return np.random.randn(n, latent_dim)
    
# Train the GAN Model
def train_gan(generator, discriminator, gan_model, latent_dim ,n_epochs=100, n_batch=128):
    half_batch=int(n_batch / 2)

    for epoch in range(n_epochs):
        # Generate real samples with half samples+
        x_real, y_real = generate_real_samples(half_batch)

        # Train discriminator on real samples
        d_loss_real,_ = discriminator.train_on_batch(x_real, y_real)

        # Generate fake samples with half batch
        x_fake = generate_latent_points(latent_dim, half_batch)
        y_fake = np.zeros((half_batch, 1))

        # Train discriminator on fake samples
        d_loss_fake, _ = discriminator.train_on_batch(generator.predict(x_fake), y_fake)

        # Generate latent points as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)

        # Generate labels for generated samples(real ones)
        y_gan = np.ones((n_batch, 1))

        # Train the generator via the discrininator error 
        g_loss = gan_model.train_on_batch(x_gan, y_gan)

        # print progress
        if (epoch + 1) % 2 == 0:
            print(f"epoch:{epoch+1},D Real Loss: {d_loss_real},D Fake Loss :{d_loss_fake},G Loss :{g_loss}")

# Set Parameters;
latent_dim = 5

# Define and compile discriminator
discriminator = define_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the  generator
generator = define_generator(latent_dim) 

# Define the GAN Model
gan_model = define_gan(generator,discriminator)

#Train the GAN 
train_gan(generator, discriminator,gan_model, latent_dim)
