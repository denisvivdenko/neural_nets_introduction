import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import numpy as np



class DDIM(keras.Model):
    def __init__(self, unet, **kwargs):
        super().__init__(**kwargs)
        
        self.network = unet
        self.ema_network = keras.models.clone_model(self.network)
        self.max_signal_rate = 0.95
        self.min_signal_rate = 0.05
        self.ema = 0.999
        self.normalizer = layers.Normalization()
    
    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss = keras.metrics.Mean(name="noise_loss")
        self.image_loss = keras.metrics.Mean(name="image_loss")

    @property
    def metrics(self):
        return [self.noise_loss, self.image_loss]
    
    def train_step(self, images_batch):
        images = self.normalizer(images_batch, training=True)
        batch_size, self.image_size, self.n_channels = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[-1]
        
        noise = self.generate_noise(
            batch_size=batch_size,
            image_size=self.image_size,
            n_channels=self.n_channels
        )

        diffusion_step = np.random.uniform(
            0, 1, (batch_size, 1, 1, 1)
        )
        noise_rate, signal_rate = self.diffusion_schedule(diffusion_step)
        noised_images = noise_rate * noise + images * signal_rate

        with tf.GradientTape() as tape:
            pred_noise, pred_images = self.denoise(noised_images, noise_rate, signal_rate, training=True)

            noise_loss = self.loss(pred_noise, noise)
            image_loss = self.loss(pred_images, images)
            self.noise_loss(noise_loss)
            self.image_loss(image_loss)

        grads = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))

        for ema_weights, weights in zip(self.ema_network.weights, self.network.weights):
            ema_weights.assign(ema_weights * self.ema + (1 - self.ema) * weights)
            
        return {m.name(): m.result() for m in self.metrics}
    
    def generate(self, num_samples, diffusion_steps):
        initial_noise = tf.random.normal(0, 1, (num_samples, self.image_size, self.image_size, self.n_channels))
        samples = self.reverse_diffusion(initial_noise, diffusion_steps)
        samples = self.denormalize(samples)
        return samples
    
    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(images, noise_rates, signal_rates, training=False)

            next_noise_rates, next_signal_rates = self.diffusion_schedule(diffusion_times - step_size)
            images = next_noise_rates * pred_noises + next_signal_rates * pred_images
            
        return pred_images

    def diffusion_schedule(self, diffusion_times):
        start_angle = tf.acos(self.max_signal_rate) # Pay attention
        end_angle = tf.asin(self.min_signal_rate)
                            
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
                            
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
                   
        return noise_rates, signal_rates

    def denoise(self, images, noise_rates, signal_rates, training=False):
        if training:
            network = self.network
        else:
            network = self.ema
        pred_noises = network([images, noise_rates**2], training=training) # TODO
        pred_images = (images - pred_noises * noise_rates) / signal_rates
        return pred_noises, pred_images

    def denormalize(self, images):
        denormalized_images = self.normalizer.mean + images * self.normalizer.variance ** 0.5 #TODO
        return tf.clip_by_value(denormalized_images, 0.0, 1.0)
    
    def generate_noise(self, batch_size, image_size, n_channels):
        return np.random.normal(0, 1, (batch_size, image_size, image_size, n_channels))