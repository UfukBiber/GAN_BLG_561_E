import tensorflow as tf 
from os.path import join


ADAM_OPTIMIZER_INIT = {"learning_rate":1e-4, "beta_1":0.0, "beta_2":0.9}


class WP_GAN(tf.keras.models.Model):
    def __init__(self, units, latent_dim, *args, leaky_alpha = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.units = units
        self.leaky_alpha = leaky_alpha
        self.epoch = 0

        self.generator = self.Build_Generator()
        self.discriminator = self.Build_Discriminator()

        self.gradient_penalty_coeff = 10.0
        self.n_critic = 5
        self.step = tf.Variable(0, trainable=False, dtype=tf.uint8)

        self.compile()
        self.generator_path = join("WGAN_GP", "Generator")
        self.discriminator_path = join("WGAN_GP", "Discriminator")

    def Build_Generator(self):
        inp = tf.keras.layers.Input(shape = (self.latent_dim))
        out = tf.keras.layers.Dense(4 * 4 * self.units * 2**(3), 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                    use_bias=False)(inp)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.activations.relu(out)
        out = tf.keras.layers.Reshape(target_shape = (4, 4, self.units * 2**(3)))(out)

        out = tf.keras.layers.Conv2DTranspose(self.units * 2**3, 
                                                  kernel_size=5,
                                                  strides=2, 
                                                  padding = "same", 
                                                  kernel_initializer= tf.keras.initializers.RandomNormal(mean = 0.0, stddev=0.02),
                                                  use_bias=False)(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.activations.relu(out)

        for i in range(4):
            out = tf.keras.layers.Conv2DTranspose(self.units * 2**(3 - i), 
                                                  kernel_size=5,
                                                  strides=2, 
                                                   padding = "same", 
                                                  kernel_initializer= tf.keras.initializers.RandomNormal(mean = 0.0, stddev=0.02),
                                                  use_bias=False)(out)
            out = tf.keras.layers.BatchNormalization()(out)
            out = tf.keras.activations.relu(out)
        out = tf.keras.layers.Conv2D(3, 1, padding = "same", activation="tanh", use_bias=False)(out)
        return tf.keras.models.Model(inp, out, name = "Generator")
    

    def Build_Discriminator(self):
        inp = tf.keras.layers.Input(shape = (128, 128, 3), name = "Discriminator_Input")
        out = tf.keras.layers.Conv2D(self.units, 
                                     kernel_size=5,
                                     strides=2,
                                     padding = "same", 
                                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                     use_bias=False)(inp)
        out = tf.keras.layers.LeakyReLU(self.leaky_alpha)(out)

        for i in range(3):
            out = tf.keras.layers.Conv2D(self.units * 2**(1 + i), 
                                         kernel_size=5,
                                         strides=2,
                                         padding="same",
                                         kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                         use_bias=False)(out)
            out = tf.keras.layers.LayerNormalization()(out)
            out = tf.keras.layers.LeakyReLU(self.leaky_alpha)(out)

        out = tf.keras.layers.Conv2D(self.units * 2**(1 + i), 
                                         kernel_size=5,
                                         strides=2,
                                         padding="same",
                                         kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                         use_bias=False)(out)
        out = tf.keras.layers.LayerNormalization()(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_alpha)(out)
    
        out = tf.keras.layers.Flatten()(out)
        out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=False)(out)
        return tf.keras.models.Model(inp, out, name = "Discriminator")
    
    def gradient_penaly(self, real_img, fake_imgs, batch_size):
        alpha = tf.random.uniform(shape = (batch_size, 1, 1, 1))
        interpolation = alpha * real_img + (1 - alpha) * fake_imgs
        with tf.GradientTape() as tape:
            tape.watch(interpolation)
            pred = self.discriminator(interpolation)
        grad = tape.gradient(pred, [interpolation])
        norm_grad = tf.sqrt(tf.math.reduce_sum(tf.math.square(grad), axis= [1, 2, 3]))
        norm_grad = tf.math.reduce_mean(tf.math.square(norm_grad - 1.0))
        return norm_grad


    def compile(self):
        super(WP_GAN, self).compile()
        self.d_optimizer = tf.keras.optimizers.Adam(**ADAM_OPTIMIZER_INIT)
        self.g_optimizer = tf.keras.optimizers.Adam(**ADAM_OPTIMIZER_INIT)
        self.g_loss_metric = tf.keras.metrics.Mean(name = "gen_loss")
        self.d_loss_metric = tf.keras.metrics.Mean(name = "disc_loss")

    def train_step(self, real_imgs):
        batch_size = tf.shape(real_imgs)[0]
        for _ in range(self.n_critic):
            latent_vector = tf.random.normal(shape = (batch_size, self.latent_dim))
            with tf.GradientTape() as d_tape:
                gen_imgs = self.generator(latent_vector)
                y_pred = self.discriminator(gen_imgs)
                y_true = self.discriminator(real_imgs)
                d_loss = tf.math.reduce_mean(y_pred) - tf.math.reduce_mean(y_true)
                gradient_penalty = self.gradient_penaly(real_imgs, gen_imgs, batch_size)
                d_loss += self.gradient_penalty_coeff * gradient_penalty
            gradient = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(gradient, self.discriminator.trainable_variables))
        
        latent_vector = tf.random.normal(shape = (batch_size, self.latent_dim))
        with tf.GradientTape() as g_tape:
            gen_imgs = self.generator(latent_vector)
            predictions = self.discriminator(gen_imgs)
            g_loss = -tf.math.reduce_mean(predictions)

        gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss)
        return {"disc_loss" : self.d_loss_metric.result(), "gen_loss":self.g_loss_metric.result()}

    def Save_Weights(self, epoch = None):
        if epoch is not None:
            generator_path = join(self.generator_path, str(epoch), "Gen")
            discriminator_path = join(self.discriminator_path, str(epoch), "Disc")
        self.generator.save_weights(generator_path)
        self.discriminator.save_weights(discriminator_path)
    
    def Load_Weights(self, epoch = None):
        if epoch is not None:
            generator_path = join(self.generator_path, str(epoch), "Gen")
            discriminator_path = join(self.discriminator_path, str(epoch), "Disc")
        self.epoch = epoch + 1
        self.generator.load_weights(generator_path)
        self.discriminator.load_weights(discriminator_path)




