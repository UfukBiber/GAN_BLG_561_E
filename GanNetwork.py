import tensorflow as tf 
from os.path import join

class GAN(tf.keras.models.Model):
    def __init__(self, units:int,  latent_dim:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.units= units

        self.generator = self.Build_Generator()
        self.discriminator = self.Build_Discriminator()

        self.generator_path = join("GAN", "Generator")
        self.discriminator_path = join("GAN", "Discriminator")


    def Build_Generator(self):
        inp = tf.keras.layers.Input(shape = (self.latent_dim))

        out = tf.keras.layers.Dense(32 * 32  * 2**3, 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                    use_bias=False)(inp)
        out = tf.keras.activations.relu(out)
        out = tf.keras.layers.Dense(32 * 32  * 2 ** 2, 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                    use_bias=False)(out)
        
        out = tf.keras.activations.relu(out)

        out = tf.keras.layers.Dense(32 * 32 *2, 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                    use_bias=False)(out)
        
        out = tf.keras.activations.relu(out)
        
        out = tf.keras.layers.Dense(32 * 32,
                                    use_bias=False,
                                    activation="tanh")(out)
        out = tf.keras.layers.Reshape((32, 32, 1))(out)
        return tf.keras.models.Model(inp, out, name = "Generator")
    

    def Build_Discriminator(self):
        inp = tf.keras.layers.Input(shape = (32, 32, 1), name = "Discriminator_Input")

        out = tf.keras.layers.Flatten()(inp)

        out = tf.keras.layers.Dense(32 * 32 * 2, 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                    use_bias=False)(out)
        
        out = tf.keras.activations.relu(out)

        out = tf.keras.layers.Dense(32 * 32 * 2**2, 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                    use_bias=False)(out)
        out = tf.keras.activations.relu(out)

        out = tf.keras.layers.Dense(32 * 32 * 2**3, 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                    use_bias=False)(out)
        out = tf.keras.activations.relu(out)

        out = tf.keras.layers.Dense(1, 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                                    use_bias=False,
                                    activation="sigmoid")(out)
        return tf.keras.models.Model(inp, out, name = "Discriminator")
    
    def compile(self):
        super(GAN, self).compile()
        self.d_optimizer = tf.keras.optimizers.Adam()
        self.g_optimizer = tf.keras.optimizers.Adam()
        self.g_loss_metric = tf.keras.metrics.Mean(name = "gen_loss")
        self.d_loss_metric = tf.keras.metrics.Mean(name = "disc_loss")
        self.loss = tf.keras.losses.BinaryCrossentropy()
    
    @tf.function
    def train_step(self, real_imgs):
        batch_size = tf.shape(real_imgs)[0]
        fake_labels = tf.zeros(shape = (batch_size, 1), dtype = tf.float32)
        real_labels = tf.ones(shape = (batch_size, 1), dtype = tf.float32)
        labels = tf.concat((real_labels, fake_labels), axis = 0, name = "labels")
        latent_vector = tf.random.normal(shape = (batch_size, self.latent_dim))
        gen_imgs = self.generator(latent_vector)
        imgs = tf.concat((real_imgs, gen_imgs), axis = 0, name = "imgs")
        with tf.GradientTape() as d_tape:
            predictions = self.discriminator(imgs)
            d_loss = self.loss(labels, predictions)
            gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        
        latent_vector = tf.random.normal(shape = (batch_size, self.latent_dim))
        with tf.GradientTape() as g_tape:
            gen_imgs = self.generator(latent_vector)
            predictions = self.discriminator(gen_imgs)
            g_loss = self.loss(real_labels, predictions)
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
        self.generator.load_weights(generator_path)
        self.discriminator.load_weights(discriminator_path)




