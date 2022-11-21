import tensorflow as tf 
from os.path import join
import os
from customlayers import Conv2D, Dense, MiniBatchStd, PixelNormalization, FadeIn
from input import load_dataset
from matplotlib.pyplot import imsave

STEP_TO_IMAGE_RES = dict(((i, 2**(i+2)) for i in range(8)))
RES_TO_FILTER_SIZE_GEN = {4:[512, 512], 8:[512, 512], 16:[256, 256], 32:[128, 128], 64:[128, 128], 128:[64, 64], 256:[64, 64]}
RES_TO_FILTER_SIZE_DISC = {4:[512, 512, 512], 8:[512, 512, 256], 16:[256, 256, 128], 32:[128, 128, 128], 64:[128, 64, 64], 128:[64, 64, 64], 256:[64, 64, 64]}
EPOCHS = {4:4, 8:4, 16:4, 32:4, 64:4, 128:4, 256:4}
BATCH_SIZE = {4:32, 8:16, 16:8, 32:8, 64:4, 128:4, 256:4}
OPTIMIZER_INITIALIZER = {"learning_rate":1e-3, "beta_1":0.0, "beta_2":0.99, "epsilon":1e-8}


class ProgressiveGan():
    def __init__(self, leaky_relu_alpha, latent_dim, n_steps, *args, **kwargs):
        self.leaky_relu_alpha = leaky_relu_alpha
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.describe()

        self.generator_blocks = self.build_all_generator_blocks()
        self.discriminator_blocks = self.build_all_discriminator_blocks()
        self.to_rgb_blocks = self.build_all_to_rgb_blocks()
        self.from_rgb_blocks = self.build_all_from_rgb_blocks()

    def describe(self):
        for i in range(self.n_steps):
            img_res = STEP_TO_IMAGE_RES[i]
            if i!=0:
                print(f"{i+1}_{img_res}x{img_res} fade-in model will be trained.")
            print(f"{i+1}_{img_res}x{img_res} standard model will be trained.")

    def build_mapping_block(self):
        pass 
    def build_base_generator_block(self):
        pass 
    def build_generator_block(self, i):
        pass


    def build_to_rgb_block(self, step):
        img_res = STEP_TO_IMAGE_RES[step]
        inp = tf.keras.layers.Input(shape = (img_res, img_res, RES_TO_FILTER_SIZE_GEN[img_res][1]), name = f"To_RGB_{img_res}_Input")
        out = Conv2D(3, 1, gain = 1.0)(inp)
        return tf.keras.models.Model(inp, out, name = f"To_RGB_{img_res}_Block")

    def build_all_generator_blocks(self):
        generator_blocks = []
        generator_blocks.append(self.build_base_generator())
        for i in range(1, self.n_steps):
            generator_blocks.append(self.build_generator_block(i))
        return generator_blocks
    
    def build_all_to_rgb_blocks(self):
        to_rgb_blocks = []
        for i in range(self.n_steps):
            to_rgb_blocks.append(self.build_to_rgb_block(i))
        return to_rgb_blocks


    def build_base_discriminator(self):
        inp = tf.keras.layers.Input(shape = (4, 4, RES_TO_FILTER_SIZE_DISC[4][0]), name = "Discriminator_Base_Input")
        out = MiniBatchStd()(inp)

        out = Conv2D(RES_TO_FILTER_SIZE_DISC[4][1], 3, name = "Discriminator_4_Block_Conv_1")(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)

        out = Conv2D(RES_TO_FILTER_SIZE_DISC[4][0], 4)(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)

        out = tf.keras.layers.Flatten()(out)
        out = Dense(1, gain = 1./8)(out)

        return tf.keras.models.Model(inp, out, name = "Discriminator_4_Block")

    def build_discriminator_block(self, step):
        if step < 1: raise ValueError("Step must be bigger than zero")
        img_res = STEP_TO_IMAGE_RES[step]
        inp = tf.keras.layers.Input(shape = (img_res, img_res, RES_TO_FILTER_SIZE_DISC[img_res][2]), name = f"Discriminator_{img_res}_Block_Input")

        out = Conv2D(RES_TO_FILTER_SIZE_DISC[img_res][1], 3)(inp)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)
        
        out = Conv2D(RES_TO_FILTER_SIZE_DISC[img_res][0], 3)(out)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)

        out = tf.keras.layers.AveragePooling2D()(out)

        return tf.keras.models.Model(inp, out, name = f"Discriminator_{img_res}_Block")

    def build_from_rgb_block(self, step):
        img_res = STEP_TO_IMAGE_RES[step]
        inp = tf.keras.layers.Input(shape = (img_res, img_res, 3), name = f"Discriminator_{img_res}_From_RGB_Input")
        out = Conv2D(RES_TO_FILTER_SIZE_DISC[img_res][2], 1, gain = 1.0)(inp)
        out = tf.keras.layers.LeakyReLU(self.leaky_relu_alpha)(out)
        return tf.keras.models.Model(inp, out, name = f"From_RGB_{img_res}_Block")

    def build_all_discriminator_blocks(self):
        discriminator_blocks = []
        discriminator_blocks.append(self.build_base_discriminator())
        for i in range(1, self.n_steps):
            discriminator_blocks.append(self.build_discriminator_block(i))
        return discriminator_blocks

    def build_all_from_rgb_blocks(self):
        from_rgb_blocks = []
        for i in range(self.n_steps):
            from_rgb_blocks.append(self.build_from_rgb_block(i))
        return from_rgb_blocks

    def build_standard_generator_model(self, step):
        pass

    def build_standard_discriminator_model(self, step):
        img_res = STEP_TO_IMAGE_RES[step]
        inp = tf.keras.layers.Input(shape = (img_res, img_res, 3), name = f"{img_res}_Standard_Discriminator_Input")
        out = self.from_rgb_blocks[step](inp)
        for i in range(step, -1, -1):
            out = self.discriminator_blocks[i](out)
        return tf.keras.models.Model(inp, out, name = f"{img_res}_Standard_Discriminator_Model")

    def build_fade_in_generator_model(self, step):
        pass
    def build_fade_in_discriminator_model(self, step):
        if step < 1: raise ValueError("Step must be bigger than zero")
        img_res = STEP_TO_IMAGE_RES[step]
        inp = tf.keras.layers.Input(shape = (img_res, img_res, 3), name = f"{img_res}_Fade_In_Discriminator_Model_Input")
        alpha = tf.keras.layers.Input(shape = (1,), name= "Alpha")

        new_out = self.from_rgb_blocks[step](inp)
        new_out = self.discriminator_blocks[step](new_out)

        old_out = tf.keras.layers.AveragePooling2D()(inp)
        old_out = self.from_rgb_blocks[step-1](old_out)

        out = FadeIn()(old_out, new_out, alpha)
        for i in range(step-1, -1, -1):
            out = self.discriminator_blocks[i](out)
        return tf.keras.models.Model([inp, alpha], out, name = f"{img_res}_Fade_In_Discriminator_Model")

    def train(self):
        img_res = STEP_TO_IMAGE_RES[0]
        print(f"\nTraining {img_res}_{img_res} standard model\n\n")
        generator = self.build_standard_generator_model(0)
        discriminator = self.build_standard_discriminator_model(0)
        standard_model = StandardModel(generator, discriminator, self.latent_dim)
        ds = load_dataset((img_res, img_res), BATCH_SIZE[img_res], 30000)
        standard_model.compile()
        history = standard_model.fit(ds, epochs = EPOCHS[img_res], callbacks=[StandardCallback(img_res)])
        self.save_loss_history(history.history, standard_model.name)
        self.save_blocks(standard_model)
        for i in range(1, 3):
            img_res = STEP_TO_IMAGE_RES[i]
            ds = load_dataset((img_res, img_res), BATCH_SIZE[img_res])
            print("-"*100)
            print(f"\n\nTraining {img_res}_{img_res} fade-in model\n\n")
            generator = self.build_fade_in_generator_model(i)
            discriminator = self.build_fade_in_discriminator_model(i)
            alpha_step = 1./EPOCHS[img_res]
            fade_in_model = FadeInModel(generator, discriminator, self.latent_dim, alpha_step)
            fade_in_model.compile()
            history = fade_in_model.fit(ds, epochs = EPOCHS[img_res], callbacks=[FadeInCallback()])
            self.save_blocks(fade_in_model)
            self.save_loss_history(history.history, fade_in_model.name)

            print(f"\nTraining {img_res}_{img_res} standard model\n\n")
            generator = self.build_standard_generator_model(i)
            discriminator = self.build_standard_discriminator_model(i)
            standard_model = StandardModel(generator, discriminator, self.latent_dim)
            standard_model.compile()
            standard_model.fit(ds, epochs = EPOCHS[img_res], callbacks=[StandardCallback(img_res)])
            self.save_loss_history(history.history, standard_model.name)
            self.save_blocks(standard_model)

    def save_loss_history(self, history, model_name):
        keys = list(history.keys())
        losses = list(history.values())
        string = "%20s\n\n"%(model_name.upper())
        string += "%1s%10s%10s\n\n"%("Epochs", keys[0], keys[1])
        for i in range(len(losses[1])):
            string += "%3i%12.4f%12.4f\n"%(i, losses[0][i], losses[1][i])
        string += "\n\n\n"
        with open("Loss_History.txt", "a") as f:
            f.write(string)
            f.close()
        
    def save_blocks(self, model):
        generator_path = join("Generator_Blocks", model.generator.name)
        discriminator_path = join("Discriminator_Blocks", model.discriminator.name)
        for i in range(len(self.generator_blocks)):
            self.generator_blocks[i].save_weights(join(generator_path, self.generator_blocks[i].name))
            self.discriminator_blocks[i].save_weights(join(discriminator_path, self.discriminator_blocks[i].name))
        
    def load_blocks(self, step, is_fade_in = False):
        img_res = STEP_TO_IMAGE_RES[step]
        if is_fade_in:
            model_name = "Fade_In"
        else:
            model_name = "Standard"
        generator_path = join("Generator_Blocks", f"{img_res}_{model_name}_Generator_Model")
        discriminator_path = join("Discriminator_Blocks", f"{img_res}_{model_name}_Discriminator_Model")
        for i in range(len(self.generator_blocks)):
            self.generator_blocks[i].save_weights(join(generator_path, self.generator_blocks[i].name))
            self.discriminator_blocks[i].save_weights(join(discriminator_path, self.discriminator_blocks[i].name))
        


class StandardModel(tf.keras.models.Model):
    def __init__(self, generator:tf.keras.models.Model, discriminator:tf.keras.models.Model, latent_dim:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator 
        self.discriminator = discriminator

        self.gradient_penalty_coeff = 10.0
        self.latent_dim = latent_dim

        self.generator_path = join("Generator", "Progressive_GAN", str(self.name))
        self.discriminator_path = join("Discriminator", "Progressive_GAN", str(self.name))

        self.generator_metric = tf.keras.metrics.Mean("G_Loss")
        self.discriminator_metric = tf.keras.metrics.Mean("D_Loss")

    def compile(self):
        super(StandardModel, self).compile()
        self.d_optimizer = tf.keras.optimizers.Adam(**OPTIMIZER_INITIALIZER)
        self.g_optimizer = tf.keras.optimizers.Adam(**OPTIMIZER_INITIALIZER)

    def loss_fn(self, y_true, y_pred):
        return -tf.math.reduce_mean(y_true * y_pred)

    def save_model(self, epoch):
        self.generator.save_weights(join(self.generator_path, str(epoch)))
        self.discriminator.save_weights(join(self.discriminator, str(epoch)))
    
    def load_model(self):
        self.generator.load_weights(self.generator_path)
        self.discriminator.load_weights(self.discriminator_path)
    
    def gradient_penalty(self, real_img_batch, fake_img_batch, batch_size):
        alpha = tf.random.uniform(shape = (batch_size, 1, 1, 1))
        interpolation = alpha * real_img_batch + (1 - alpha) * fake_img_batch
        with tf.GradientTape() as tape:
            tape.watch(interpolation)
            pred = self.discriminator(interpolation)
        grad = tape.gradient(pred, [interpolation])
        norm_grad = tf.sqrt(tf.math.reduce_sum(tf.math.square(grad), axis= [1, 2, 3]))
        norm_grad = tf.math.reduce_mean(tf.math.square(norm_grad - 1.0))
        return norm_grad

    @tf.function
    def train_step(self, real_img_batch):
        batch_size = tf.shape(real_img_batch)[0]
        latent_vector = tf.random.normal(shape = (batch_size, self.latent_dim))

        real_labels = tf.ones(shape = (batch_size, 1), dtype = tf.float32)
        fake_labels = -tf.ones(shape = (batch_size, 1), dtype = tf.float32)
        with tf.GradientTape() as d_tape:
            gen_img_batch = self.generator(latent_vector)
            gen_img_preds = self.discriminator(gen_img_batch)
            real_img_preds = self.discriminator(real_img_batch)
            d_loss_fake = self.loss_fn(fake_labels, gen_img_preds)
            d_loss_real = self.loss_fn(real_labels, real_img_preds)
            d_loss = d_loss_fake + d_loss_real
            gradient_penalty = self.gradient_penalty(real_img_batch, gen_img_batch, batch_size)
            d_loss += gradient_penalty * self.gradient_penalty_coeff
            d_loss += 1e-3 * tf.math.reduce_mean(tf.concat([real_img_preds, gen_img_preds], axis = 0)**2)

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        latent_vector = tf.random.normal(shape = (batch_size, self.latent_dim))
        with tf.GradientTape() as g_tape:
            gen_img_batch = self.generator(latent_vector)
            preds = self.discriminator(gen_img_batch)
            g_loss = self.loss_fn(real_labels, preds)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        self.discriminator_metric.update_state(d_loss)
        self.generator_metric.update_state(g_loss)

        return {"G_Loss":self.generator_metric.result(), "D_Loss":self.discriminator_metric.result()}
    
class FadeInModel(tf.keras.models.Model):
    def __init__(self, generator:tf.keras.models.Model, discriminator:tf.keras.models.Model, latent_dim:int, alpha_step:float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator 
        self.discriminator = discriminator
        self.alpha = tf.Variable(0.0, trainable = False, dtype=tf.float32)
        self.alpha_step = alpha_step

        self.gradient_penalty_coeff = 10.0
        self.latent_dim = latent_dim

        self.generator_path = join("Generator", "Progressive_GAN", self.name)
        self.discriminator_path = join("Discriminator", "Progressive_GAN", self.name)

        self.generator_metric = tf.keras.metrics.Mean("G_Loss")
        self.discriminator_metric = tf.keras.metrics.Mean("D_Loss")

    def compile(self):
        super(FadeInModel, self).compile()
        self.d_optimizer = tf.keras.optimizers.Adam(**OPTIMIZER_INITIALIZER)
        self.g_optimizer = tf.keras.optimizers.Adam(**OPTIMIZER_INITIALIZER)

    def loss_fn(self, y_true, y_pred):
        return -tf.math.reduce_mean(y_true * y_pred)
            
    def gradient_penalty(self, real_img_batch, fake_img_batch, batch_size):
        alpha = tf.random.uniform(shape = (batch_size, 1, 1, 1))
        interpolation = alpha * real_img_batch + (1 - alpha) * fake_img_batch
        with tf.GradientTape() as tape:
            tape.watch(interpolation)
            pred = self.discriminator([interpolation, self.alpha])
        grad = tape.gradient(pred, [interpolation])
        norm_grad = tf.sqrt(tf.math.reduce_sum(tf.math.square(grad), axis= [1, 2, 3]))
        norm_grad = tf.math.reduce_mean(tf.math.square(norm_grad - 1.0))
        return norm_grad

    @tf.function
    def train_step(self, real_img_batch):
        batch_size = tf.shape(real_img_batch)[0]
        latent_vector = tf.random.normal(shape = (batch_size, self.latent_dim))

        real_labels = tf.ones(shape = (batch_size, 1), dtype = tf.float32)
        fake_labels = -tf.ones(shape = (batch_size, 1), dtype = tf.float32)
        with tf.GradientTape() as d_tape:
            gen_img_batch = self.generator([latent_vector, self.alpha])
            gen_img_preds = self.discriminator([gen_img_batch, self.alpha])
            real_img_preds = self.discriminator([real_img_batch, self.alpha])
            d_loss_fake = self.loss_fn(fake_labels, gen_img_preds)
            d_loss_real = self.loss_fn(real_labels, real_img_preds)
            d_loss = d_loss_fake + d_loss_real
            gradient_penalty = self.gradient_penalty(real_img_batch, gen_img_batch, batch_size)
            d_loss += gradient_penalty * self.gradient_penalty_coeff
            d_loss += 1e-3 * tf.math.reduce_mean(tf.concat([real_img_preds, gen_img_preds], axis = 0)**2)

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        latent_vector = tf.random.normal(shape = (batch_size, self.latent_dim))
        with tf.GradientTape() as g_tape:
            gen_img_batch = self.generator([latent_vector, self.alpha])
            preds = self.discriminator([gen_img_batch, self.alpha])
            g_loss = self.loss_fn(real_labels, preds)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        self.discriminator_metric.update_state(d_loss)
        self.generator_metric.update_state(g_loss)

        return {"G_Loss":self.generator_metric.result(), "D_Loss":self.discriminator_metric.result()}
        

class StandardCallback(tf.keras.callbacks.Callback):
    def __init__(self, img_res):
        super().__init__()
        self.img_res = img_res
    
    def on_epoch_end(self, epoch, logs=None):
        latent_vector = tf.random.normal(shape = (5, self.model.latent_dim))
        gen_imgs = self.model.generator(latent_vector)
        gen_imgs *= 127.5
        gen_imgs += 127.5
        gen_imgs = tf.cast(gen_imgs, tf.uint8)
        for i in range(5):
            path = join("Gen_Imgs", "Progressive_GAN", f"{self.img_res}_{epoch}_{i}.jpg")
            imsave(path, gen_imgs[i].numpy())

class FadeInCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    
    def on_epoch_end(self, epoch, logs=None):
        self.model.alpha.assign_add(self.model.alpha_step)
            
if __name__ == "__main__":
    porg = ProgressiveGan(0.2, 512, 6)
    porg.train()
        
