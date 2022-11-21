import tensorflow as tf
from matplotlib.pyplot import imsave 
from os.path import join

class Conv2D(tf.keras.layers.Layer):
    def __init__(self, filter_size, kernel_size, gain = 2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.gain = gain
    
    def build(self, input_shape):
        self.w = self.add_weight(name = "Kernel",
                        shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filter_size),
                        initializer=tf.keras.initializers.RandomNormal(mean = 0.0, stddev=1.0),
                        trainable=True)
        self.b = self.add_weight(name = "Bias",
                        shape = (self.filter_size),
                        initializer=tf.keras.initializers.zeros(),
                        trainable=True)
        self.scale = self.gain / (self.kernel_size * self.kernel_size * input_shape[-1])**0.5
    
    def call(self, input):
        return tf.nn.conv2d(input, self.w * self.scale, 1, padding = "SAME") + self.b
    

class Dense(tf.keras.layers.Layer):
    def __init__(self, units, gain = 2.0, learning_rate = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.gain = gain
        self.learning_rate = learning_rate
    
    def build(self, input_shape):
        self.w = self.add_weight(name = "Kernel",
                        shape = (input_shape[-1], self.units),
                        initializer=tf.keras.initializers.RandomNormal(mean = 0.0, stddev=1.0 / self.learning_rate),
                        trainable=True)
        self.b = self.add_weight(name = "Bias",
                        shape = (self.units),
                        initializer=tf.keras.initializers.zeros(),
                        trainable=True)
        self.scale = self.gain / (input_shape[-1])**0.5
    
    def call(self, input):
        return (tf.matmul(input, self.w * self.scale) + self.b) * self.learning_rate
    
class MiniBatchStd(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = 4
    def build(self, input_shape):
        self.n, self.h, self.w, self.c = input_shape

    def call(self, input_tensor):
        x = tf.reshape(input_tensor, [self.group_size, -1, self.h, self.w, self.c])
        group_mean, group_var = tf.nn.moments(x, axes=(0), keepdims=False)
        group_std = tf.sqrt(group_var + 1e-8)
        avg_std = tf.reduce_mean(group_std, axis=[1, 2, 3], keepdims=True)
        x = tf.tile(avg_std, [self.group_size, self.h, self.w, 1])
        return tf.concat([input_tensor, x], axis = -1)

class PixelNormalization(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def call(self, inputs):
        squarred_inputs = tf.math.square(inputs)
        mean_squarred_inputs = tf.math.reduce_mean(squarred_inputs, axis = -1, keepdims=True)
        mean_squarred_inputs += 1e-8
        out = tf.math.sqrt(mean_squarred_inputs)
        return inputs / out

class FadeIn(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def call(self, old_outputs, new_outputs, alpha):
        return alpha * new_outputs + (1 - alpha) * old_outputs


class Callback(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
    def on_epoch_end(self, epoch, logs=None):
        epoch += self.model.epoch 
        self.model.Save_Weights(epoch)
        latent_vector = tf.random.normal(shape = (5, self.model.latent_dim))
        gen_imgs = self.model.generator(latent_vector)
        gen_imgs *= 127.5
        gen_imgs += 127.5
        gen_imgs = tf.cast(gen_imgs, tf.uint8)
        for i in range(5):
            path = join("Gen_Imgs", self.path, str(epoch)+"_"+str(i)+".jpg")
            imsave(path, gen_imgs[i].numpy())
        self.save_loss(logs, epoch)
    
    def save_loss(self, logs, epoch):
        keys = list(logs.keys())
        s = "Epoch : %i%10s : %11.5f%16s : %17.5f\n"%(epoch, "G_Loss", logs[keys[1]], "D_Loss", logs[keys[0]])
        with open("%s_Losses.txt"%self.path, "a") as f:
            f.write(s)
            f.close()





if __name__ == "__main__":
    inp = tf.ones(shape = (8, 16, 16, 128), dtype = tf.float32)
    out = PixelNormalization()(inp)
    
