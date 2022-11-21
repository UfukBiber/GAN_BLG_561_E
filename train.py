import tensorflow as tf 
from WPGanNetwork import WP_GAN
from DCGanNetwork import DC_GAN
from GanNetwork import GAN
from input import load_dataset
from customlayers import Callback
from os import listdir

if __name__ == "__main__":
    gan = GAN(64, 100)
    dc_gan = DC_GAN(64, 100, 0.2)
    wgan_gp = WP_GAN(64, 100, 0.2)

    ds = load_dataset((128, 128), 16)
    #gan.compile()
    #gan.fit(ds, epochs = 10, callbacks = [Callback("GAN")])
    dc_gan.compile()
    try:
        l_epoch = max([int(path) for path in listdir("DC_GAN/Generator")])
        dc_gan.Load_Weights(l_epoch)
        print("Loaded Weights from %i. epoch"%l_epoch)
    except:
        pass
    # wgan_gp.compile()
    dc_gan.fit(ds, epochs = 10, callbacks = [Callback("DC_GAN")])
    # wgan_gp.fit(ds, epochs = 10, callbacks = [Callback("WGAN_GP")])
