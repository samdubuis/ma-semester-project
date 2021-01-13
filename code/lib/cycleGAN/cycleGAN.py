from lib.data import final_stn_loader
import pathlib

import cv2
from pathlib import Path
from lib.cycleGAN.module import remove_pad_and_classify, remove_pad
import functools

import lib.cycleGAN.imlib as im
import numpy as np
import lib.cycleGAN.pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import lib.cycleGAN.tf2lib as tl
import lib.cycleGAN.tf2gan as gan
import tqdm

import lib.cycleGAN.data as data
import lib.cycleGAN.module as module


class cycleGAN:
    def __init__(self,
                 dataset="tmp_gan",
                 datasets_dir="../data/",
                 output_dir="output/",
                 load_size=256,
                 crop_size=256,
                 batch_size=1,
                 epochs=200,
                 epoch_decay=100,
                 lr=0.0002,
                 beta_1=0.5,
                 # choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
                 adversarial_loss_mode="lsgan",
                 # choices=['none', 'dragan', 'wgan-gp'])
                 gradient_penalty_mode="none",
                 gradient_penalty_weight=10.0,
                 cycle_loss_weight=10.0,
                 identity_loss_weight=0.0,
                 classification_loss_weight=0.01,
                 pool_size=50,
                 stn=None):

        self.dataset = dataset
        self.datasets_dir = datasets_dir
        self.load_size = load_size
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_decay = epoch_decay
        self.lr = lr
        self.beta_1 = beta_1
        self.adversarial_loss_mode = adversarial_loss_mode
        self.gradient_penalty_mode = gradient_penalty_mode
        self.gradient_penalty_weight = gradient_penalty_weight
        self.cycle_loss_weight = cycle_loss_weight
        self.identity_loss_weight = identity_loss_weight
        self.pool_size = pool_size

        self.output_dir = datasets_dir+output_dir+dataset
        self.stnet = stn.stnet
        self.classification_loss_weight = classification_loss_weight

        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # save settings
        # py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

        print("\nInitialization cycleGAN ...")
        # ==============================================================================
        # =                                    data                                    =
        # ==============================================================================

        A_img_paths = py.glob(py.join(self.datasets_dir, self.dataset, 'trainA'), '*.jpg')
        B_img_paths = py.glob(py.join(self.datasets_dir, self.dataset, 'trainB'), '*.jpg')
        self.A_B_dataset, self.len_dataset = data.make_zip_dataset(
            A_img_paths, B_img_paths, self.batch_size, self.load_size, self.crop_size, training=True, repeat=False)

        self.A2B_pool = data.ItemPool(self.pool_size)
        self.B2A_pool = data.ItemPool(self.pool_size)

        A_img_paths_test = py.glob(py.join(self.datasets_dir, self.dataset, 'testA'), '*.jpg')
        B_img_paths_test = py.glob(py.join(self.datasets_dir, self.dataset, 'testB'), '*.jpg')
        self.A_B_dataset_test, _ = data.make_zip_dataset(
            A_img_paths_test, B_img_paths_test, self.batch_size, self.load_size, self.crop_size, training=False, repeat=True)

        # ==============================================================================
        # =                                   models                                   =
        # ==============================================================================

        self.G_A2B = module.ResnetGenerator(input_shape=(self.crop_size, self.crop_size, 1))
        self.G_B2A = module.ResnetGenerator(input_shape=(self.crop_size, self.crop_size, 1))

        self.D_A = module.ConvDiscriminator(input_shape=(self.crop_size, self.crop_size, 1))
        self.D_B = module.ConvDiscriminator(input_shape=(self.crop_size, self.crop_size, 1))

        self.d_loss_fn, self.g_loss_fn = gan.get_adversarial_losses_fn(self.adversarial_loss_mode)
        self.cycle_loss_fn = tf.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.losses.MeanAbsoluteError()

        # add
        # SparseCategoricalCrossentropy()
        self.classification_loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

        G_lr_scheduler = module.LinearDecay(self.lr, self.epochs * self.len_dataset, self.epoch_decay * self.len_dataset)
        D_lr_scheduler = module.LinearDecay(self.lr, self.epochs * self.len_dataset, self.epoch_decay * self.len_dataset)
        self.G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=self.beta_1)
        self.D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=self.beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================


    @tf.function
    def train_G(self, A, B):
        with tf.GradientTape() as t:
            A2B = self.G_A2B(A, training=True)
            B2A = self.G_B2A(B, training=True)
            A2B2A = self.G_B2A(A2B, training=True)
            B2A2B = self.G_A2B(B2A, training=True)
            A2A = self.G_B2A(A, training=True)
            B2B = self.G_A2B(B, training=True)

            A2B_d_logits = self.D_B(A2B, training=True)
            B2A_d_logits = self.D_A(B2A, training=True)

            A2B_g_loss = self.g_loss_fn(A2B_d_logits)
            B2A_g_loss = self.g_loss_fn(B2A_d_logits)
            A2B2A_cycle_loss = self.cycle_loss_fn(A, A2B2A)
            B2A2B_cycle_loss = self.cycle_loss_fn(B, B2A2B)
            A2A_id_loss = self.identity_loss_fn(A, A2A)
            B2B_id_loss = self.identity_loss_fn(B, B2B)

            
            label_A = remove_pad_and_classify(self.stnet, A)
            label_B = remove_pad_and_classify(self.stnet, B)
            label_A2B = remove_pad_and_classify(self.stnet, A2B)
            label_B2A = remove_pad_and_classify(self.stnet, B2A)
            
            A2B_class_loss = tf.abs(self.classification_loss_fn(label_A, label_A2B))
            B2A_class_loss = tf.abs(self.classification_loss_fn(label_B, label_B2A))

            G_loss = (A2B_g_loss + B2A_g_loss) + \
                (A2B2A_cycle_loss + B2A2B_cycle_loss) * self.cycle_loss_weight + \
                (A2A_id_loss + B2B_id_loss) * self.identity_loss_weight + \
                (A2B_class_loss + B2A_class_loss) * self.classification_loss_weight

        G_grad = t.gradient(
            G_loss, self.G_A2B.trainable_variables + self.G_B2A.trainable_variables)
        self.G_optimizer.apply_gradients(
            zip(G_grad, self.G_A2B.trainable_variables + self.G_B2A.trainable_variables))

        return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                          'B2A_g_loss': B2A_g_loss,
                          'A2B2A_cycle_loss': A2B2A_cycle_loss,
                          'B2A2B_cycle_loss': B2A2B_cycle_loss,
                          'A2A_id_loss': A2A_id_loss,
                          'B2B_id_loss': B2B_id_loss,
                          'A2B_class_loss': A2B_class_loss,
                          'B2A_class_loss': B2A_class_loss}

    @tf.function
    def train_D(self, A, B, A2B, B2A):
        with tf.GradientTape() as t:
            A_d_logits = self.D_A(A, training=True)
            B2A_d_logits = self.D_A(B2A, training=True)
            B_d_logits = self.D_B(B, training=True)
            A2B_d_logits = self.D_B(A2B, training=True)

            A_d_loss, B2A_d_loss = self.d_loss_fn(A_d_logits, B2A_d_logits)
            B_d_loss, A2B_d_loss = self.d_loss_fn(B_d_logits, A2B_d_logits)
            D_A_gp = gan.gradient_penalty(functools.partial(
                self.D_A, training=True), A, B2A, mode=self.gradient_penalty_mode)
            D_B_gp = gan.gradient_penalty(functools.partial(
                self.D_B, training=True), B, A2B, mode=self.gradient_penalty_mode)

            D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + \
                (D_A_gp + D_B_gp) * self.gradient_penalty_weight

        D_grad = t.gradient(D_loss, self.D_A.trainable_variables + self.D_B.trainable_variables)
        self.D_optimizer.apply_gradients(
            zip(D_grad, self.D_A.trainable_variables + self.D_B.trainable_variables))

        return {'A_d_loss': A_d_loss + B2A_d_loss,
                'B_d_loss': B_d_loss + A2B_d_loss,
                'D_A_gp': D_A_gp,
                'D_B_gp': D_B_gp}

    def train_step(self, A, B):
        A2B, B2A, G_loss_dict = self.train_G(A, B)

        # cannot autograph `A2B_pool`
        # or A2B = A2B_pool(A2B.numpy()), but it is much slower
        A2B = self.A2B_pool(A2B)
        # because of the communication between CPU and GPU
        B2A = self.B2A_pool(B2A)

        D_loss_dict = self.train_D(A, B, A2B, B2A)

        return G_loss_dict, D_loss_dict

    @tf.function
    def sample(self, A, B):
        A2B = self.G_A2B(A, training=False)
        B2A = self.G_B2A(B, training=False)
        A2B2A = self.G_B2A(A2B, training=False)
        B2A2B = self.G_A2B(B2A, training=False)
        return A2B, B2A, A2B2A, B2A2B


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

    def run(self):
        # epoch counter
        ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

        # checkpoint
        checkpoint = tl.Checkpoint(dict(G_A2B=self.G_A2B,
                                        G_B2A=self.G_B2A,
                                        D_A=self.D_A,
                                        D_B=self.D_B,
                                        G_optimizer=self.G_optimizer,
                                        D_optimizer=self.D_optimizer,
                                        ep_cnt=ep_cnt),
                                   py.join(self.output_dir, 'checkpoints'),
                                   max_to_keep=5)
        try:  # restore checkpoint including the epoch counter
            checkpoint.restore().assert_existing_objects_matched()
        except Exception as e:
            print(e)

        # summary
        # train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

        # sample
        test_iter = iter(self.A_B_dataset_test)
        sample_dir = py.join(self.output_dir, 'samples_training')
        py.mkdir(sample_dir)

        # main loop
        # with train_summary_writer.as_default():
        for ep in tqdm.trange(self.epochs, desc='Epoch Loop'):
            if ep < ep_cnt:
                continue

            # update epoch counter
            ep_cnt.assign_add(1)

            # train for an epoch
            for A, B in tqdm.tqdm(self.A_B_dataset, desc='Inner Epoch Loop', total=self.len_dataset):
                self.G_loss_dict, self.D_loss_dict = self.train_step(A, B)

                # summary
                # tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
                # tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
                # tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

                # sample
                if self.G_optimizer.iterations.numpy() % 1000 == 0:
                    A, B = next(test_iter)
                    A2B, B2A, A2B2A, B2A2B = self.sample(A, B)
                    img = im.immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=2)
                    im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % self.G_optimizer.iterations.numpy()))

            # print losses
            print("\nLosses : A2B_g_loss, A2B2A_cycle_loss, A2B_classification_loss")
            tf.print(self.G_loss_dict["A2B_g_loss"])
            tf.print(self.G_loss_dict["A2B2A_cycle_loss"])
            tf.print(self.G_loss_dict["A2B_class_loss"])

            # save checkpoint
            checkpoint.save(ep)


    def apply_final(self):
        
        tl.Checkpoint(dict(G_A2B=self.G_A2B, G_B2A=self.G_B2A), py.join(self.datasets_dir+"output/tmp_gan/", 'checkpoints')).restore()

        
        Path(self.datasets_dir+"output/final_gan/train/").mkdir(parents=True, exist_ok=True)
        Path(self.datasets_dir+"output/final_gan/test/").mkdir(parents=True, exist_ok=True)

        A_ds = final_stn_loader(self.datasets_dir+"output/final_stn/train/", stn=False)
        B_ds = final_stn_loader(self.datasets_dir+"output/final_stn/test/", stn=False)

        for i, (A, y) in enumerate(A_ds.batch(1), start=1):
            img = self.G_A2B(A, training=False)
            img = remove_pad(img)
            img = im.immerge(np.concatenate([img], axis=0), n_rows=1)
            path = self.datasets_dir+"output/final_gan/train/finger{}_u{}.jpg".format(str(i).zfill(4), y.numpy()[0])
            if i==1 : print(path)
            im.imwrite(img, path)
            
        for i, (B, y) in enumerate(B_ds.batch(1), start=1):
            img = self.G_A2B(B, training=False)
            img = remove_pad(img)
            img = im.immerge(np.concatenate([img], axis=0), n_rows=1)
            path = self.datasets_dir+"output/final_gan/test/finger{}_u{}.jpg".format(str(i).zfill(4), y.numpy()[0])
            im.imwrite(img, path)
            
        print("Done applying GAN to images\n")