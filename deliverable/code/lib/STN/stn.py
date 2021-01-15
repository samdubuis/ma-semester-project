import pathlib

from tensorlayer.models.core import Model

from lib.data import USM_DB_loader
from lib.STN.net import STNet
import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
# import pickle
# import os

class STN:
    def __init__(self, n_epoch, data_dir, data_name, gan_dir, output_dir, resize, learning_rate=0.0001, print_freq=5, batch_size=1):
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.print_freq = print_freq
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr=learning_rate)
        self.data_dir = data_dir
        self.data_name = data_name
        self.gan_dir = gan_dir
        self.resize = resize

        self.train_dir = self.data_dir+self.data_name+"/train/"
        self.test_dir = self.data_dir+self.data_name+"/test/"
        self.output_dir = output_dir

        self.train_ds = USM_DB_loader(self.train_dir, self.resize)
        self.eval_ds = self.train_ds.skip(int(0.7*self.train_ds.cardinality().numpy()))
        self.train_ds = self.train_ds.take(int(0.7*self.train_ds.cardinality().numpy()))
        self.test_ds = USM_DB_loader(self.test_dir, self.resize)

        ##================== DEFINE MODEL ============================================##
        self.stnet = STNet()

        ##================== DEFINE TRAIN OPS ========================================##
        self.train_weights = self.stnet.trainable_weights
        
        self.checkpoint_path = self.data_dir+"checkpoints/stn/"
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        
        print("Initialization of STN done\n")

    ##================== TRAINING ================================================##
    def train(self):
        
        # if os.path.isfile(self.checkpoint_path+"stn.p"):
        #     self.train_weights = pickle.load(open(self.checkpoint_path+"stn.p", "rb"))
        #     print("Checkpoint restored, skipping training")
        # else :    
        print("\nTraining ...")
        train_loss_array = []
        train_acc_array = []
        val_loss_array = []
        val_acc_array = []
            
        for epoch in tqdm(range(self.n_epoch), desc="STN Training"):
            start_time = time.time()

            self.stnet.train()  # enable dropout

            # for x_train, y_train in tqdm(self.train_ds.batch(self.batch_size), desc="Inner STN training loop", total=len(self.train_ds)//self.batch_size):
            for x_train, y_train in self.train_ds.batch(self.batch_size):
                with tf.GradientTape() as tape:
                    # compute outputs
                    # alternatively, you can use MLP(x, is_train=True) and remove MLP.train()
                    _logits, _ = self.stnet(x_train)
                    # compute loss and update model
                    _loss = tl.cost.cross_entropy(_logits, y_train, name='train_loss')

                grad = tape.gradient(_loss, self.train_weights)
                self.optimizer.apply_gradients(zip(grad, self.train_weights))

            

            # use training and evaluation sets to evaluate the model every print_freq epoch
            if epoch == 0 or (epoch + 1) % self.print_freq == 0:

                self.stnet.eval()  # disable dropout

                print("Epoch %d of %d took %fs" %
                    (epoch + 1, self.n_epoch, time.time() - start_time))

                train_loss, train_acc, n_iter = 0, 0, 0
                for x_train, y_train in self.train_ds.batch(self.batch_size):
                    # alternatively, you can use MLP(x, is_train=False) and remove MLP.eval()
                    _logits, _ = self.stnet(x_train)
                    train_loss += tl.cost.cross_entropy(_logits,
                                                        y_train, name='eval_train_loss')
                    train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_train))
                    n_iter += 1
                    
                train_loss_array.append(train_loss/n_iter)
                train_acc_array.append(train_acc/n_iter)
                print("   train loss: %f" % (train_loss / n_iter))
                print("   train acc: %f" % (train_acc / n_iter))

                val_loss, val_acc, n_iter = 0, 0, 0
                for x_val, y_val in self.eval_ds.batch(self.batch_size):
                    _logits, _ = self.stnet(x_val)  # is_train=False, disable dropout
                    val_loss += tl.cost.cross_entropy(_logits, y_val, name='eval_loss')
                    val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_val))
                    n_iter += 1
                    
                val_loss_array.append(val_loss/n_iter)
                val_acc_array.append(val_acc/n_iter)
                print("   validation loss: %f" % (val_loss / n_iter))
                print("   validation acc: %f" % (val_acc / n_iter))

                # pickle.dump(self.train_weights, open(self.checkpoint_path+"stn.p", "wb"))
                # print("Checkpoint")       
                
                for x, y in self.test_ds.batch(1):
                    _, img = self.stnet(x)
                    img = img[0].numpy()
                    img = np.concatenate((x[0], img), axis=1)
                    pathlib.Path(self.data_dir+self.output_dir+"sample_stn/").mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(self.data_dir+self.output_dir+"sample_stn/epoch{}.jpg".format(epoch), img)
                    break
    
        x = np.arange(0, self.n_epoch+1, self.print_freq)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x, train_loss_array)
        axs[0, 0].set_title('Training loss')
        axs[0, 1].plot(x, train_acc_array, 'tab:orange')
        axs[0, 1].set_title('Training accuracy')
        axs[1, 0].plot(x, val_loss_array, 'tab:green')
        axs[1, 0].set_title('Validation loss')
        axs[1, 1].plot(x, val_acc_array, 'tab:red')
        axs[1, 1].set_title('Validation accuracy')
        
        for ax in axs.flat:
            ax.set(xlabel='epoch', ylabel='loss or accuracy')
            
            
        fig.savefig(self.data_dir+self.output_dir+"stn_loss_accuracy_plot_e{}_b{}_lr{}.png".format(self.n_epoch, self.batch_size, self.learning_rate), bbox_inches='tight')
        # pickle.dump(self.train_weights, open(self.checkpoint_path+"stn.p", "wb"))

    ##================== EVALUATION ==============================================##
    def evalutation(self):
        print('\nEvaluation')

        self.stnet.eval()

        test_loss, test_acc, n_iter = 0, 0, 0
        for x_test, y_test in self.test_ds.batch(self.batch_size):
            _logits, _ = self.stnet(x_test)
            test_loss += tl.cost.cross_entropy(_logits, y_test, name='test_loss')
            test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_test))
            n_iter += 1
        print("   test loss: %f" % (test_loss / n_iter))
        print("   test acc: %f" % (test_acc / n_iter))
        
        f=open(self.data_dir+"results.txt", "a")
        f.write("Epoch : {}, batch_size : {}, lr : {}\n".format(self.n_epoch, self.batch_size, self.learning_rate))
        f.write("test loss: {}\n".format((test_loss / n_iter)))
        f.write("test acc: {}\n\n".format(test_acc / n_iter))


    def apply_stn_to_data(self):
        print("\nApplying to database")
        self.stnet.eval()
        
        train_ds = USM_DB_loader(self.train_dir, self.resize)
        test_ds = USM_DB_loader(self.test_dir, self.resize)
        Path(self.data_dir+self.output_dir+"final_stn/train/").mkdir(parents=True, exist_ok=True)
        Path(self.data_dir+self.output_dir+"final_stn/test/").mkdir(parents=True, exist_ok=True)
        
        i=1
        for x, y in train_ds.batch(1):
            _, img = self.stnet(x)
            img = img[0].numpy()
            path = self.data_dir+self.gan_dir+"/trainA/finger{}_u{}.jpg".format(str(i).zfill(4), y.numpy()[0])
            if i==1: print(path)
            cv2.imwrite(path, img)
            path = self.data_dir+self.output_dir+"final_stn/train/finger{}_u{}.jpg".format(str(i).zfill(4), y.numpy()[0])
            cv2.imwrite(path, img)
            i+=1

        print("{} images are now spatially transformed in trainA\n".format(i))

        i=1
        for x, y in test_ds.batch(1):
            _, img = self.stnet(x)
            img = img[0].numpy()
            path = self.data_dir+self.gan_dir+"/testA/finger{}_u{}.jpg".format(str(i).zfill(4), y.numpy()[0])
            if i==0: print(path)
            cv2.imwrite(path, img)
            path = self.data_dir+self.output_dir+"final_stn/test/finger{}_u{}.jpg".format(str(i).zfill(4), y.numpy()[0])
            cv2.imwrite(path, img)
            i+=1
        
        print("{} images are now spatially transformed in testA\n".format(i))

