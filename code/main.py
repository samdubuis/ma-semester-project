import os
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']= "2,3"

import argparse
from pathlib import Path

from lib.STN.stn import STN
from lib.cycleGAN.cycleGAN import cycleGAN
from lib.data import create_domainB, USM_DB_loader, final_stn_loader

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# GENERAL
ap.add_argument("--phase", default="train", help="Train or test ?")
ap.add_argument("--data_dir", default="../data/", help="Directory path of data")
ap.add_argument("--data_name", default="Published_database_FV-USM_Dec2013", help="Data name")
ap.add_argument("--output_dir", default="output/", help="Directory where samples are stored")
ap.add_argument("--brightness", default=0.2, help="Value for brightness adjustement when creating second domain")

# STN
ap.add_argument("--stn_epoch", default=300, type=int, help="Number of epochs for the STN")
ap.add_argument("--lr_stn", type = float, default = 0.0001, help="Learning rate for the STN")
ap.add_argument("--batch_size_stn", type=int, default = 32, help="Batch size for STN")
ap.add_argument("--print_freq", default=10, type=int, help="Frequency of printing info, the higher the less")

# CYCLEGAN
ap.add_argument("--batch_size_gan", type=int, default = 5, help="Batch size for GAN")
ap.add_argument("--gan_dir", default="tmp_gan", help="Directory path for the GAN part")
ap.add_argument("--resize", default=256, type=int, help="Size to which images are resized and then used")
ap.add_argument("--gan_epoch", default=200, type=int, help="Number of epochs for the GAN")

args = ap.parse_args()

# if args.phase == "train":
        
print("\nBeginning training .. ")

print(args)
print("\n")

Path(args.data_dir+args.gan_dir+"/trainA/").mkdir(parents=True, exist_ok=True)
Path(args.data_dir+args.gan_dir+"/testA/").mkdir(parents=True, exist_ok=True)
Path(args.data_dir+args.gan_dir+"/trainB/").mkdir(parents=True, exist_ok=True)
Path(args.data_dir+args.gan_dir+"/testB/").mkdir(parents=True, exist_ok=True)
    
stn = STN(n_epoch=args.stn_epoch, 
            data_name=args.data_name,
            data_dir=args.data_dir,
            gan_dir=args.gan_dir,
            resize=args.resize,
            output_dir = args.output_dir,
            print_freq=args.print_freq, 
            batch_size=args.batch_size_stn,
            learning_rate=args.lr_stn)

stn.train()
stn.evalutation()
stn.apply_stn_to_data()


create_domainB(args.data_dir, args.gan_dir, args.brightness, phase="train")
create_domainB(args.data_dir, args.gan_dir, args.brightness, phase="test")

cycleGAN = cycleGAN(batch_size=args.batch_size_gan, 
                    epochs=args.gan_epoch, 
                    dataset=args.gan_dir,
                    datasets_dir=args.data_dir,
                    output_dir=args.output_dir,
                    stn=stn)


# cycleGAN.run()
    
print("Applying GAN to DB")
cycleGAN.apply_final()

stn.train_ds = final_stn_loader(args.data_dir+args.output_dir+"final_gan/train/")
stn.eval_ds = stn.train_ds.skip(int(0.7*stn.train_ds.cardinality().numpy()))
stn.train_ds = stn.train_ds.take(int(0.7*stn.train_ds.cardinality().numpy()))
stn.test_ds = final_stn_loader(args.data_dir+args.output_dir+"final_gan/test/")

print("\nRetraining classifier with gan output")
stn.train()

print("Final evaluation")
stn.evalutation()