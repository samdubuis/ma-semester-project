# ma-semester-project
Master semester project @ EPFL 2020 autumn semester

Startup Company “Global ID", Swiss startup specialized in cyber-security through high levelof security and confidentiality, and fighting against identity thief has developed a 3D fingervein biometric identification system.

To improve and verify the performance of their biometric identification system, they have toget a large number of the finger vein images in person. However it is very cumbersome, time consuming, and labor intensive to collect a large-scale dataset from subjective experiments. To address that, in this project, we will generate a large number of finger vein images from a small number of the real finger vein images using deep generative model. For this purpose, we will study the generative adversarial network (GANs). Furthermore, we will design a GAN-based data augmentation algorithm for synthesizing realistic finger vein images.

GANs are a type of machine learning, more precisely deep learning, framework where two different neural networks compete against each other. The generator tries to produce a result which the discriminator evaluates until a point where the generator is able to fool its adversary.At first we use a spatial transformer network to improve classification and processing of the data. Then we use a cycleGAN in order to augment the dataset. In the cycleGAN part we have added to our framework an additional loss based on the classification of the finger veins to improve the result images.

Finally the full framework with tunable parameters was implemented, with default valueshaving been pre-optimized.
