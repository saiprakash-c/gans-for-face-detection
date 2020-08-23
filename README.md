# GANs for face detection

This project is part of EECS 504 - Computer Vision course at the University of Michigan Ann Arbor. 

Detection and recognition of tiny faces (ex. 10*10 pixels) in an image is hard as they lack fine details of a face. This
problem can be solved by enhancing these images using the knowledge of how human faces look before passing into the
detector.

We used Generative adverserial networks to enhance the tiny faces so that they are identified by discriminator. 
The Generator network consists of two sub-networks in sequence. First sub-network is an up-sampling network which takes a
low-resolution image and increases the resolution by 4 times. The second sub-network is Refinement network, which takes this blurred
image, refines it and outputs a clear super-resolution image. Discriminator Network is
based on VGG19 network. It distinguishes generated vs true high-resolution images and faces vs non-faces jointly.

## Prerequisites

Make sure the following libraries are installed 

1. Python 3.x
2. Numpy >=1.15
3. Tensorflow >=1.12
4. scikit-image >=0.14.1
5. matplotlib >=3.0.x


## Installing and running

git clone https://www.github.com/saiprakash-c/gans-for-face-detection.git

Download the data from [here](http://shuoyang1213.me/WIDERFACE/) and keep it in the appropriate folders as follows. 

Pretrained model of VGG19 is obtained from [here](https://mega.nz/file/xZ8glS6J#MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

### Training

- Keep WIDER_train folder, labels.csv and vgg19.npy (weights of vgg19) in the directory above the main directory
- Run gan.py
- Stores tensorflow checkpoint files for each batch and each epoch
- The full model gets stored as `model_full.ckpt` in the main directory itself

### Testing

- Keep WIDER_val folder, vgg19.npy in the directory above the main directory
- Run evaluation.py
- Uses `model_full.ckpt` by default
- Prints the results on the terminal

## Report

The report for this project can be found [here](https://drive.google.com/drive/folders/16yEuUZwcpTRzwYLKJfcYLGylxw-Vp2g_)



