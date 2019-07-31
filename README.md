# DeepHashing
This is a basic implementation of Dual Asymmetric Deep Hashing Learning paper [DADH](https://arxiv.org/abs/1801.08360).

The main aim is to generate binary vectors corresponding to an image which then can later be used for hashing the images.

Squeezenet is used to implement both the branches of DADH, with loss function described in the paper itself.

I had tried running the code on FVC2002, FVC2004 dataset containing biometric images of fingerprints which are captured using mulitple sensors.
