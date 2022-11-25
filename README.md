# Unofficial pytorch code for Self-supervised Light Field View Synthesis Using Cycle Consistency.

This repository contains unofficial pytorch implementation of Self-supervised Light Field View Synthesis Using Cycle Consistency. [MMSP2020](https://ieeexplore.ieee.org/document/9287105)

## Requirement
* Ubuntu 18.04
* Python 3.6
* Pyorch 1.7
* Matlab

## Dataset
We provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders. The real-world training data is available in [SIGGRAPH/ACM Trans. Graph.](https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/).

## Code
In this implementation, we only provide a model with 5-layer channel attention-based residual blocks, and the original paper utilized a pre-trained video frame interpolation network. You can design your own framework for training.

* If you find any mistakes/bugs, please contact us.

* For training, run:
  ```python
  python train.py

```
## Acknowledgement
Our work and implementations are based on the following projects: <br> 
[LF-DFnet](https://github.com/YingqianWang/LF-DFnet)<br> 
[LF-EASR](https://github.com/GaoshengLiu/LF-EASR)<br> 
We sincerely thank the authors for sharing their code and amazing research work!
