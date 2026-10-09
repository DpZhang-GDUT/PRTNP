# PRTNP

### Paper
Coarse-to-fine residual tensor network framework for test-time adversarial purification

### Environment
Environment configuration details are provided in requirements.txt.

### Dataset

#### CIFAR-10
The `adv/cifar10_32/AA-linf/sc_512_8.pth` and `adv/cifar10_32/AA-linf/rc_cifar10` files contain adversarial examples generated using AutoAttack against a standard classifier and a robust classifier, respectively.

#### CIFAR-100
The `adv/cifar100_32/AA-linf/512_8.pth` file contains adversarial examples generated using AutoAttack against a robust classifier.

#### ImageNet
The adversarial examples for ImageNet are available at the following link:
https://huggingface.co/datasets/ZhangDongping/ImageNet-AutoAttack

### CIFAR-10 under AutoAttack

#### pipeline+standard classfier
For evaluation on CIFAR-10 using AutoAttack with an $l_\infty$ perturbation budget of $\epsilon = 8/255$, run:

```bash
python train.py --config configs/sc_cifar10.yaml
```

#### pipeline+robust classfier
For evaluation on CIFAR-10 using AutoAttack with an $l_\infty$ perturbation budget of $\epsilon = 8/255$, run:

```bash
python train.py --config configs/rc_cifar10.yaml
```

### CIFAR-100 under AutoAttack
For evaluation on CIFAR-100 using AutoAttack with an $l_\infty$ perturbation budget of $\epsilon = 8/255$, run:

```bash
python train.py --config configs/cifar100.yaml
```

### ImageNet under AutoAttack

For evaluation on ImageNet using AutoAttack with an $l_\infty$ perturbation budget of $\epsilon = 4/255$, run:

```bash
python train.py --config configs/imagenet.yaml
```
