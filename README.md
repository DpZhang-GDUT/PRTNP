# PRTNP

### CIFAR-10 under AutoAttack

For evaluation on CIFAR-10 using AutoAttack with an $l_\infty$ perturbation budget of $\epsilon = 8/255$, run:

```bash
python train.py --config configs/cifar10.yaml
```

### ImageNet under AutoAttack

For evaluation on ImageNet using AutoAttack with an $l_\infty$ perturbation budget of $\epsilon = 4/255$, run:

```bash
python train.py --config configs/imagenet.yaml
```
