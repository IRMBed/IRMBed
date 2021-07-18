### IRMBed: the test bed for IRM methods on deep models
HI, this is the project of ICML2021 workshop UDL paper: Yong Lin, Qing Lian and Tong Zhang. An Empirical Study of Invariant Risk Minimization on Deep Models.

We aim to study IRM methods on deep models. Now, Resnet18 amd Resnet50 are supported.

We offer the implementation of IRMv1[1], InvRat[2], REx[3], RVP[4], IRMGame[5].

### The dataset:MNIST-Cifar10
Inspired by [6], we construct a Cifar-MNIST dataset, in which each image is synthesized by concating two component images, one from Cifar and the other from MNIST. We make the Cifar and MNIST component behave as invariant and spurious features, respectively.  Specifically, the label of the synthesized image is generated from the Cifar component and the MNIST component exhibit a high but unstable correlation with the label. Following [1], we construct several environments. The MNIST component's correlation with label is changing across different environments while the Cifar component's correlation remains invariant.

### Results
We consider two settings for the training sets: 1). 2 Env: the training data contains two environments, in which the spurious correlations are 99.9\% and 80.0\%, respectively, 2). 4 Env: the training data contains four environments, in which the spurious correlations are 99.9%, 95.0%, 90.0%, 80.0%, respectively. In both settings, we set the correlation of spurious features to 10% in test environment to see whether the learned model relies on the spurious feature. We also add a certain level (10%) of noise to label as [1] does. 

### How to run the code
How to run the code?
```
python nrun_expt_irmv12.py  -d SPCM --cons_ratios 0.999_0.95_0.9_0.8_0.1 --label_noise_ratio 0.10 --irm_type invrat  --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet18_invrat_ec --n_epoch 100  --opt SGD  --irm_penalty --irm_penalty_weight 100 --num_inners 1  --seed 0
```
### set the pramater
"--cons_ratios" # setting of environment. 
cons_ratios specify the correlation of the spurious feature with the label for both training and testing data set.
For example,  "0.999_0.95_0.9_0.8_0.1" stands for 4 enviornments in training datset, whose spurious correlations are (0.999, 0.95, 0.9, 0.8) and 1 enviornment in testing dataset, whose spurious correlation is 0.1;

"--label_noise_ratio" # noise ratio of label

"--irm_type" # the IRM method to run, choose in {invrat, irmv1, rex, rvp, irmgame}

"--model" # the deep network to run, choose in {resnet18, resnet50, resnet18_invrat_ec, resnet18_invrat_ec, resnet50_invrat_ec, resnet50_invrat_ec}

"--irm_penalty_weight" # penalty weight

"--num_inners" # number of inner steps for invrat.

Have fun with it.

### References


[1] Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D.  Invariant risk minimization.

[2] Chang, S., Zhang, Y., Yu, M., & Jaakkola, T.  Invariant rationalization.

[3] Krueger, D., Caballero, E., Jacobsen, J. H., Zhang, A., Binas, J., Zhang, D., ... & Courville, A.  Out-of-distribution generalization via risk extrapolation (rex).

[4] Xie, C., Chen, F., Liu, Y., & Li, Z. Risk variance penalization: From distributional robustness to causality

[5] Ahuja, K., Shanmugam, K., Varshney, K., & Dhurandhar, A. Invariant risk minimization games.

[6] Shah, Harshay and Tamuly, Kaustav and Raghunathan, Aditi and Jain, Prateek and Netrapalli, Praneeth. The pitfalls of simplicity bias in neural networks

