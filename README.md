### IRMBed: the test bed for IRM methods on deep models
HI, this is the project of ICML2021 workshop UDL paper by Yong Lin, Qing Lian and Tong Zhang: [`An Empirical Study of Invariant Risk Minimization on Deep Models.`](http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-044.pdf)

We aim to study IRM methods on deep models. Now, Resnet18 amd Resnet50 are supported.

We offer the implementation of IRMv1[1], InvRat[2], REx[3], RVP[4], IRMGame[5].
### Requirements
The program runs with the following packages.
```
pandas==1.1.5
pytorch-transformers==1.2.0
torch==1.3.1
torchvision==0.4.2
tqdm==4.26.0
numpy==1.19.4
```
To install the required packages, please run `pip install -r requirements.txt`
### Datasets
#### MNIST-Cifar10
Inspired by [6], we construct a Cifar-MNIST dataset, in which each image is synthesized by concating two component images, one from Cifar and the other from MNIST. We make the Cifar and MNIST component behave as invariant and spurious features, respectively.  Specifically, the label of the synthesized image is generated from the Cifar component and the MNIST component exhibit a high but unstable correlation with the label. Following [1], we construct several environments. The MNIST component's correlation with label is changing across different environments while the Cifar component's correlation remains invariant. The illustration of the dataset is shown as following:
![Illustration of the synthetic dataset from CIFAR-10 and MNIST. We first randomly select two classes ("car" and "bird") from CIFAR-10. Then each CIFAR-10 image is concatanated with an image from MNIST ("0" and "1"). The CIFAR-10  component serves as the invariant feature and the label is generated from the CIFAR-10 component. The MNIST component serves as the spurious feature. The MNIST component is highly correlated with the label in the training dataset,  however, the correlation reverses in the testing dataset. ](./dataset_illustration.png)

#### User Input Dataset
IRMBed also provide interface for user specified dataset. Two steps are needed to 
* Implement the `SpuriousDataset` in `data/data.py` by passing `x`(the feature tensor), `y`(the label tensor), `e`(the environment index tensor) and `sp`(optional, the tensor of indicators that whether the spurious features are consistent with the label).
* Pass the Trainging and Testing dataset to `data_processor` in `run.py`. In addition, one also need to  provide the total environment number `env_nums` and class number`n_classes`.  

### Outputs
The project outputs the trained model and prints the performance of the model.
  * The trained model is saved in `results/model.pth`;
  * The performance of the model on each environment of training and testing dataset at each epoch: 
    * `loss` is the empirical loss, `penalty` is the invariance penalty and `main_loss` is the weighted sum of `loss` and `penalty` by the irm penalty weight;
    * `acc` is the precision of the model on the data from a specific environment; `major_acc` and `minor_acc`

### Results
We consider two settings for the training sets: 1). 2 Env: the training data contains two environments, in which the spurious correlations are 99.9\% and 80.0\%, respectively, 2). 4 Env: the training data contains four environments, in which the spurious correlations are 99.9%, 95.0%, 90.0%, 80.0%, respectively. In both settings, we set the correlation of spurious features to 10% in test environment to see whether the learned model relies on the spurious feature. We also add a certain level (10%) of noise to label as [1] does. 

|Model|  ResNet-18(2Env)   | ResNet-18(4Env)  | ResNet-50(2Env) | ResNet-50(4Env)|
|----|  ----  | ----  |  ----  | ----  |
|  ERM   | 39.5±0.4 |35.7±0.7 |38.1±0.8|32.0±2.1|
|  IRMv1 | 70.8±0.4|53.6±3.1|43.3±12.5|51.8±4.9|
|  REx   | 52.5±2.2 |44.5±1.7 |51.1±2.2|37.6±7.3|
|  RVP   | 49.4±3.1 |49.4±1.1 |50.2±0.3|50.1±0.7|
|  IRMGame   | 43.0±1.9 | 37.2±2.8|44.5±1.5|40.1±5.5|
|  InvRat-EB   | 77.6±2.0 | 68.6±9.4|68.5±3.6|36.8±3.1|
|  InvRat-EC   | 75.8±1.2 | 73.4±4.0 |81.2±1.3|77.9±4.2|


For detailed explanation of these results, please refer to our workshop paper.
### Quick Start
How to run the code?Here is an exmaple for InvRat-EC.
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python run.py  -d SPCM --cons_ratios 0.999_0.95_0.9_0.8_0.1 --label_noise_ratio 0.10 --irm_type invrat  --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet18_invrat_ec --n_epoch 100  --opt SGD  --irm_penalty --irm_penalty_weight 100 --num_inners 1  --irm_anneal_epochs 2 --seed 0
```
### Set the pramater
Here are some important parameters for the program.
* `cons_ratios` # setting of environment. 
cons_ratios specify the correlation of the spurious feature with the label for both training and testing data set.
For example,  `0.999_0.95_0.9_0.8_0.1` stands for 4 enviornments in training datset, whose spurious correlations are (0.999, 0.95, 0.9, 0.8) and one enviornment in testing dataset, whose spurious correlation is 0.1;

* `label_noise_ratio`" # noise ratio of label, in this provided example, there are 10% label noise.

* `irm_type`" # the IRM method to run, choose in {`invrat`, `irmv1`, `rex`, `rvp`, `irmgame`}

* `model`" # the deep network to run, choose in {`resnet18`, `resnet50`, `resnet18_invrat_eb`, `resnet50_invrat_eb`, 
`resnet18_invrat_ec`, `resnet50_invrat_ec`}. For models except `invrat`, please choose `resnet18` or `resnet50`.

* `irm_penalty_weight`" # penalty weight

* `num_inners`" # number of inner steps for `invrat`.


### Citation
```
@article{yong2021empirical,
  title={An Empirical Study of Invariant Risk Minimization on Deep Models},
  author={Yong Lin and Qing Lian and Tong Zhang},
  journal={ICML 2021 Workshop on Uncertainty and Robustness in Deep Learning},
  year={2021}
}
```

### References
[1] Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D.  Invariant risk minimization.

[2] Chang, S., Zhang, Y., Yu, M., & Jaakkola, T.  Invariant rationalization.

[3] Krueger, D., Caballero, E., Jacobsen, J. H., Zhang, A., Binas, J., Zhang, D., ... & Courville, A.  Out-of-distribution generalization via risk extrapolation (rex).

[4] Xie, C., Chen, F., Liu, Y., & Li, Z. Risk variance penalization: From distributional robustness to causality

[5] Ahuja, K., Shanmugam, K., Varshney, K., & Dhurandhar, A. Invariant risk minimization games.

[6] Shah, Harshay and Tamuly, Kaustav and Raghunathan, Aditi and Jain, Prateek and Netrapalli, Praneeth. The pitfalls of simplicity bias in neural networks

### Contact
Please submit a github issue if you have any problem on this project.
You can also send email to `ylindf@connect.ust.hk` for personal contact.
If you are also interested in IRM and want to discuss with me, you can also chat with me(`linyongverycool`) by Wechat. 
