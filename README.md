### Implementation of IRM methods with CifarMNIST

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
