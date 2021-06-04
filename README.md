### Implementation of IRM methods

How to run the code?

python nrun_expt_irmv12.py  -d SPCM --cons_ratios 0.999_0.95_0.9_0.8_0.1 --train_envs_ratio -1 --label_noise_ratio 0.10 --oracle 0 --irm_type invrat  --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet18_invrat_eb --n_epoch 100  --opt SGD  --irm_penalty --irm_penalty_weight 100 --penalty_wlr 1 --penalty_welr 1 --irm_anneal_epochs 2 --num_inners 1 --irm_anneal_type jump  --pretrained 1  --seed 0

Have fun with it.
