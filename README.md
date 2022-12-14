# Projected Nesterov Momentum Iterative Fast Gradient Sign Method (PNMI-FGSM)
**Deep Learning Final Project - Yotam Granov, Spring 2022**

The main script to be run is `run_attacks.py`, which is run in the command line as follows:

```
srun -c 2 --gres=gpu:1 --pty -w lambda3 python run_attacks.py --model-name {VO_MODEL_FILE} --test-dir {DATA_LOCATION} --save_best_pert --attack {ATTACK_TYPE} --attack_k {NUMBER_OF_ITERATIONS} --alpha {LEARNING_RATE} --mu {MOMENTUM}
```
For example, one might input the following command in order to train the PNMI-FGSM attack over 200 iterations with alpha=0.001 and mu=0.001:
```
srun -c 2 --gres=gpu:1 --pty -w lambda3 python run_attacks.py --model-name tartanvo_1914.pkl --test-dir "VO_adv_project_train_dataset_8_frames" --save_best_pert --attack pnmi --attack_k 200 --alpha 0.001 --mu 0.001
```

The file `pnmi_fgsm.py` (located inside the `\attacks` folder) contains the `PNMI-FGSM` class, which is the implementation of our model. The `perturb()` method conducts the actual attack generation, while the `PNMI_FGSM_gradient_ascent_step()` method conducts one iteration of gradient descent and produces the next iteration of the perturbation. The `utils.py` file was edited to include an option for our model to be used when the command `--attack pnmi` is called at the command line when `run_attacks.py` is run.

There are no other new files from the original assignemnt codebase.