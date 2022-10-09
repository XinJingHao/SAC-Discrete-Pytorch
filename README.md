# SAC-Discrete-Pytorch
This is a **clean and robust Pytorch implementation of Soft-Actor-Critic** on **discrete** action space. Here is the training curve:  

<img src="https://github.com/XinJingHao/SAC-Discrete-Pytorch/blob/main/imgs/sacd_result.jpg"/>
All the experiments are trained with same hyperparameters. Other RL algorithms by Pytorch can be found

[here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch)

A quick render here:

![avatar](https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/Render%20of%20DDQN.gif)


## Dependencies
gym==0.19.0  
numpy==1.21.2  
pytorch==1.8.1  
tensorboard==2.5.0

## How to use my code
### Train from scratch
run **'python main.py'**, where the default enviroment is CartPole-v1.  
### Play with trained model
run **'python main.py --write False --render True --Loadmodel True --ModelIdex 50'**  
### Change Enviroment
If you want to train on different enviroments, just run **'python main.py --EnvIdex 1'**.  
The --EnvIdex can be set to be 0 and 1, where   
'--EnvIdex 0' for 'CartPole-v1'  
'--EnvIdex 1' for 'LunarLander-v2'   
### Visualize the training curve
You can use the tensorboard to visualize the training curve. History training curve is saved at '\runs'
### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'
### References
Christodoulou P. Soft actor-critic for discrete action settings[J]. arXiv preprint arXiv:1910.07207, 2019.

Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor[C]//International conference on machine learning. PMLR, 2018: 1861-1870.

Haarnoja T, Zhou A, Hartikainen K, et al. Soft actor-critic algorithms and applications[J]. arXiv preprint arXiv:1812.05905, 2018.

