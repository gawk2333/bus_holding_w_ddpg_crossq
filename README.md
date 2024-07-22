# bus_holding_ddpg

This is a bus holding control problem using reinforcement learning approach based on research from Jiawei Wang 
(https://github.com/TransitGym/TransitGym.github.io/tree/master/research/transit_control_research_caac)

To create the python environment (conda):
```
conda env create -f environment.yaml

conda activate bus-ddpg
```

To train split-attention DDPG model:

```
python main.py ---model=ddpg_split_attention
```

To use trained split-attention DDPG model :
```
python main.py --control=2 --model=ddpg_split_attention --train=0  --restore=1 --episode=100 --para_flag=A_0_1 --seed=1 --vis=1
```

