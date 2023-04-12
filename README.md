# RND_for_RestMin_v1
Solving RestMin_v1 with  Random-Network-Distillation, in the framework of Stable Baseline 3.

> Random Network Distillation (RND) is a technique used in the field of reinforcement learning that focuses on exploration in unknown environments. RND utilizes a randomly initialized neural network called the "exploration network" to generate a pseudo-count for states visited by the agent during training. This pseudo-count is then used as an intrinsic reward signal to encourage the agent to explore regions of the state space that are less visited.
>
> Compared to other exploration strategies, such as epsilon-greedy or softmax exploration, RND has several benefits, including its simplicity and scalability. Additionally, because RND does not require any information about the task at hand, it can be easily applied to many different types of reinforcement learning problems. Overall, Random Network Distillation has proven to be a highly effective method for improving the exploration capabilities of reinforcement learning agents.

In this enviroment, the external reward can only be obtained in terminal state.
Thus, before reaching the optimal solution, the decision of the agent is entirely guided by intrinsic rewards provided by RND.
BTW, it is worth noting that since RND always gives high rewards for rare states, it is unlikely to achieve 100% perfect solution with this algorithm alone.

## Size=5, Mode=0
Repo [here](https://github.com/wwsyan/RND_for_RestMin_v1/tree/main/size5_mode0), 
run file [here](https://github.com/wwsyan/RND_for_RestMin_v1/blob/main/size5_mode0/run.py).

| Episode length | Episode reward |
| :---: | :---: |
|<img src="size5_mode0/images/rollout_ep_len_mean.png">|<img src="size5_mode0/images/rollout_ep_rew_mean.png">

## Size=6, Mode=0
### n_steps: 1024 VS 2048
| Episode length | Episode reward | Info |
| :---: | :---: | :---: |
|<img src="size6_mode0/images/ep_len_1.png">|<img src="size6_mode0/images/ep_rew_1.png">||
### RND: observation normalization VS non-normalization
### reward: single VS diverse
 
