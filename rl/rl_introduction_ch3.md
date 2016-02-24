<!-- toc -->

@(Cabinet)[ml_dl_rl, aca_book|published_gitbook]

date = "2016-01-10"

# RL an introduction, Ch3

> p64

## 3.1 The Agent-Environment Interface

At each time step, the agent implements a mapping from states to probabilities of selecting each possible action. This mapping is called the agent's `policy` and is denoted $\pi_{t}$, where $\pi_{t}(s,a)$ is the probability that $a_{t}=a$ if $s_{t}=s$. 

## 3.3 Returns
In general, we seek to maximize the `expected return`, where the return, $R_{t}$, is defined as some specific function of the reward sequence. In the simplest case the return is the sum of the rewards: $R_{t}=r_{t+1}r_{t+2}+r_{t+3}+...+r_{T}$. 

## 3.4 Unified Notation for Episodic and Continuing Tasks
The return is $R_{t}=\sum_{k=0}^{T}\gamma^{k}r_{t+k+1}$, including the possibility that $T=\infty$ or $\gamma=1$. 

## 3.6 Markov Decision Processes
A reinforcement learning task that satisfies the Markov property is called Markov decision process, or MDP. A particular finite MDP is defined by its state and action sets and by the one-step dynamics of the environment. 

Given any state and action, $s$ and $a$, the probability of each possible next state, $s'$, is $\mathcal{P}_{ss'}^{a}=Pr\{s_{t+1}=s'|s_{t}=s,a_{t}=a\}$. These quantities are called `transition probabilities`. 

Similarly, given any current state and action, $s$ and $a$, together with any next state, $s'$, the expected value of the next reward is $\mathcal{R}_{ss'}^{a}=E\{r_{t+1}|s_{t}=s,a_{t}=a,s_{t+1}=s'\}$. 

These quantities, $\mathcal{P}_{ss'}^{a}$ and $\mathcal{R}_{ss'}^{a}$, completely specify the most important aspects of the dynamics of the finite MDP (only information about the distribution of rewards around the expected value is lost). 

## 3.7 Value Functions
Almost all reinforcement learning algorithms are based on estimating `value functions` -- functions of states (or of state-action pairs) that estimate *how good* it is for the agent to be in a given state (or how good it is to perform a given action in a given state). The notation of "how good" here is defined in terms of future rewards that can be expected, or, to be precise, in terms of expected return. Also, value functions are defined w.r.t. particular `policies`. 

A policy, $\pi$, is a mapping from each state, $s\in\mathcal{S}$, and action, $a\in\mathcal{A}(s)$, to the probability $\pi(s,a)$ of taking action $a$ when in state $s$. Informally, the `value` of a state $s$ under a policy $\pi$, denoted $V^{\pi}(s)$, is the expected return when starting in $s$ and following $\pi$ thereafter. For MDPs, we define $V^{\pi}(s)$ formally as: $V^{\pi}(s)=E_{\pi}\{R_{t}|s_{t}=s\}=E_{\pi}\{\sum_{k=0}^{\infty}\gamma^{k}r_{r+k+1}|s_{t}=s\}$. We call the function $V^{\pi}$ the `state-value` function for policy $\pi$. 

Similarly, we define the value of taking action $a$ in state $s$ under a policy $\pi$, denoted $Q^{\pi}(s,a)$, as the expected return starting from $s$, taking the action $a$, and thereafter following policy $\pi$: $Q^{\pi}(s,a)=E_{\pi}\{R_{t}|s_{t}=s,a_{t}=a\}=E_{\pi}\{\sum_{k=0}^{\infty}\gamma^{k}r_{r+k+1}|s_{t}=s,a_{t}=a\}$. We call $Q^{\pi}$ the `action-value` function for policy $\pi$.

(in p83), The `Bellman equation` for $V^{\pi}$ is: $V^{\pi}(s)=\sum_{a}\pi(s,a)\sum_{s'}\mathcal{P}_{ss'}^{a}[\mathcal{R}_{ss'}^{a}+\gamma V^{\pi}(s')]$. 

It expresses a relationship between the value of a state and the values of its successor states. 

## 3.8 Optimal Value Functions
Solving a reinforcement learning task means, roughly, finding a policy that achieves a lot of rewards over the long run. Value functions define a partial ordering over policies. A policy $\pi$ is defined to be better than or equal to a policy $\pi'$ if its expected return is greater than or equal to that of $\pi'$ for `all` states. There is *always* at least one policy that is better than or equal to all other policies. Although there may be more than one *optimal policy*, we denote all the optimal policies by $\pi^{*}$. They share the same state-value function, called the `optimal state-value` function, denoted $V^{*}$, and defined as $V^{*}(s)=max_{\pi}V^{\pi}(s)$, for all $s\in\mathcal{S}$. 
Optimal policies also share the same `optimal action-value` function, denoted $Q^{*}$, and defined as $Q^{*}(s,a)=max_{\pi}Q^{\pi}(s,a)$, for all $s\in\mathcal{S}$ and $a\in\mathcal{A}(s)$, this function gives the expected return for taking action $a$ in state $s$ and thereafter following an optimal policy. 

We can write $Q^{*}$ in terms of $V^{*}$ as follows: $Q^{*}(s,a)=E\{r_{t+1}+\gamma V^{*}(s_{t+1})|s_{t}=s,a_{t}=a\}$. 

> p89

Intuitively, the Bellman optimality equation expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state: $V^{*}(s)=max_{a}\sum_{s'}\mathcal{P}_{ss'}^{a}[\mathcal{R}_{ss'}^{a}+\gamma V^{*}(s')]$. 

The Bellman optimality equation for $Q^{*}$ is: $Q^{*}(s,a)=\sum_{s}\mathcal{P}_{ss'}^{a}[\mathcal{R}_{ss'}^{a}+\gamma max_{a'}Q^{*}(s',a')]$. 

For finite MDPs, the Bellman optimality equation has a unique solution independent of the policy. The Bellman optimality equation is actually a system of equations, one for each state, so if there are $N$ states, then there are $N$ equations in $N$ unknowns. 

## 3.9 Optimality and approximation
