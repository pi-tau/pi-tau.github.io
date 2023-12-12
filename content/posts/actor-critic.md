---
sitemap:
    changefreq: weekly
    priority: 0.8
title: Which policy gradient algorithm are you using?
date: 2023-08-25
description: An in depth, detailed review of some of the most famous policy gradient algorithms. Starting from vanilla policy gradient, then discussing actor-critic algorithms and finally visiting PPO. Reference implementations are shown and step-by-step improvements are discussed.
keywords: ["reinforcement learning", "policy gradient", "vpg", "ppo", "actor-critic", "a2c", "gae", "parallel environment", "entropy regularization", "atari", "Lunar Lander"]
mathjax: true
ToC: true
---


So, you probably all know the formula for updating the policy network using the
policy gradient theorem:

$$ \nabla_\theta J(\theta) = E_{a, s \sim \pi_\theta} \Big[ \nabla \log \pi_\theta(a|s) R(s, a) \Big]. $$

Here, the action $a$ is drawn from the current policy $\pi_\theta$. The state
$s$ is sampled from the state distribution function of the environment by
following $\pi_\theta$. And $R(s, a)$ is the return obtained for the state $s$
if you select action $a$ and would continue to follow the policy $\pi_\theta$.
In order to update, you perform a rollout with your current policy, you estimate
the gradient using this formula, and you backpropagate. I will not go into
detail where this formula comes from, but if you want to read more you can check
out [this comprehensive blog
post](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#proof-of-policy-gradient-theorem)
from Lilian Weng.

You've probably also heard that there are different algorithms for computing the
return $R$, e.g. Monte-Carlo, actor-critic, A2C, GAE, etc. What we will do is
start from the most basic algorithm - vanilla policy gradient, and continuously
improve on it until we arrive at the most successful and widely used algorithm
today - PPO. Each agent that we implement will conform to the following api:
```python
class Agent:
    def __init__(self, policy_network, value_network):
        self.policy_network = policy_network
        self.value_network = value_network

    @torch.no_grad()
    def policy(self, s):
        return Categorical(logits=self.policy_network(s))

    @torch.no_grad()
    def value(self, s):
        return self.value_network(s)

    def update(self, states, actions, rewards):
        raise RuntimeError("not implemented")
```

Disclaimer: The code snippets given in this blog post will probably not work
directly out of the box. The full working code as well as results on training
these agents on the Atari game LunarLander can be seen on
[github](https://github.com/pi-tau/playing-with-RL-models).


## VANILLA POLICY GRADIENT
In vanilla policy gradient (VPG) there is nothing fancy; we just calculate the
return $R$ as a Monte-Carlo estimate. We perform a full episode rollout and we
collect the states, actions, and rewards ($s_t$, $a_t$, $r_{t+1}$) at each step
until we reach a terminal state. Once we've reached the terminal state we can
estimate the return for each of the visited states:

$$ R_t = \sum_{i=t}^{T} r_{i+1}. $$

Finally we compute the gradient:

$$ \nabla_\theta J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \nabla \log \pi_\theta (a_t | s_t) R_t. $$

Each of the states was visited using the current policy and the state
distribution function of the environment, so for each state-action pair we can
compute an estimate of the gradient. Then, we average over the samples, much
like a mini-batch update in supervised learning.

To implement this algorithm we will first make use of a function that describes
an agent-environment loop. We will assume that `env` is a non-vectorized
environment that conforms to the [OpenAI Gym
API](https://gymnasium.farama.org/api/env/).
```python
def environment_loop(agent, env, num_iters):
    for _ in range(num_iters):
        states, actions, rewards = [], [], []
        s, _ = env.reset()
        done = False
        while not done:
            states.append(s)
            act = agent.policy(s).sample() # policy() returns a distribution
            s, r, t, tr, _ = env.step(acts)
            actions.append(act)
            rewards.append(r)
            done = (t | tr)

        agent.update(np.array(states), np.array(actions), np.array(rewards))
```

The environment loop will run multiple iterations and on each iteration it will
rollout an episode using the current policy, and then it will update the policy
using the sampled data.

```python
class VPGAgent(Agent):
    def update(self, states, actions, rewards):
        T = rewards.shape[0]
        returns = rewards @ np.tril(np.ones(T))
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        logits = self.policy_network(states)
        logp = F.cross_entropy(logits, actions, reduction="none")
        loss = (logp * returns).mean()

        self.optim.zero_grad() # assume optim is defined in agent.__init__()
        loss.backward()
        self.optim.step()
```

The returns are calculated by simply adding the rewards obtained from the given
time-step onwards. I've used a simple vector-matrix multiplication here where I
multiply the rewards vector by a lower-triangular matrix of ones in order to get
the desired sums.

One trick that is usually done in practice is to normalize the returns before
computing the gradient. This is done in order to stabilize the training of the
neural network. Note that this modification [does not change the
expectation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#expected-grad-log-prob-lemma)
of the gradient estimation.

<!--
How is this similar or different from [REINFORCE](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)[^REINFORCE]?
 -->

## VPG WITH BASELINE
Since the single-sample Monte-Carlo estimate of the returns might have a lot of
variance, we may want to perform multiple rollouts instead of just one. That is,
rollout several episodes and only then calculate the gradient and backpropagate.
However, it is not clear how helpful that would be because you will still get
(mostly) single-sample Monte-Carlo estimates; it's just that you will have more
data points in the batch. To see why is that, consider rolling out a second
episode with the same policy. At some step $t_d$ your second rollout will
diverge from the first rollout and from then onward you will sample new states
and for the returns of these states you will have single-sample estimates. Only
for the states before step $t_d$ you will have a more accurate estimate.

A much better approach to reduce the variance of the estimation would be to
[add a baseline](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients)
to the estimate of the return.

We will baseline the returns using the value function $V^{\pi}(s)$. As a side
note: the [best baseline in terms of variance
reduction](https://arxiv.org/abs/1301.2315)[^Baseline] is actually:

$$E_{a \sim \pi_\theta} \Bigg[ \frac{
    \nabla \log^2 \pi_\theta (a|s) R(s, a) }{
    \nabla \log^2 \pi_\theta (a|s)
} \Bigg], $$

and the baseline that we are using is $V(s) = \mathbb{E} \Big[ R(s, a) \Big]$.

The formula for the gradient now becomes:

$$
\begin{align}
    \nabla_\theta J(\theta) &= E_{a, s \sim \pi_\theta} \Big[ \nabla \log \pi_\theta(a|s) R(s, a) - V(s) \Big] \\\\
    &= E_{a, s \sim \pi_\theta} \Big[ \nabla \log \pi_\theta(a|s) A(s, a) \Big],
\end{align}
$$

where $A(s, a)$ is the *advantage*, describing how much better it is to take
action $a$ compared to the other actions.

The value function is approximated using a second neural network which is
trained concurrently with the policy. The update function will be modified like
this:
```python
class VPGAgent(Agent):
    def update(self, states, actions, rewards):
        T = rewards.shape[0]
        returns = rewards @ np.tril(np.ones(T))
        adv = returns - self.value(states)

        # Update the policy network.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logits = self.policy_network(states)
        logp = F.cross_entropy(logits, actions, reduction="none")
        loss = (logp * adv).mean()

        self.policy_optim.zero_grad() # assume defined in agent.__init__()
        loss.backward()
        self.policy_optim.step()

        # Update the value network.
        for o, r in DataLoader((states, returns), self.batch_size):
            pred = self.value_network()
            vf_loss = F.mse_loss(pred, r)
            self.value_optim.zero_grad() # assume defined in agent.__init__()
            vf_loss.backward()
            self.value_optim.step()
```

Note that even though we baseline the returns we are still normalizing the
advantages after that. This effectively means that we are using a different
baseline and not $V(s)$, but as already explained this will not change the
expectation of the gradient. It will simply improve the training of our network.

<!--
Also note that we could update the value network **before** computing the advantages with which we update the policy network. However, later we **must** update only after computing the advantages.
-->


## ONLINE ADVANTAGE ACTOR-CRITIC
Now that we have a value network we can actually compute the return by
bootstrapping instead of using a Monte-Carlo estimate:

$$ R(s_t, a_t) = r_{t+1} + V(s_{t+1}). $$

This is the so-called actor-critic setup, where we have an actor (the policy
network) that selects actions to perform the rollout and a critic (the value
network) that is used to compute the returns, i.e. it grades the performance.

Combining the actor-critic with a baseline we get the advantage actor-critic
([A2C](https://arxiv.org/abs/1602.01783)[^A2C]). The formula for the gradient
now becomes:

$$ \nabla_\theta J(\theta) = E_{a, s, r \sim \pi_\theta} \Big[ \nabla \log \pi_\theta(a|s) \Big( r + V(s') - V(s) \Big) \Big]. $$

Here $r$ is the reward obtained when performing action $a$, and $s'$ is the next
state of the environment.

With this setup we actually don't have to run episodes until the end. In fact we
can update the policy (and the value network) at every single step. But that is
not a very good idea because we will be estimating the gradient using a single
sample. Instead, what we could do is:
 * either run multiple environments in parallel in order to obtain multiple
   samples at every step,
 * or rollout the episode for several steps and only then perform the update.

We will actually do both.

```python
def environment_loop(agent. env, num_iters, steps):
    num_envs = env.num_envs
    s, _ = env.reset()

    states = np.zeros(
        shape=(steps, num_envs, *env.single_observation_space.shape),
        dtype=np.float32,
    )
    next_states = np.zeros_like(states)
    actions = np.zeros(shape=(steps, num_envs), dtype=int)
    rewards = np.zeros(shape=(steps, num_envs), dtype=np.float32)
    done = np.zeros(shape=(steps, num_envs), dtype=bool)

    for _ in range(num_iters):
        for i in range(steps):
            states[i] = s
            acts = agent.policy(s).sample()
            s, r, t, tr, info = env.step(acts)

            actions[i] = acts
            rewards[i] = r
            next_states[i] = s
            done[i] = (t | tr)

        agent.update(states, actions, rewards, next_states, done)
```

We will now assume that `env` is a [vectorized
environment](https://gymnasium.farama.org/api/vector/). The environment loop
will run multiple iterations and on each iteration it will rollout the policy
for a fixed amount of steps. Note that vectorized environments will auto-reset
any sub-environment that is terminated or truncated. This means that a given
segment of experiences might contain data from multiple episodes, and for this
reason we will use the `done` tensor to indicate which of the states are
terminal states.

Once we update, we continue stepping the vectorized environment from where we
left off, i.e. the environment is not reset at the beginning of the new
iteration. This is actually very helpful in the case of long horizon tasks where
episodes could be extremely long (think 100K steps).

The update function is not much different, but note that it now accepts
additional parameters:
```python
class A2CAgent(Agent):
    def update(self, states, actions, rewards, next_states, done):
        values = self.value(states)
        next_values = self.value(next_states)
        next_values = np.where(done, 0., next_values)
        returns = rewards + next_values
        adv = rewards + next_values - values

        # Update the policy network.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logits = self.policy_network(states)
        logp = F.cross_entropy(logits, actions, reduction="none")
        loss = (logp * adv).mean()

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

        # Update the value network.
        for o, r in DataLoader((states, returns), self.batch_size):
            pred = self.value_network()
            vf_loss = F.mse_loss(pred, r)
            self.value_optim.zero_grad()
            vf_loss.backward()
            self.value_optim.step()
```

Here we can see how we are actually using the information provided by `done`: in
case our agent steps a sub-environment from its current state into a terminal
state, then we should treat the obtained reward as the true return for that
state - no bootstrapping.


## MULTI-STEP A2C
The A2C algorithm provides additional variance reduction by bootstrapping the
estimate of the gradient, but it also adds bias to the estimate. While the
vanilla policy gradient provides an un-biased estimator for the gradient, the
A2C is not.

With the current setup there is one very obvious improvement that we could do to
reduce the bias of the gradient estimate. Note that we are rolling out the
policy for multiple steps, but we are computing the return using a single-step
bootstrap. What we could do instead is use an n-step bootstrap estimation: the
return for each state will be computed by summing all the rewards along the
current trajectory and only at the end we will bootstrap:

$$ R(s_t, a_t) = r_{t+1} + r_{t+2} + \cdots + r_{T} + V(s_{T}). $$

```python
class A2CAgent(Agent):
    def update(self, states, actions, rewards, next_states, done):
        values = self.value(states)
        next_values = self.value(next_states)
        next_values = np.where(done, 0., next_values)

        N, T = rewards.shape
        returns = np.zeros_like(rewards)
        returns[:, -1] = np.where(done[:, -1], rewards[:, -1], values[:, -1])
        for t in range(T-2, -1, -1):
            returns[:, t] = rewards[:, t] + returns[:, t+1] * ~done[:, t]
        adv = returns - values

        # Update the policy network.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logits = self.policy_network(states)
        logp = F.cross_entropy(logits, actions, reduction="none")
        loss = (logp * adv).mean()

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

        # Update the value network.
        for o, r in DataLoader((states, returns), self.batch_size):
            pred = self.value_network()
            vf_loss = F.mse_loss(pred, r)
            self.value_optim.zero_grad()
            vf_loss.backward()
            self.value_optim.step()
```

The computation of the returns might seem a bit confusing at first, because it
runs the loop in reverse, starting from the end. We only need to bootstrap at
the end for non-terminal states, and then we just keep adding the rewards from
the previous steps to the current estimate of the return. If at some point we
reach a terminal state, then the return is simply the obtained reward - no
bootstrapping.


## GAE
Note that estimating the return using an n-step bootstrap reduces the bias, but
increases the variance of the estimate. There is an obvious trade-off between
bias and variance when choosing how large $n$ should be, i.e. how many steps to
perform in the agent-environment loop.

An effective approach to reduce the variance and perfectly balance the trade-off
is the [Generalized Advantage
Estimation](https://arxiv.org/abs/1506.02438)(GAE)[^GAE]. The formula is very
similar to $TD(\lambda)$ where we have a weighted summation of different n-step
returns. The $TD(\lambda)$-return is given by:

$$
\begin{align}
    R^{\lambda}(s_t, a_t) & = (1-\lambda) \Big( r_{t+1} + V(s_{t+1}) \Big) + \\\\
    & + (1-\lambda)\lambda \Big( r_{t+1} + r_{t+2} + V(s_{t+2}) \Big) + \\\\
    & + \cdots + \\\\
    & + (1-\lambda)\lambda^{T-t-2} \Big( r_{t+1} + r_{t+2} + \cdots + r_{T-1} + V(s_{T-1}) \Big) + \\\\
    & + \lambda^{T-t-1} \Big( r_{t+1} + r_{t+2} + \cdots + r_{T} + V(s_{T}) \Big).
\end{align}
$$

Here $\lambda$ is a parameter that controls the exponential weighting of the
different n-step returns. For $\lambda=0$ we get the simple one-step return used
in online A2C, and for $\lambda=1$ we get the full n-step return used in the
multi-step A2C.

To get the generalized advantage estimator we simply subtract the value of $s_t$:
$$ A^{GAE}(s_t, a_t) = R^{\lambda}(s_t, a_t) - V(s_t). $$

Let us define $\delta_t = r_{t+1} + V(s_{t+1}) - V(s_t)$. Working with
advantages instead of returns will actually allow us to nicely simplify the
formula using telescoping sums. For example the two-step advantage can be
written as:
$$
\begin{aligned}
    A_t^{(2)} &= r_{t+1} + r_{t+2} + V(s_{t+2}) - V(s_t) \\\\
    &= r_{t+1} + V(s_{t+1}) - V(s_t) + r_{t+2} + V(s_{t+2}) - V(s_{t+1}) \\\\
    &= \delta_t + \delta_{t+1}.
\end{aligned}
$$

Finally, for the GAE we get:
$$ A^{GAE} = \sum_{i=0}^{T-t-1} \lambda^{i} \delta_{t+i}. $$

Again, I am glossing over the details of how this formula is derived, and why it
works, but if you want to read more I suggest checking ot [this blog
post](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/).

<!--
\begin{align*}
    A^{GAE} &= (1-\lambda) \Big( A_t^{(1)} + \lambda A_t^{(2)} + \lambda^2 A_t^{(3)} + \cdots + \lambda^{T-t-2}A_t^{(T-t-2)} \Big) + \lambda^{T-t-1} A_t^{(T-t-1)} \\
    &=(1-\lambda) \Big(\delta_t + \lambda(\delta_t + \delta_{t+1}) + \lambda^2 (\delta_t + \delta_{t+1} + \delta_{t+2}) + \cdots + \lambda^{T-t-2}(\delta_t + \cdots + \delta_{T-2}) \Big) + \lambda^{T-t-1} (\delta_t + \cdots + \delta_{T-1}) \\
    &=(1-\lambda) \Big( \delta_t(1+\lambda+\lambda^2+\cdots+\lambda^{T-t-2}) + \lambda \delta_{t+1}(1+\lambda+\lambda^2+\cdots+\lambda^{T-t-3}) + \cdots \Big)  + \lambda^{T-t-1} (\delta_t + \cdots + \delta_{T-1}) \\
    &= (1-\lambda^{T-t-1}) \delta_t + (\lambda - \lambda^{T-t-1})\delta_{t+1} + \cdots + (\lambda^{T-t-2} - \lambda^{T-t-1})\delta_{T-2} + \lambda^{T-t-1} (\delta_t + \cdots + \delta_{T-1}) \\
    &= \delta_t + \lambda \delta_{t+1} + \lambda^2 \delta_{t+2} + \cdots + \lambda^{T-t-2} \delta_{T-2} + \lambda^{T-t-1} \delta_{T-1} \\
    &= \sum_{i=0}^{T-t-1} \lambda^{i} \delta_{t+i}.
\end{align*}
-->

To use GAE in our A2C agent we simply have to modify the way we estimate the
advantages:
```python
class A2CAgent(Agent):
    def update(self, states, actions, rewards, next_states, done):
        values = self.value(states)
        next_values = self.value(next_states)
        next_values = np.where(done, 0., next_values)

        N, T = rewards.shape
        adv = np.zeros_like(rewards)
        adv[:, -1] = np.where(done[:, -1], rewards[:, -1] - values[:, -1], 0.)
        for t in range(T-2, -1, -1):
            delta = rewards[:, t] + values[:, t+1] * ~done[:, t] - values[:, t]
            adv[:, t] = delta + self.lamb * adv[:, t+1] * ~done[:, t]
        returns = adv + values

        # Update the policy network.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logits = self.policy_network(states)
        logp = F.cross_entropy(logits, actions, reduction="none")
        loss = (logp * adv).mean()

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

        # Update the value network.
        for o, r in DataLoader((states, returns), self.batch_size):
            pred = self.value_network()
            vf_loss = F.mse_loss(pred, r)
            self.value_optim.zero_grad()
            vf_loss.backward()
            self.value_optim.step()
```

Again, the computation of the advantages might seem a bit confusing, but it
follows a simple logic:
 * bootstrap at the end for non-terminal states,
 * at each step compute the current $\delta_t$ by making sure not to bootstrap
   on terminal states,
 * compute the advantage by adding the $\delta_t$ to the running sum of deltas
   decayed by $\lambda$. Again make sure not to add anything if at a terminal
   state.

Since we are directly computing the advantages, the returns are actually derived
by reversing the equation: $R^{\lambda}(s_t, a_t) = A^{GAE}(s_t, a_t) + V(s_t)$.

## PPO
Note that all of the policy updates that we did until now are simple "vanilla"
updates: we simply update the policy once, using the estimate of the gradient,
and then we discard the collected data. However, there is nothing stopping us
from using a more sophisticated update algorithm. In fact, for the experiments
in the GAE paper the authors use the
[TRPO](https://arxiv.org/abs/1502.05477)[^TRPO] update rule.

(Not really shocking news, given that both TRPO and GAE are authored by the same
people.)

Here we will take a look at [PPO](https://arxiv.org/abs/1707.06347)[^PPO]. The
problem that the authors are trying to solve is to come up with an algorithm
that allows us to take the biggest possible update step on the policy parameters
before throwing out the collected rollout data. The approach taken in this paper
is to allow for multiple update steps that, when combined, would approximate
this maximum possible update.

Note that after we update the policy parameters once, $\pi_\theta =
\pi_{\theta_{old}} + \nabla \theta_{old}$, every other update would actually be
using off-policy data. To correct for this data miss-match we can use importance
sampling:

$$
\begin{align}
\displaystyle J(\theta)
& = E_{s_t \sim \mu_\theta, \space a_t \sim \pi_\theta}
    \bigg[ \sum_{t=1} r_{t+1}\bigg] \\\\
& = E_{s_t \sim \mu_\theta, \space a_t \sim \pi_\theta}
    \bigg[
        \frac{\pi_{\theta_{old}}(a|s)}{\pi_{\theta_{old}}(a|s)} \sum_{t=1} r_{t+1}
    \bigg] \\\\
& = E_{s_t \sim \mu_\theta, \space a_t \sim \pi_{\theta_{old}}}
    \bigg[
        \frac{\pi_{\theta_{old}}(a|s)}{\pi_\theta(a|s)} \sum_{t=1} r_{t+1}
    \bigg]
\end{align}
$$

It seems like we could compute the objective to update the new policy weights
using the data collected with the old policy weights, as long as we correct with
the importance sampling weight:

$$ \displaystyle \rho(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}. $$

However, note that, in order to compute the correct gradient estimate, the
actions have to be sampled under $\pi_{\theta_{old}}$, but the states have to be
sampled under $\mu_\theta$. Unfortunately our data was sampled under
$\mu_{\theta_{old}}$.

How bad is that?

It turns out that if $\pi_\theta$ does not deviate *too much* from
$\pi_{\theta_{old}}$, then using the old data sampled from $\mu_{\theta_{old}}$
is actually ok. The difference between the objectives calculated using
$\mu_\theta$ and $\mu_{\theta_{old}}$ is bounded, and thus, optimizing one would
also optimize the other. In simple words, it is ok to optimize the objective
with the old data as long as $\pi_\theta$ is close to $\pi_{\theta_{old}}$. A
proof of this claim can be found in Appendix A of the [TRPO
paper](https://arxiv.org/abs/1502.05477)[^TRPO].

There are two different proximal policy algorithms each using a different
heuristic to try to ensure that $\pi_\theta$ is close to $\pi_{\theta_{old}}$:

* **PPO-Penalty** - constraints the *KL divergence* between the two
distributions by adding it as a penalty to the objective:
$$
    J(\theta) =
    E_{s_t \sim \mu_\theta, \space a_t \sim \pi_{\theta_{old}}}
    \bigg[
        \rho(\theta) A(s,a) - \beta KL(\pi_\theta(\cdot|s), \pi_{\theta_{old}}(\cdot|s))
    \bigg]
$$

* **PPO-CLIP** - clips the objective function if $\pi_\theta$ deviates too much
from $\pi_{\theta_{old}}$:
$$
    J(\theta) =
    E_{s_t \sim \mu_\theta, \space a_t \sim \pi_{\theta_{old}}}
    \bigg[
        \min \big(
            \rho(\theta) A(s,a), \space \text{clip}(\rho(\theta), 1-\epsilon, 1+\epsilon) A(s,a)
        \big)
    \bigg]
$$


The algorithm implemented here is **PPO-CLIP** augmented with a check for early
stopping. We will split the collected rollout data into mini-batches and we will
iterate for several epochs over the batches. At the end of every epoch we will
check the *KL divergence* between the original policy $\pi_{\theta_{old}}$ and
the newest policy $\pi_\theta$. If a given threshold is reached, then we will
stop updating and collect new rollout data.

```python
class PPOAgent(Agent):
    def update(self, states, actions, rewards, next_states, done):
        values = self.value(states)
        next_values = self.value(next_states)
        next_values = np.where(done, 0., next_values)

        N, T = rewards.shape
        adv = np.zeros_like(rewards)
        adv[:, -1] = np.where(done[:, -1], rewards[:, -1] - values[:, -1], 0.)
        for t in range(T-2, -1, -1):
            delta = rewards[:, t] + values[:, t+1] * ~done[:, t] - values[:, t]
            adv[:, t] = delta + self.lamb * adv[:, t+1] * ~done[:, t]
        returns = adv + values

        # Update.
        logp_old = self.policy(states).log_prob(actions)
        values_old = self.value(states)
        for _ in range(self.n_epochs):
            dataset = DataLoader(
                (states, actions, returns, adv, logp_old, values_old),
                self.batch_size,
            )
            for s, a, r, ad, lp_old, v_old in dataset:
                # Update the policy network.
                logits = self.policy_network(s)
                logp = -F.cross_entropy(logits, a, reduction="none")
                rho = (logp - lp_old).exp()
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                clip_adv = np.clip(rho, 1-self.pi_clip, 1+self.pi_clip) * ad
                pi_loss = (min(rho * ad, clip_adv)).mean()
                self.policy_optim.zero_grad()
                pi_loss.backward()
                self.policy_optim.step()

                # Update the value network.
                v_pred = self.value_network(s)
                v_clip = v_old + np.clip(v_pred-v_old, -self.vf_clip, self.vf_clip)
                vf_loss = (max((v_pred - r)**2, (v_clip - r)**2)).mean()
                self.value_optim.zero_grad()
                vf_loss.backward()
                self.value_optim.step()

            # Check for early stopping.
            logp = self.policy(states).log_prob(actions)
            KL = logp_old - logp
            if KL.mean() > 1.5 * self.tgt_KL:
                break
```

Note that the advantages are normalized at the mini-batch level, in the
beginning of every iteration before computing the loss. This should come as no
surprise, because as we mentioned earlier, the goal of this normalization is to
stabilize the gradient updates of the neural network. It has nothing to do with
the baseline for the return.

In addition to clipping the objective for the policy, we are also clipping the
value function loss before updating the parameters of the value network. This
approach was first introduced in the GAE paper[^GAE] (see Section 5), and was
later also applied in the [official PPO
implementation](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75).

Finally, to compute the *KL divergence* between the new and the old policy we
use the following equality:
$$
D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P} \Big[ \log P(x) - \log Q(x) \Big].
$$
Checkout [this blog post](http://joschu.net/blog/kl-approx.html) by John
Schulman where he proposes a different unbiased estimator for the KL divergence
that has less variance.

<!--
#TODO:
add some text about entropy reg
## ENTROPY REGULARIZATION
 -->


[^REINFORCE]: [1992](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
"Simple statistical gradient-following algorithms for connectionist reinforcement
learning" by Ronald J. Williams
[^Baseline]: [2013](https://arxiv.org/abs/1301.2315) "The optimal reward baseline
for gradient-based reinforcement learning" by Lex Weaver, Nigel Tao
[^A2C]: [2016](https://arxiv.org/abs/1602.01783)) "Asynchronous methods for deep
reinforcement learning" by Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza,
Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu
[^GAE]: [2015](https://arxiv.org/abs/1506.02438) "High-dimensional continuous
control using generalized advantage estimation" by John Schulman, Philipp Moritz,
Sergey Levine, Michael Jordan, Pieter Abbeel
[^TRPO]: [2015](https://arxiv.org/abs/1502.05477) "Trust Region Policy Optimization"
by John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel
[^PPO]: [2017](https://arxiv.org/abs/1707.06347) "Proximal Policy Optimization
Algorithms" by John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford,
Oleg Klimov
