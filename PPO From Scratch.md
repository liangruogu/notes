## PPO From Scratch

##### Policy Optimization

Consider a stochastic policy $\pi_{\theta}$. 

We aims to maximize  $J(\pi_{\theta}) = \mathop{\mathbb{E}}\limits_{\tau \sim \pi_\theta}[R(\tau)]$

optimize this policy by gradient ascent, i.e.

$$
\theta_{k+1} = \theta_{k} + \alpha\nabla_\theta{J(\pi_{\theta})|_{\theta_k}}
$$

$\nabla_\theta{J(\pi_{\theta})}$ is called policy gradient;

The algorithm is called **policy gradient algorithms**

To actually use this algorithm, we need to express $\tau$ by $(s_0, a_0, ...)$, for a trajectory: 

$$
P(\tau|\theta) = P_0(s_o)\prod_{t=0}^{\infty}{P(s_{t+1}|s_t, a_t)\pi(a_t|s_t)}
$$

sample $\tau$ according to $\pi_{\theta}$ (for estimate)

$$
\nabla_\theta{J(\pi_{\theta})} = \nabla_\theta{\mathop{\mathbb{E}}\limits_{\tau \sim \pi_{\theta}}}[R(\tau)]=\nabla_\theta\int_\tau P(\tau|\theta)R(\tau)
$$

$$
=\int_\tau P(\tau|\theta)\nabla_\theta \log P(\tau|\theta)R(\tau)
$$

$$
=\mathop{\mathbb{E}}\limits_{\tau\sim\pi_\theta}[\sum\limits_{t=0}^T\nabla_\theta \log\pi_\theta(a_t|s_t)R(\tau)]
$$

Finally, we can get:

$$
\hat{g} = \frac{1}{|\mathcal{D}|}\sum\limits_{\tau\sim\mathcal{D}}\sum\limits_{t=0}^{T} \nabla_\theta \log\pi_{\theta}(a_t|s_t)R(\tau)
$$

where $|\mathcal{D}|$ is the number of trajectories in $\mathcal{D}$

> This is the simplest version of the computable expression.

However, $R(\tau)$ seem to be global in the trajectory, we are supposed to remove the reward before $t$

$$
\hat{R_{t}} = \sum\limits_{t'=t}^{T}R(s_{t'}, a_{t'}, s_{t'+1})
$$

Generally:

$$
\nabla_\theta{J(\pi_{\theta})} =\mathop{\mathbb{E}}\limits_{\tau\sim\pi_\theta}[\sum\limits_{t=0}^T\nabla_\theta \log\pi_\theta(a_t|s_t)A_t]
$$

where $A_t$ is the advantage

### GAE

One step TD Error:

$$
\delta_t = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_{t})
$$

We can train a value netword by minimize $\delta_t^2$

---

From another perspective, $\delta_t$ means the action $a_t$'s advantage, because if $a_t$ is better, $r_t + \gamma V_\theta(s_t)$ is larger, and $\delta_t$ is greater, so we can use $\delta_t$ to estimate   $A_t$. However, that's not exactly, for the bais and variance of $\delta_t$

---

$n$ step TD Error:

$$
\delta_t^{(n)} = \sum\limits_{l=0}^{n-1}\gamma^l r_{t+l} + \gamma^n V_\theta(s_{t+n}) - V_\theta(s_t)
$$

**A larger $n$ leads to lower bais but higher variance**

whereas a smaller n leads to higher bais but lower var

---

Which n should we choose or how can we define $A_t$

> Intro hypr-parameter $\lambda$ to balance bais and var

$$
\hat{A_t} = (1-\lambda)\sum\limits_{n=0}^{\infty}\lambda^n\delta_t^{(n+1)}
$$

Each $\delta_t^{(n+1)}$ is weighted by $\lambda^n$, while the prefactor $1-\lambda$ serves to normalize the resulting geometric mixture.

$$
\sum\limits_{n=0}^{\infty}\lambda^n = \frac{1}{1-\lambda}

$$

It's hard to compute $\delta_t^{(n+1)}$, and we can use:

$$
\delta_t^{(n+1)} = \sum\limits_{l=0}^n \gamma^l\delta_{t+l}
$$

$$
\hat{A_t} = (1-\lambda)\sum\limits_{n=0}^{\infty}\sum\limits_{l=0}^{n}\lambda^n\gamma^l\delta_{t+l}
$$

---

The next step is the most elegant part of derivation.

Since we can express $\hat{A_t}$ using a series of $\delta_n$ terms.

$$
\hat{A_t} = \sum\limits_{l=0}^{\infty}a_l\delta_{t+l}
$$

**Only the lower-left portion survives.**

$$
A_{t}^{GAE} = \sum\limits_{l=0}^{\infty}(\lambda\gamma)^{l}\delta_{t+l}
$$

### PPO

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta{old}}(a_t|s_t)}
$$

$$
L^{CLIP}_t(\theta) = \min(r_t(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)
$$

$$
L^{VF}_t(\theta) = \delta_t^2
$$

$$
L_t^{ENT}(\theta) = H(\pi_\theta)
$$

$$
Loss = - L_t^{CLIP}(\theta) + c_1L_t^{VF}(\theta) - c_2L_t^{ENT}(\theta)
$$

> The clip term limits how far the new policy is allowed to deviate from the old one, while the min operator ensures that updates are penalized whenever At​ and rθ​ move in opposite directions.

# 
