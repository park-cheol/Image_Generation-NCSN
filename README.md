# Generative Modeling by Estimating Gradients of the Data Distribution

- **Score-based generative modeling**에 관련된 내용 정리

### Limitations of previous methods

- **Likelihood-Based Models** (VAE, Flow model, EBM, etc)
  - Strong restrictions on the model architecture to ensure a tractable normalizing constant for likelihood computation.
  - Surrogate objectives to approximate maximum likelihood training. (ELBO)

- **Implicit Generative Models** (GAN)
  - Notoriously unstable training.
  - Mode collapse.

> **Score function(the gradient of the log probability density function)** 를 사용하여 위의 문제들을 해결

### Overview of score-based generative modeling

- *Train*: **Score Matching(2005) -> Denoising Score Matching(2011)**
- *Test(Sampling)*: **Langevin dynamics -> Annealed Langevin dynamics**

![img1](https://user-images.githubusercontent.com/76771847/186809602-1f1c1924-5261-4475-8c55-f521019f936c.jpg)

### Score function

- $p(x)$: data distribution
- ${x_1, x_2, ..., x_N}$: $p(x)$로부터 독립적으로 추출한 dataset
- $f_\theta(x)$: real-valued function parameterized by a learnable parameter $\theta$
- $Z_\theta (> 0)$: normalizing constant dependent on $\theta$ ($\int p(x)_\theta dx = 1$) 

$$p.d.f = p_\theta (x) = \frac{\mathrm{e}^{-f_\theta (x)}}{Z_\theta}$$




# Reference
