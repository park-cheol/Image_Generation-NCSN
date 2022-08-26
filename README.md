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
- $\lbrace x_1, x_2, ..., x_N \rbrace$: $p(x)$로부터 독립적으로 추출한 dataset
- $f_\theta(x)$: real-valued function parameterized by a learnable parameter $\theta$
- $Z_\theta (> 0)$: normalizing constant dependent on $\theta$ ( $\int p_\theta(x) dx = 1$ )

$$p.d.f = p_\theta (x) = \frac{\mathrm{e}^{-f_\theta (x)}}{Z_\theta}$$

- Train: Maximize the log-likelihood of the data

$$\max_{\theta} \sum_{i=1}^{N} \log p_\theta (x_i)$$

- 하지만, $Z_\theta = a intractable quantity for any general f_\theta (x)$
  - 이를 tractable 해주기 위해서 architecture에 제약
  - 또는, normalizing constant를 추정하기에는 계산량이 너무 큼
  

> - *Sol) Score 사용하면 Z term 삭제 가능*
> - $S_\theta (x)$: Score-based model
> 
> $$Score = \nabla_x \log p(x)$$
> 
> $$S_\theta (x) \thickapprox \nabla_x \log p(x) = - \nabla_x f_\theta (x) - \nabla_x \log Z_\theta = -\nabla_x f_\theta (x)$$
> 
> **Fisher divergence**
>
> $$\mathbb{E}_p(x) \[ \parallel \nabla_x \log p(x) - S_\theta (x) \parallel_{2}^{2} \]$$

### [Score Matching](!https://scholar.google.co.kr/scholar?hl=ko&as_sdt=0%2C5&q=Estimation+of+non-normalized+statistical+models+by+score+matching&btnG=) and [Denoising Score Matching](!http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)



# Reference
