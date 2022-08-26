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

- 하지만, $Z_\theta$ = a intractable quantity for any general $f_\theta (x)$
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

![2](https://user-images.githubusercontent.com/76771847/186828121-250d06ac-834f-4bc3-9bb3-ff8a8444d6da.png)

### Score Matching and Denoising Score Matching

- 하지만, $\nabla_x \log p(x)$ 는 unknown data score -> **[Score Matching](https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)**

> - $p(x)$ term 삭제
> 
> $$\mathbb{E}_p(x) \[ tr(\nabla_x S_\theta (x)) + \frac{1}{2} \parallel S_\theta (x) \parallel_{2}^{2} \]$$
 
- 하지만, $tr(\nabla_x S_\theta (x))$ 이 너무 계산량이 많음 -> **[Denoising Score Matching](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)**

> - $tr(\nabla_x S_\theta (x))$ term 삭제
> - $q_\sigma (\tilde{x} | (x))$ : 사전에 정의된 노이즈
> - $q_\sigma (\tilde{x}) \triangleq \int q_\sigma (\tilde{x} | (x)) p(x) dx$ : the perturbed data distribution
> 
> $$\frac{1}{2} \mathbb{E}_{q_\sigma (\tilde{x} | (x)) p(x)} \[ \parallel S_\theta (\tilde{x}) - \nabla_x \log q_\sigma (\tilde{x} | x) \parallel_{2}^{2} \]$$
> 
> - 노이즈가 충분히 작으면,
> 
> $$S_{\theta^*} (x) = \nabla_x \log q_\sigma (x) \thickapprox \nabla_x \log p(x)$$
> $$q_\sigma (x) \thickapprox p(x)$$

### Langevin dynamics
 
- 학습이 완료된 score-based model( $S_\theta (x) \thickapprox \nabla_x \log p(x)$ )을 sampling

> - $z_i \sim N(0, I)$
> - $\varepsilon \rightarrow 0 \And K \rightarrow \infty$ 할 때, $x_K$ 는 $p(x)$ 에 수렴 
> 
> $$x_{i+1} \leftarrow x_i + \varepsilon \nabla_x \log p(x) + \sqrt{2 \varepsilon}z_i, i=0, 1, ..., K$$
![3](https://user-images.githubusercontent.com/76771847/186839953-e4c977f4-075c-4c9b-aa34-be444fafda7f.gif)

### Pitfalls

- Low density regions에서는 거의 없는 data point때문에 부정확한 score function을 예측
![5](https://user-images.githubusercontent.com/76771847/186840279-db12f8d9-acbe-424b-b793-684474247c4a.jpg)


- Data에 가우시안 노이즈를 추가하여 low density regions를 채워줌
![6](https://user-images.githubusercontent.com/76771847/186840597-cf83ca70-b7a2-4cd3-aa10-36a15d981361.jpg)
  - Noise 너무 크면, 원래의 분포를 over-corrupts
  - Noise 너무 작으면, Low Density Regions을 충분히 채워주지 못함

> - **Multiple Scales of Noise Pertubations**
> - $\sigma_1 < \sigma_2 < ... < \sigma_L$
> - $N(0, \sigma_{i}^{2}I), i= 1, 2, ..., L$
> 
> $$p_{\sigma_i} (x) = \int p(y)N(x; y, \sigma_{i}^{2}I) dy$$

# Reference
