## O2O

**O2O: ODE-based learning to optimize (O2O)** is a comprehensive framework integrating the dynamical inertial Newton with asymptotic vanishing damping (DIN-AVD) equation for developing optimization methods. We focus on the uncontrained convex smooth optimization:
$$\min_{x}\quad f(x),\qquad x\in\mathbb{R}^n.$$

The pipeline of O2O is as follows:

![pipeline](pipeline.png)

The key ingredients of the proposed pipeline are:

* Stable discretization guarantee: Ensures the stable discretization of the ODE using the forward Euler scheme, consisting of a convergent condition and a stable condition.

* Learning to optimize framework: Utilizes a learning to optimize framework and a corresponding algorithm to find the optimal coefficients numerically.

* Stopping time: A measure of the efficiency of the algorithm generated by discretizing the ODE, generalizing complexity from discrete-time to continuous-time cases.

* Probability distribution of a parameterized function family: Defines the probability distribution by establishing equivalence with corresponding parameters.

* Penalty function method and stochastic optimization algorithms: Solves the stochastic optimization problem using these techniques.

* Conservative gradients: Derives conservative gradients of the stopping time and constraint functions to make the algorithm more robust and general.

* Convergence guarantees: Provides convergence guarantees for the training algorithm under the sufficient decrease assumption, using only the conservative gradients.

## Code Structure
```
	├── classic_optimizer # classic first-order methods for comparison
	│   
	├── dataset # datasets that are used to generate training and testing functions
	│   
	├── problem # contain the logistic regression and lpp norm minimization
	│   
	├── run.sh # scripts to run different experiments
	│   
	├── train.py # main script to train the model
	│   
	├── vector_field # vector fields for different ODEs
	│   
	└── visualization
		├── calc_complexity.py # calculate the averaged complexity of different algorithms for certain optimization problems
		│   
		├── calc_lastgrad.py # calculate the averaged gradient norm at last iteration of different algorithms for certain optimization problems
		│   
		├── compare_diff_epochs.py # compare the performance of the learned algorithm at different epochs
		│   
		├── generate_test_result.py # use the trained coefficents to generate test results
		│   
		├── table_complexity.py # organize the result generated by calc_complexity.py to a table (Table 4,5 in the paper)
		│   
		├── table_lastgrad.py # organize the result generated by calc_complexity.py to a table (Table 2,3 in the paper)
		│   
		├── visualize_diff_epochs.py # visualize the performance of the learned algorithm at different epochs (Fig. 4 in the paper)
		│   
		└── visualize_test_result.py # visualize the test results (Fig. 5,6,7 in the paper)
```

## Requirements
The environment dependencies are exported in the form of "requirements.yaml". For the most convenient installation of these environments, we highly recommend using conda.
```
conda env create -f requirements.yaml
```

## Quick Start

Run the following commands to train the model in different optimization tasks.
```
python train.py --problem lpp --dataset a5a --num_epoch 80 --pen_coeff 0.5
python train.py --problem lpp --dataset separable --num_epoch 15 --pen_coeff 0.5 --eps 1e-4
python train.py --problem lpp --dataset covtype --num_epoch 50 --pen_coeff 0.5 --batch_size 10240
python train.py --problem logistic --dataset covtype --num_epoch 50 --pen_coeff 0.5 --batch_size 10240
python train.py --problem lpp --dataset w3a --num_epoch 100 --pen_coeff 0.5
```

When the training process is finished in specific task, run the files in visualization to generate the test results and visualize them. The trained model and checkpoints are saved in the folder "saved_models" and "checkpoints" under "train_log". You need to specify the path of the model and checkpoints when run the visualization scripts.

## Usage
```
usage: python train.py [-h] [--problem {logistic,lpp}] [--dataset {mushrooms,a5a,w3a,phishing,separable,covtype}] [--pretrain]
                       [--num_epoch NUM_EPOCH] [--pen_coeff PEN_COEFF] [--lr LR] [--momentum MOMENTUM] [--batch_size BATCH_SIZE]
                       [--seed SEED] [--init_it INIT_IT] [--discrete_stepsize DISCRETE_STEPSIZE] [--eps EPS] [--l2 L2] [--p P]
                       [--optim {SGD,Adam}] [--threshold THRESHOLD]

Train the neural ODE using exact L1 penalty method.

optional arguments:
  -h, --help            show this help message and exit
  --problem {logistic,lpp}
                        Either logistic regression (default: without L2 regularization) or Lpp minimization (default: p=4)
  --dataset {mushrooms,a5a,w3a,phishing,separable,covtype}
                        Dataset use for training
  --pretrain            Load the pre-trained model or not
  --num_epoch NUM_EPOCH
                        The number of the training epoch
  --pen_coeff PEN_COEFF
                        The penalty coefficient of the L1 exact penalty term
  --lr LR               Learning rate of SGD
  --momentum MOMENTUM   Momentum coefficient of SGD
  --batch_size BATCH_SIZE
                        Batch size for training, default 1024, 10240 is recommended for covtype
  --seed SEED           Random seed for reproducing. 3407 is all you need
  --init_it INIT_IT     The number of iterate used to initialize the neural ODE, default is 300
  --discrete_stepsize DISCRETE_STEPSIZE
                        the step size used in discretization, default is 0.04
  --eps EPS             epsilon used to define the stopping time
  --l2 L2               the coefficient of the L2 regularization term in logistic regression
  --p P                 the exponential index of lpp minimization
  --optim {SGD,Adam}    the optimizer using in training
  --threshold THRESHOLD
                        the threshold using in constraints
```

## Summary of Dataset
The datasets used in our experiments are summarized in Table 1. In this table, $n$, $N_{\text{train}}$, and $N_{\text{test}}$ represent the dimension of the variable, the number of instances in the training dataset, and the number of instances in the test dataset, respectively.

<div align="center">

| Dataset | $n$ | $N_{\text{train}}$ | $N_{\text{test}}$ | Separable | References |
|---------|-----|-------------------|------------------|-----------|------------|
| `a5a`   | $123$ | $6,414$         | $26,147$         | No        | [Dua and Graff, 2019] |
| `w3a`   | $300$ | $4,912$         | $44,837$         | No        | [Platt, 1998] |
| `mushrooms` | $112$ | $3,200$     | $4,924$          | Yes       | [Dua and Graff, 2019] |
| `covtype` | $54$ | $102,400$      | $478,612$        | No        | [Dua and Graff, 2019] |
| `phishing` | $68$ | $8,192$       | $2,863$          | No        | [Dua and Graff, 2019] |
| `separable` | $101$ | $20,480$   | $20,480$         | Yes       | [Wilson et al., 2019] |

*Table 1: A summary of the datasets used in experiments.*

</div>

All the datasets are designed for binary classification problems, and downloaded from the [LIBSVM data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), except the `separable` dataset. We construct the `separable` dataset using the code snippet downloaded from \[Wilson et al., 2019\]. They are generated by sampling $10240$ instances from $\mathcal{N}(\mu,I_{d})$ with label $b_{f}=1$ and $\mathcal{N}(\mu+\nu,I_{d})$ with label $b_{f}=0$, respectively. Here, $I_{d}\in\mathbb{R}^{d\times d}$ denotes the identity matrix. Each element of the vector $\mu\in\mathbb{R}^{d}$ is sampled from $\{0,1,\ldots,19\}$ uniformly, while the elements of the margin vector $\nu$ are drawn from $\{0,0.1,\ldots,0.9\}$ uniformly.

For each dataset, the label of each sample belongs to $\{0,1\}$. The value of each attribute are normalized to $[-1,1]$ by dividing the data-matrix $(a_1,a_2,\ldots,a_N)$ with the max absolute value of each attribute. The training and testing sets are pre-specified for `a5a` and `w3a`. For datasets that do not specify the testing set and training set, we divide them manually.

## Compared Methods
We set $L=\min\\{\|A^\top A\|/N,4\Vert \nabla^2 f_{D}(x_0)\Vert\\}$, $x_0=x_1=\mathbf{1}/n-\nabla f_{D}(\mathbf{1}/n)/L$, and $v_{0}=x_{0}+\beta(t_{0})\nabla f(x_{0})$. The compared methods in our experiments are listed below.

- **GD**. The vanilla gradient descent GD is the standard method in optimization. We set the stepsize as $h=1/L$.

- **NAG**. Nesterov's accelerated gradient descent method NAG is a milestone of the acceleration methods. We employ the version for convex functions

  $$y_{k+1}=x_{k}-h\nabla f(x_{k}),\quad x_{k+1}=y_{k+1}+\frac{k-1}{k+2}(y_{k+1}-y_{k}),$$

  where the stepsize is chosen as $h=1/L$.

- **EIGC**. This method is obtained by applying the explicit-implicit gradient correction scheme for

  $$\ddot{x}(t)+\frac{a}{t}\dot{x}(t)+\sqrt{s}\nabla^2 f(x(t))\dot{x}(t)+\left(1+\frac{a\sqrt{s}}{t}\right)\nabla f(x(t))=0.$$

  Let $s=1/L$. In each iteration, the method performs

  $$x_{k}-x_{k-1}=\sqrt{s}v_{k-1},\quad v_{k}-v_{k-1}=-\frac{a}{k}v_{k}-\sqrt{s}(\nabla f(x_{k})-\nabla f(x_{k-1}))-\left(1+\frac{a}{k}\right)\sqrt{s}\nabla f(x_{k}).$$

- **INVD**. Inertial Newton algorithm with vanishing damping **(INVD)** with the coefficients learned by **O2O**.
  $$\frac{x_{k+1}-x_{k}}{h}=v_{k}-\beta(t_{k})\nabla f(x_{k}),\quad \frac{v_{k+1}-v_{k}}{h}=-\frac{\alpha}{t}\left(v_{k}-\beta(t_{k})\nabla f(x_{k})\right)+(\dot{\beta}(t_{k})-\gamma(t_{k}))\nabla f(x_{k}).$$

## Experimental Results
We empirically compare these methods on two tasks: logistic regression and minimization of $\ell_{p}^p$ in six distinct datasets.  In our first task, we consider the logistic regression problem defined by a finite set $D$, a subset of a given dataset or is sampled from a distribution
$$\min_{x\in\mathbb{R}^n}f_{D}(x)=\frac{1}{|D|}\sum_{(a_{i},b_i)\in D}\log (1+\exp(-b_i\langle a_i,x \rangle)),$$
where the data pairs $\{a_{i},b_{i}\}\in \mathbb{R}^n\times \{0,1\},i\in [|D|]$. Let $\sigma(t) = \frac{1}{1+\exp(-t)}\in(0,1)$, the Hessian matrix of $f_{D}$ is
$$\nabla^2 f_{D}(x) = \frac{1}{|D|}\sum_{(a_{i},b_i)\in D}b_i^2a_ia_i^\top\sigma(b_i\langle a_i,x \rangle)(1-\sigma(b_i\langle a_i,x \rangle)).$$
Let $A=(a_1,\ldots,a_{|D|})$, the Lipschitz constant of $\nabla f$ is bounded by $L = \|AA^\top\|/|D|$.

In our second task, given an even integer $p\geq 4$ and a set of the samples $D$, we consider the $\ell_p^p$ minimization as follows
$$\min_{x\in\mathbb{R}^n}f_{D}(x)=\frac{1}{|D|}\sum_{(a_{i},b_i)\in D}\frac{1}{p}(\langle a_i,x \rangle-b_i)^p.$$
The Hessian matrix of $f_{D}$ writes
$$\nabla^2 f_{D}(x)=\frac{1}{|D|}\sum_{(a_{i},b_i)\in D}(p-1)(\langle a_i,x\rangle-b_i)^{p-2}a_i a_i^\top.$$
Since $(\langle a_i,x\rangle-b_i)^{p-2}$ is unbounded of each $i$, the Lipschitz constant for $\nabla f_{D}$ cannot be bounded globally.

We randomly generate 100 test functions for each problem, varying the instances from the dataset. The problems are specified by the dataset, batch size, and formulation (e.g., `lpp_a5a` for `a5a` dataset and $\ell_{p}^{p}$ minimization).

### Averaged performance measure at the $N$-th iteration:
The averaged performance measure is a metric that evaluates the effectiveness of a method in minimizing the objective functions in the test set $F_{\text{test}}$. It is calculated by taking the average of the logarithm of the gradient norm $\Vert\nabla f(x_N)\Vert$ for each function $f$ in the test set at the $N$-th iteration (in this case, $N=500$). The gradient norm measures how close the current solution $x_N$ is to a stationary point (a point where the gradient is zero). A smaller value of $m(F_{\text{test}})$ indicates that the method has effectively minimized the objective functions and reached a point closer to the optimal solution.

$$m(F_{\text{test}})=\frac{1}{|F_{\text{test}}|}\sum_{f\in F_{\text{test}}}\log\Vert\nabla f(x_{N})\Vert$$

The averaged performance measure is reported in Tables 2 and 3. INVD outperforms other methods with at least a magnitude in most cases.

<div align="center">

| Method | mushrooms | a5a | w3a | phishing | covtype | separable |
|--------|-----------|-----|-----|----------|---------|-----------|
| GD     | -1.55     | -1.81 | -1.90 | -1.35 | -1.89 | -1.56 |
| NAG    | -3.37     | -3.11 | -3.26 | -3.01 | -3.07 | -3.66 |
| INVD(initial)   | -3.02     | -2.97 | -3.02 | -2.80 | -3.48 | -3.32 |
| EIGC   | -3.02     | -2.97 | -3.02 | -2.80 | -3.48 | -3.31 |
| INVD(learned)   | -4.83     | -4.38 | -4.46 | -4.82 | -4.37 | -5.49 |

*Table 2: Averaged performance measure in logistic regression problems.*

</div>

<div align="center">

| Method | mushrooms | a5a | w3a | phishing | covtype | separable |
|--------|-----------|-----|-----|----------|---------|-----------|
| GD     | -2.49     | -2.79 | -3.18 | -2.36 | -2.65 | -2.95 |
| NAG    | -4.35     | -4.19 | -4.72 | -4.06 | -4.43 | -6.11 |
| INVD(initial)   | -4.16     | -3.99 | -4.66 | -4.37 | -4.47 | -6.15 |
| EIGC   | -4.16     | -4.05 | -4.66 | -4.37 | -4.51 | -6.14 |
| INVD(learned)   | -5.27     | -5.11 | -5.71 | -5.65 | -5.14 | -7.55 |

*Table 3: Averaged performance measure in* $\ell_{p}^{p}$ *minimization problems.*

</div>

### Averaged complexity when achieving the $\varepsilon$-suboptimality:
The averaged complexity is a metric that measures the computational efficiency of a method in reaching a desired level of accuracy. It is calculated by taking the average of the complexity measure $N(f, \varepsilon)$ for each function $f$ in the test set $F_{\mathrm{test}}$. The complexity measure $N(f, \varepsilon)$ is defined as the number of iterations required by the method to reach a gradient norm below the threshold $\varepsilon$ (in this case, $\varepsilon = 3 \times 10^{-4}$). If the method does not reach the threshold within the maximum number of iterations (in this case, 500), its complexity is denoted as 500. A smaller value of $N(F_{\mathrm{test}})$ indicates that the method is computationally efficient and requires fewer iterations to reach the desired level of accuracy.

$$N(F_{\text{test}})=\frac{1}{|F_{\text{test}}|}\sum_{f\in F_{\text{test}}}N(f,\varepsilon)$$
where the complexity is defined as
$$N_F:=\inf\\{N : m(f,M(f,x_0,N))\leq\epsilon,\text{ for all }f\in F\\}.$$
$M(f,x_0,N)$ denotes the $N$-th iteration produced by algorithm $M$ in function $f$ with initial point $x_0$.

The averaged complexity is presented in Tables 4 and 5. INVD consistently improves complexity, requiring only half the iterations of other methods in most problems.

<div align="center">

| Method | mushrooms | a5a | w3a | phishing | covtype | separable |
|--------|-----------|-----|-----|----------|---------|-----------|
| GD     | 500.00    | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 |
| NAG    | 500.00    | 500.00 | 500.00 | 500.00 | 500.00 | 424.71 |
| INVD(initial)   | 500.00    | 500.00 | 500.00 | 500.00 | 497.12 | 500.00 |
| EIGC   | 500.00    | 500.00 | 500.00 | 500.00 | 497.36 | 500.00 |
| INVD(learned)   | 153.48    | 227.32 | 216.42 | 182.15 | 237.88 | 11.49 |

*Table 4: Averaged complexity in logistic regression problems.*

</div>

<div align="center">

| Method | mushrooms | a5a | w3a | phishing | covtype | separable |
|--------|-----------|-----|-----|----------|---------|-----------|
| GD     | 500.00    | 500.00 | 500.00 | 500.00 | 500.00 | 500.00 |
| NAG    | 183.77    | 211.87 | 92.61 | 252.43 | 167.31 | 52.15 |
| INVD(initial)   | 235.07    | 245.68 | 96.17 | 224.36 | 203.02 | 22.10 |
| EIGC   | 235.84    | 239.49 | 96.03 | 224.53 | 204.12 | 29.98 |
| INVD(learned)   | 93.12     | 122.16 | 50.93 | 85.57 | 109.15 | 11.00 |

*Table 5: Averaged complexity in* $\ell_{p}^{p}$ *minimization problems.*

</div>

## Contact

We hope that the package is useful for your application. If you have any bug reports or comments, please feel free to email one of the toolbox authors:

- Zhonglin Xie, zlxie@pku.edu.cn
- Zaiwen Wen, wenzw@pku.edu.cn

## Reference
[Zhonglin Xie, Wotao Yin, Zaiwen Wen, O2O: ODE-based Learning to Optimize, arXiv:2307.00783, 2023.](https://arxiv.org/abs/2307.00783)

## License
GNU General Public License v3.0
