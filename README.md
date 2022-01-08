# Anomaly Detection
## Project's scope

The primary goal of the Anomaly Detection project is to implement **Parzen window estimation** and **Gaussian mixture model**. The secondary goal is to train these models on train dataset (only normal samples), setup hyper-parametres on validation dataset (only normal samples) by maximizing likelihood of the model (proxy-parametr) and calculate roc-AUC on test data. The tertiary goal is to compare models with robust statistical test and decide which model is better. The project was implemented in Julia language and its key points are as follows:

- Parzen window estimation.
- Gaussian mixture model.
- Expectation–maximization algorithm
- Wilcoxon signed-rank test.

## Models & tools

##### Parzen window estimation
The Parzen window estimation is defined as <br/>
![\Large f(\vec{x}) = \frac{1}{hN}\sum_{i=1}^{N} k(\frac{\vec{x}-\vec{x}_i}{h})](https://latex.codecogs.com/svg.image?f(\vec{x})&space;=&space;\frac{1}{hN}\sum_{i=1}^{N}&space;k(\frac{\vec{x}-\vec{x}_i}{h}))
<br/>
where _h_ is a window size hyper-parametr, _N_ is count of train samples and _k_ is a kernel function defined as follow <br/>
![\Large k(\vec{x}) = \frac{1}{\sqrt{2\pi}}e^{-\frac{\vec{x}^T\vec{x}}{2}}](https://latex.codecogs.com/svg.image?k(\vec{x})&space;=&space;\frac{1}{\sqrt{2\pi}}e^{-\frac{\vec{x}^T\vec{x}}{2}})
<br/>

##### Gaussian mixture model
The Gaussian mixture model is defined as
<br/>
![\Large p(\vec{x}) = \sum_{i=1}^{K} \alpha_i G(\vec{x}|\vec{\mu}_i, \Sigma_i)](https://latex.codecogs.com/svg.image?p(\vec{x})&space;=&space;\sum_{i=1}^{K}&space;\alpha_i&space;G(\vec{x}|\vec{\mu}_i,&space;\Sigma_i))
<br/>
where _K_ is a number of components hyper-parametr, ![\Large \alpha_i](https://latex.codecogs.com/svg.image?\alpha_i) are component weights, ![\Large \vec{\mu_i}](https://latex.codecogs.com/svg.image?\vec{\mu_i}) are component means and ![\Large \Sigma_i](https://latex.codecogs.com/svg.image?\Sigma_i) are component covariance matrices. Gaussian component is defined as follow 
<br/>
![\Large G(\vec{x}|\vec{\mu}_i, \Sigma_i) = \frac{1}{\sqrt{(2\pi)^K|\Sigma_i|}}e^{-\frac{(\vec{x}-\vec{\mu}_i)^T\Sigma_i(\vec{x}-\vec{\mu}_i)}{2}}](https://latex.codecogs.com/svg.image?G(\vec{x}|\vec{\mu}_i,&space;\Sigma_i)&space;=&space;\frac{1}{\sqrt{(2\pi)^K|\Sigma_i|}}e^{-\frac{(\vec{x}-\vec{\mu}_i)^T\Sigma_i(\vec{x}-\vec{\mu}_i)}{2}})
<br/>
And the sum of component weights is equal one.
<br/>
![\Large \sum_{i=1}^{K}\alpha_i = 1](https://latex.codecogs.com/svg.image?\sum_{i=1}^{K}\alpha_i&space;=&space;1)
<br/>

**Expectation–maximization algorithm**
> Initialization Step:
- Randomly assign samples without replacement from the dataset to the component mean estimates. 
- Set all component covariance estimates to the sample covariance.
- Set all component distribution prior estimates to the uniform distribution ![\Large K^{-1}](https://latex.codecogs.com/svg.image?K^{-1})
> Expectation (E) Step: <br/>

![\Large \hat{\gamma}_{ik} = \frac{\hat{\alpha}_kG(\vec{x}_i|\vec{\mu}_k, \Sigma_k)}{\sum_{j=1}^{K}\hat{\alpha}_jG(\vec{x}_i|\vec{\mu}_j, \Sigma_j)}](https://latex.codecogs.com/svg.image?\hat{\gamma}_{ik}&space;=&space;\frac{\hat{\alpha}_kG(\vec{x}_i|\vec{\mu}_k,&space;\Sigma_k)}{\sum_{j=1}^{K}\hat{\alpha}_jG(\vec{x}_i|\vec{\mu}_j,&space;\Sigma_j)}) <br/>
> Maximization (M) Step: <br/>

![\Large \hat{\alpha}_k = \sum_{i=1}^{N} \frac{\hat{\gamma}_{ik}}{N} ](https://latex.codecogs.com/svg.image?\hat{\alpha}_k&space;=&space;\sum_{i=1}^{N}&space;\frac{\hat{\gamma}_{ik}}{N}&space;) <br/>

![\Large \hat{\mu}_{k} = \frac{\sum_{i=1}^{N}\hat{\gamma}_{ik}x_i}{\sum_{i=1}^{N}\hat{\gamma}_{ik}}](https://latex.codecogs.com/svg.image?\hat{\mu}_{k}&space;=&space;\frac{\sum_{i=1}^{N}\hat{\gamma}_{ik}x_i}{\sum_{i=1}^{N}\hat{\gamma}_{ik}}) <br/>

## License
MIT
