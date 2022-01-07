# Anomaly Detection
## Gaussian mixture model & parzen window estimation

The goal of the Anomaly Detection project is... The Julia project of GMM a PWE for anomaly detection.

- Parzen window estimation.
- Gaussian mixture model.
- Wilcoxon signed-rank test.

## Models

##### Parzen window estimation
![\Large f(\vec{x}) = \frac{1}{hN}\sum_{i=1}^{N} k(\frac{\vec{x}-\vec{x_i}}{h})](https://latex.codecogs.com/svg.image?f(\vec{x})&space;=&space;\frac{1}{hN}\sum_{i=1}^{N}&space;k(\frac{\vec{x}-\vec{x_i}}{h}))
<br/>
![\Large k(\vec{x}) = \frac{1}{\sqrt{2\pi}}e^{-\frac{\vec{x}^T\vec{x}}{2}}](https://latex.codecogs.com/svg.image?k(\vec{x})&space;=&space;\frac{1}{\sqrt{2\pi}}e^{-\frac{\vec{x}^T\vec{x}}{2}})

##### Gaussian mixture model
The Gaussian mixture model is defined as
![\Large p(\vec{x}) = \sum_{i=1}^{K} \alpha_i G(\vec{x}|\vec{\mu_i}, \Sigma_i)](https://latex.codecogs.com/svg.image?p(\vec{x})&space;=&space;\sum_{i=1}^{K}&space;\alpha_i&space;G(\vec{x}|\vec{\mu_i},&space;\Sigma_i))
<br/>
where ![\Large \alpha_i](https://latex.codecogs.com/svg.image?\alpha_i) are component weights, ![\Large \vec{\mu_i}](https://latex.codecogs.com/svg.image?\vec{\mu_i}) are component means and ![\Large \Sigma_i](https://latex.codecogs.com/svg.image?\Sigma_i) are component covariance matrices. Gaussian component is defined as follow 
<br/>
![\Large G(\vec{x}|\vec{\mu_i}, \Sigma_i) = \frac{1}{\sqrt{(2\pi)^K|\Sigma_i|}}e^{-\frac{(\vec{x}-\vec{\mu_i})^T\Sigma_i(\vec{x}-\vec{\mu_i})}{2}}](https://latex.codecogs.com/svg.image?G(\vec{x}|\vec{\mu_i},&space;\Sigma_i)&space;=&space;\frac{1}{\sqrt{(2\pi)^K|\Sigma_i|}}e^{-\frac{(\vec{x}-\vec{\mu_i})^T\Sigma_i(\vec{x}-\vec{\mu_i})}{2}})
And the sum of component weights is equal one.
<br/>
![\Large \sum_{i=1}^{K}\alpha_i = 1](https://latex.codecogs.com/svg.image?\sum_{i=1}^{K}\alpha_i&space;=&space;1)

**Expectation–maximization algorithm**
> Initialization Step:
- Randomly assign samples without replacement from the dataset to the component mean estimates. 
- Set all component covariance estimates to the sample covariance.
- Set all component distribution prior estimates to the uniform distribution ![\Large K^{-1}](https://latex.codecogs.com/svg.image?K^{-1})
> Expectation (E) Step:

> Maximization (M) Step:

## License
MIT
