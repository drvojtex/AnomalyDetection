# Anomaly Detection
## Project's scope

The primary goal of the Anomaly Detection project is to implement **Parzen window estimation** and **Gaussian mixture model** for anomaly detection. The secondary goal is to train these models on train dataset (only normal samples), set hyper-parametres on validation dataset (only normal samples) by maximizing likelihood of the model (proxy-parametr) and calculate roc-AUC on test data. The tertiary goal is to compare models with robust statistical test and decide which model is better. The project was implemented in Julia language and its key points are as follows:

- Parzen window estimation.
- Gaussian mixture model.
- Expectation–maximization algorithm
- Wilcoxon signed-rank test.

**The repository is organised as follow**
In the folder _src_ are implemented Gaussian mixture model, Parzen window estimation, EM algorithm, Wilcoxon signed-rank test and evaluation report (including roc-AUC). In the folder _data_anomalyproject_ are stored datasets, in the folder _auc_statistics_ are stored CSV files with roc-AUC for models (each CSV for one dataset, each line of CSV corresponds to one evaluation of learning and testing the model and each column of CSV corresponds to one model). In the file _main.jl_ are functions to learn and compare models. In the folder _examples_ are codes to evaluate models on random generated dataset with visualisation (human eye-check such as charts bellow).

<img src="https://github.com/kozvojtex/AnomalyDetection/blob/master/examples/gmm_heatmap.svg" width="50%" height="50%"><img src="https://github.com/kozvojtex/AnomalyDetection/blob/master/examples/parzenwindow_plot.svg" width="50%" height="50%">

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

##### Expectation–maximization algorithm 
Expectation maximization (EM) algorithm is a numerical technique for maximum likelihood estimation by iterative updating the model parameters. The first step, known as the expectation step or E step, consists of calculating the expectation of the component assignments for each data point given the model parameters. The second step the maximization (M) step, consists of maximizing the expectations calculated in the E step with respect to the model parameters. M step consists of updating the parametres. 
<br/>
> Initialization Step:
* Randomly assign samples without replacement from the dataset to the component mean estimates. 
* Set all component covariance estimates to the sample covariance.
* Set all component distribution prior estimates to the uniform distribution 1/K

> Expectation (E) Step: <br/>

![\Large \hat{\gamma}_{ik} = \frac{\hat{\alpha}_kG(\vec{x}_i|\vec{\mu}_k, \Sigma_k)}{\sum_{j=1}^{K}\hat{\alpha}_jG(\vec{x}_i|\vec{\mu}_j, \Sigma_j)}](https://latex.codecogs.com/svg.image?\hat{\gamma}_{ik}&space;=&space;\frac{\hat{\alpha}_kG(\vec{x}_i|\vec{\mu}_k,&space;\Sigma_k)}{\sum_{j=1}^{K}\hat{\alpha}_jG(\vec{x}_i|\vec{\mu}_j,&space;\Sigma_j)}) <br/>
> Maximization (M) Step: <br/>

![\Large \hat{\alpha}_k = \sum_{i=1}^{N} \frac{\hat{\gamma}_{ik}}{N} ](https://latex.codecogs.com/svg.image?\hat{\alpha}_k&space;=&space;\sum_{i=1}^{N}&space;\frac{\hat{\gamma}_{ik}}{N}&space;) <br/>

![\Large \hat{\\vec{mu}}_{k} = \frac{\sum_{i=1}^{N}\hat{\gamma}_{ik}\vec{x}_i}{\sum_{i=1}^{N}\hat{\gamma}_{ik}}](https://latex.codecogs.com/svg.image?\hat{\vec{\mu}}_{k}&space;=&space;\frac{\sum_{i=1}^{N}\hat{\gamma}_{ik}\vec{x}_i}{\sum_{i=1}^{N}\hat{\gamma}_{ik}}) <br/>

![\Large \hat{\Sigma}^2_k  = \frac{\sum_{i=1}^{N}\hat{\gamma}_{ik}(\vec{x}_i - \hat{\vec{\mu}}_i)\cdot(\vec{x}_i - \hat{\vec{\mu}}_i)^T}{\sum_{i=1}^{N}\hat{\gamma}_{ik}}](https://latex.codecogs.com/svg.image?\hat{\Sigma}^2_k&space;&space;=&space;\frac{\sum_{i=1}^{N}\hat{\gamma}_{ik}(\vec{x}_i&space;-&space;\hat{\vec{\mu}}_i)\cdot(\vec{x}_i&space;-&space;\hat{\vec{\mu}}_i)^T}{\sum_{i=1}^{N}\hat{\gamma}_{ik}}) <br/>

If the determinant of the covariance matrix is less than 10e-10, then an Identity matrix multiplied by 10e-5 is added to the covariance matrix. This prevents the zero determinant of the covariance matrix.

##### Wilcoxon signed-rank test
The Wilcoxon signed-rank test is used to decide if difference between pair follows a symmetric distribution around zero. It is used as a robust statistical test, because unlike Student's t-test, the Wilcoxon signed-rank test does not assume that the differences between paired samples are normally distributed. On a large dataset it has greater statistical power than Student's t-test and is more likely to produce a statistically significant result. Steps are follow:
1. For each pair of values compute its difference and discard zero differences.
2. Sort the differences in ascending order and assign them a rank.
3. Compute test statistic as <br/>
![\Large W = \sum_{i=1}^{N}\textrm{sign}(y_i-x_i)R_i](https://latex.codecogs.com/svg.image?W&space;=&space;\sum_{i=1}^{N}\textrm{sign}(y_i-x_i)R_i) <br/>
4. where ![\Large R_i](https://latex.codecogs.com/svg.image?R_i) are ranks. 
5. The z-score is defined as<br/>
&nbsp; <img src="https://latex.codecogs.com/svg.image?z&space;=&space;\frac{W}{\sigma_W}" title="z = \frac{W}{\sigma_W}" /><br/>
&nbsp; where<br/>
&nbsp; <img src="https://latex.codecogs.com/svg.image?\sigma^2_W&space;=&space;\frac{N(N&plus;1)(2N&plus;1)}{6}" title="\sigma^2_W = \frac{N(N+1)(2N+1)}{6}" /><br/>

Hypothesis are:
- Null hypothesis: the differences of observations are symmetric about zero mean.
- Alternative hypothesis (two-sided): the differences of observations are not symmetric about zero mean.

The metric to compare models by Wilcoxon test was roc-AUC.

## Experimental settings, evaluation and results

The evaluation of the model took place on each dataset as follows. Normal samples were randomly shuffled. Half of the normal samples were used as a training set, 25% of normal samples were used as a validation set, and 25% of normal samples and all anomalous samples were used as a test set. Both models (GMM and PWE) were trained on training set. The hyperparameters (number of components for GMM and window-size for PWE) were set by maximizing likelihood of the model on validation dataset. GMM maximized likelihood for the number of components 2 to 10, with the EM algorithm always was performed with 60 iterations. PWE maximized the likelihood for window sizes from 0.01 to 10 with 0.01 increments. The roc-AUC was calculated on the test data set for the best model (best GMM and best PWE) based on likelihood. The procedure described above was repeated ten times on each dataset. <br/>
The dataframe (two columns for models, 90 rows for models roc-AUC on 9 datasets) was the input observation for Wilcoxon test. The goal of the test was to decide if there is a statistical difference between means of roc-AUC results of the models. The test was performed with the type I error rate (significance level) 0.05. Z-score was -0.92746, which is less than 1.96 and greater than -1.96. The null hypothesis was accepted. <br/>
There is no statistical significance of the difference between means of roc-AUC observations of the models. According to the obtained results on the given datasets with roc-AUC metrics, the models give the same results. This may be due to the fact that, due to the high time required, the EM algorithm was limited to 60 iterations, which was not enough for some datasets. 

## Sources

- https://cw.fel.cvut.cz/wiki/_media/courses/b4m36san/robust.pdf
- https://cw.fel.cvut.cz/wiki/_media/courses/b4m36san/san_anomaly.pdf
- https://brilliant.org/wiki/gaussian-mixture-model/
- https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
- http://users.sussex.ac.uk/~grahamh/RM1web/WilcoxonHandoout2011.pdf
- https://pro.arcgis.com/en/pro-app/2.8/tool-reference/spatial-statistics/what-is-a-z-score-what-is-a-p-value.htm
- https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test


## License
MIT
