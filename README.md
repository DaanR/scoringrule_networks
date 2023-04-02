Strictly proper scoring rules are distributional evaluation metrics, that take as input a forecasted distribution and an observation, and assign it a realvalued score with the objective of minimization. Strictly proper scoring rules can be a valuable loss function in order to train so called distributional regression models: models that forecast target variables via conditional distributions rather than single estimators (see 3: Background). In this repo, I have implemented custom loss functions usable in Tensorflow, that implement several well-known and lesser-known strictly proper scoring rules, allowing for probabilistic forecasting via artificial neural networks. In general, we can summarize the approaches into three categories: parametric models that forecast a distribution via its parameters (e.g. mean and standard deviation for Gaussian models), models that forecast an emsemble of points that define the distribution, and (univariate only) models that forecast a distribution's quantiles.

[![Distributional regression via scoring rules, overview.](https://i.postimg.cc/kMTn68S8/dist-reg-scoring-rules-example.png)](https://postimg.cc/9zq3sDJF)

Scoring rules implemented for univariate target distributions include the Log-score/NLPD/maximum likelihood score, the Continuous Ranked Probability Score (CRPS) and quantile loss. For multivariate distributions, the multivariate Log-score, the Energy score, Conditional CRPS and Rotational CRPS.

The scoring rules and their implementations for specific families of distributions have partially been taken from literature, and our paper, in which we introduced the Conditional CRPS, Rotational CRPS and Energy Score ensemble models. A link to the thesis will be shared as soon as it is available.

# 1: Scoring rule overview #

### Currently included ###
* **Univariate scoring rules**
    * Continuous Ranked Probability Score (CRPS)
       * for Gaussian distributions. (Gneiting et al., 2005)
       * for Gaussian mixture distributions. (Grimit et al, 2006)
    * Quantile regressing neural networks (QRNN) (Taylor, 2000)
    * Log score/NLPD/MLE
    	* for Gaussian distributions.
    	* for Gaussian mixture distributions.
* **Multivariate scoring rules**
    * None yet.
### Included in the near-future ###
* **Univariate scoring rules**
    * Continuous Ranked Probability Score (CRPS)
       * for Log-normal distributions. (Baran & Lerch, 2015)
       * Non-parametric ensemble CRPS (Zamo & Naveau, 2017)
* **Multivariate scoring rules**
    * Energy score (ES)
       * Non-parametric ensemble ES
    * Conditional CRPS (CCRPS)
       * for mv. Gaussian distributions
       * for mv. Gaussian mixture distributions
    * Rotational CRPS (RCRPS)
       * for mv. Gaussian distributions
    * Log score/NLPD/MLE
       * for mv. Gaussian distributions
       * for mv. Gaussian mixture distributions

# 2: Background: why strictly proper scoring rules? #
A (strictly proper) scoring rule ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20R) is a mapping ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20R%3A%20%5Cmathcal%7BD%7D%28%5Cmathcal%7BY%7D%29%20%5Ctimes%20%5Cmathcal%7BY%7D%20%5Cto%20%5Cmathbb%7BR%7D%20%5Ccup%20%5C%7B%5Cinfty%5C%7D) that takes as inputs a forecasted distribution over a space ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20%5Cmathcal%7BY%7D) and an observation in ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20%5Cmathcal%7BY%7D), and assigns this pair a score with the objective of minimization. A scoring rule is called strictly proper if they optimize in expectency to the (unknown) true distribution. More formally, for every two distinct ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20A%2C%20B%20%5Cin%20%5Cmathcal%7BY%7D) we have:

![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20%5Cmathbb%7BE%7D_%7By%20%5Csim%20A%7D%5BR%28A%2Cy%29%5D%20%3C%20%5Cmathbb%7BE%7D_%7By%20%5Csim%20B%7D%5BR%28B%2C%20y%29%5D)

Strictly proper scoring rules can be effectively used to train models to forecast conditional distributions. Consider models ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20m%3A%20%5Cmathcal%7BX%7D%20%5Cto%20%5Cmathcal%7BD%7D%28%5Cmathcal%7BY%7D%29) that map variables from a input space ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20%5Cmathcal%7BX%7D) to distributions over a target space ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20%5Cmathcal%7BY%7D). If the target variables ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20y%20%5Cin%20%5Cmathcal%7BY%7D) are stochastically dependent on input variables ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20x%20%5Cin%20%5Cmathcal%7BX%7D) (i.e. they can be modelled as random samples from some unknown distribution ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20P) over joint random variables ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20%28X%2CY%29%20%5Cin%20%5Cmathcal%7BX%7D%20%5Ctimes%20%5Cmathcal%7BY%7D), then it can be proven that the expected strictly proper score is minimized iff m learns the ``correct'' conditionals.

![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20%5Cmathbb%7BE%7D_%7Bx%2Cy%20%5Csim%20P%7D%5BR%28m%28x%29%2Cy%29%5D%20%5Ctext%7B%20minimized%20%7D%20%5CLongleftrightarrow%20m%28x%29%20%3D%20P%28Y%20%7C%20X%20%3D%20x%29)

When done right, this results in probabilistic models that are on-par with current state-of-the-art.

This might seem as a rather counter-intuitive way of defining our learning setup, but in fact, it's not that different to ``classical regression''. Here, the input and target data is identical, but we define models ![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20m%5E*%3A%20%5Cmathcal%7BX%7D%20%5Cto%20%5Cmathcal%7BY%7D) that forecast target variables directly rather than via distributions. Mean squared error plays a similar role to scoring rules, in which the expected MSE is minimized iff the model always forecasts the correct conditional expectancy:

![image](https://latex.codecogs.com/png.latex?%5Cdpi%7B80%7D%20%5Clarge%20%5Cmathbb%7BE%7D_%7Bx%2Cy%20%5Csim%20P%7D%5B%28m%5E*%28x%29%20-%20y%29%5E2%5D%20%5Ctext%7B%20minimized%20%7D%20%5CLongleftrightarrow%20m%5E*%28x%29%20%3D%20%5Cmathbb%7BE%7D%28Y%20%7C%20X%20%3D%20x%29)

# 3: Other work #

If you are interested in this work, I emplore you also to check out some other interesting (multivariate) distributional regression approaches [Distributional Random Forest](https://github.com/lorismichel/drf) (Ćevid et al., 2020) and [Conditional-GAN for regression](https://github.com/kaggarwal/ganRegression) (Aggarwal et al., 2019) which are completely different approaches, but share the learning setup described above.



# Sources #
1. Aggarwal, Karan & Kirchmeyer, Matthieu & Yadav, Pranjul & Keerthi, S. & Gallinari, Patrick. (2019). Regression with Conditional GAN. 
2. Baran, S, and S Lerch. 2015. “Log-Normal Distribution Based EMOS Models for Probabilistic Wind Speed Forecasting.” Quarterly Journal of the Royal Meteorological Society 141: 2289–99.
3. Ćevid, D., Michel, L., Näf, J., Meinshausen, N., & Bühlmann, P. (2020). Distributional random forests: Heterogeneity adjustment and multivariate distributional regression. arXiv preprint arXiv:2005.14458.
4. Gneiting, T, A E Raftery, A H Westveld III, and T Goldman. 2005. “Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation.” Monthly Weather Review 133: 1098–1118.
5. Grimit, E P, T Gneiting, V J Berrocal, and N A Johnson. 2006. “The Continuous Ranked Probability Score for Circular Variables and Its Application to Mesoscale Forecast Ensemble Verification.” Quarterly Journal of the Royal Meteorological Society 132: 2925–42.
6. Taylor, J.W. (2000), A quantile regression neural network approach to estimating the conditional density of multiperiod returns. J. Forecast., 19: 299-311. https://doi.org/10.1002/1099-131X(200007)19:4<299::AID-FOR775>3.0.CO;2-V
7. Zamo, Michaël & Naveau, Philippe. (2017). Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts. Mathematical Geosciences. 10.1007/s11004-017-9709-7. 
