Scoring rules are functions that receive a (multivariate) distribution and a (multivariate) observation as input, and assign it a score. Scoring rules are aimed to evaluate the goodness so-called distributional regression models: models that can forecast target variables via probability distributions, rather than single estimators. Strictly proper scoring rules will minimize in expectancy towards the true distribution, they are considered powerful tools in distributional regression model evaluation (also known as distribution/distributional regression or density regression/forecasting).

In this repo, I have implemented several univariate and multivariate scoring rules as custom loss functions for Tensorflow. This allows users to train probabilistic forecasting models, by using scoring rules as ANN loss functions, instead of evaluation metrics. When done right, this results in probabilistic models that are on-par with current state-of-the-art.

The scoring rules and their implementations for specific families of distributions have partially been taken from literature, and partially from my Master's thesis, in which I introduced the Conditional CRPS, Rotational CRPS and Energy Score ensemble models. A link to the thesis will be shared as soon as it is available.

[![Distributional regression via scoring rules, overview.](https://i.postimg.cc/kMTn68S8/dist-reg-scoring-rules-example.png)](https://postimg.cc/9zq3sDJF)

If you are interested in this work, I emplore you also to check out some other interesting (multivariate) distributional regression approaches [Distributional Random Forest](https://github.com/lorismichel/drf) (Ćevid et al., 2020) and [Conditional-GAN for regression](https://github.com/kaggarwal/ganRegression) (Aggarwal et al., 2019) which are completely different, but also promising approaches.


# Scoring rules currently included #

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


# Scoring rules included in the (near) future #
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



# Sources #
1. Aggarwal, Karan & Kirchmeyer, Matthieu & Yadav, Pranjul & Keerthi, S. & Gallinari, Patrick. (2019). Regression with Conditional GAN. 
2. Baran, S, and S Lerch. 2015. “Log-Normal Distribution Based EMOS Models for Probabilistic Wind Speed Forecasting.” Quarterly Journal of the Royal Meteorological Society 141: 2289–99.
3. Ćevid, D., Michel, L., Näf, J., Meinshausen, N., & Bühlmann, P. (2020). Distributional random forests: Heterogeneity adjustment and multivariate distributional regression. arXiv preprint arXiv:2005.14458.
4. Gneiting, T, A E Raftery, A H Westveld III, and T Goldman. 2005. “Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation.” Monthly Weather Review 133: 1098–1118.
5. Grimit, E P, T Gneiting, V J Berrocal, and N A Johnson. 2006. “The Continuous Ranked Probability Score for Circular Variables and Its Application to Mesoscale Forecast Ensemble Verification.” Quarterly Journal of the Royal Meteorological Society 132: 2925–42.
6. Taylor, J.W. (2000), A quantile regression neural network approach to estimating the conditional density of multiperiod returns. J. Forecast., 19: 299-311. https://doi.org/10.1002/1099-131X(200007)19:4<299::AID-FOR775>3.0.CO;2-V
7. Zamo, Michaël & Naveau, Philippe. (2017). Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts. Mathematical Geosciences. 10.1007/s11004-017-9709-7. 
