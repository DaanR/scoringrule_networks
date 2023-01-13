This repo contains Keras/Tensorflow based implementations of various strictly proper scoring rules. These scoring rules are usable as loss functions for Artificial Neural Networks, allowing users to train probabilistic forecasting models (also known as distribution/distributional regression or density regression/forecasting), that can forecast target variables via probability distributions.

# Scoring rules currently included #

* **Univariate scoring rules**
    * Continuous Ranked Probability Score (CRPS)
       * for Gaussian distributions. (Gneiting et al., 2005)
       * for Gaussian mixture distributions. (Grimit et al, 2006)
    * Quantile regressing neural networks (QRNN) (Taylor, 2000)
* **Multivariate scoring rules**
    * None yet.


# Scoring rules included in the (near) future #
* **Univariate scoring rules**
    * Continuous Ranked Probability Score (CRPS)
       * for Log-normal distributions. (Baran & Lerch, 2015)
       * Non-parametric ensemble CRPS (Zamo & Naveau, 2017)
    * Log score/NLPD/MLE
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
1. Baran, S, and S Lerch. 2015. “Log-Normal Distribution Based EMOS Models for Probabilistic Wind Speed Forecasting.” Quarterly Journal of the Royal Meteorological Society 141: 2289–99.
2. Gneiting, T, A E Raftery, A H Westveld III, and T Goldman. 2005. “Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation.” Monthly Weather Review 133: 1098–1118.
3. Grimit, E P, T Gneiting, V J Berrocal, and N A Johnson. 2006. “The Continuous Ranked Probability Score for Circular Variables and Its Application to Mesoscale Forecast Ensemble Verification.” Quarterly Journal of the Royal Meteorological Society 132: 2925–42.
4. Taylor, J.W. (2000), A quantile regression neural network approach to estimating the conditional density of multiperiod returns. J. Forecast., 19: 299-311. https://doi.org/10.1002/1099-131X(200007)19:4<299::AID-FOR775>3.0.CO;2-V
5. Zamo, Michaël & Naveau, Philippe. (2017). Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts. Mathematical Geosciences. 10.1007/s11004-017-9709-7. 
