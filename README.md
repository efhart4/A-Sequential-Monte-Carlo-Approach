# A Sequential Monte Carlo Approach to Endogenous Time Varying Parameter Models

I propose a method to evaluate the likelihood of a nonlinear model with time-varying parameters and endogenous variables. Existing techniques to estimate time-varying parameter models with endogenous variables are restricted to conditionally linear models. The proposed approach modifies a Sequential Monte Carlo filter to evaluate the likelihood of a nonlinear process with an endogenous variable. The modified filter augments the typical measurement and state equations with an equation incorporating instrumental variables. I evaluate the performance of a Bayesian estimator based on the likelihood calculation using simulations and find that the approach generates accurate estimates of both parameters and the unobserved time-varying parameter.


* Code for experiments can be found in folders: Model1, RoleInstruments and Robust.
