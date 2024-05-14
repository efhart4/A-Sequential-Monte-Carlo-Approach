# A Sequential Monte Carlo Approach to Endogenous Time Varying Parameter Models

## Overview
This repository contains the files and data related to the research paper titled "A Sequential Monte Carlo Approach to Endogenous Time Varying Parameter Models". The paper proposes a method to evaluate the likelihood of a nonlinear model with time-varying parameters and endogenous variables.

## Research Paper
1. **Sequential_Monte_Carlo_Endogenous_TVP_Models.pdf**: This is the main research paper in PDF format. It presents the proposed method, its implementation, and the results of the evaluation.

## Python Code for Experiments
1. **Model 1**: This folder contains Python code to run experiments related to the proposed method on Model 1. The first test will use a nonlinear model with an additive error. This sample model was chosen because of its relation to a sticky information Phillips curve with drift in the rate of attentiveness. Endogeneity is known to be an issue when estimating the Phillips curve.
2. **Robustness**: This folder contains Python code to run experiments testing the robustness of the proposed method under various conditions and scenarios. The method relies heavily on the joint normal assumption between the variables. To test the robustness of the extended particle filter to misspecification, I will compare particle filters when the true error has a joint t-distribution with three degrees of freedom.
3. **Role_instruments**: This folder contains Python code to analyze the role of instrumental variables in the proposed method. The joint normal assumption furnishes the necessary structure for estimating the models and potentially enables us to directly model a relationship between the endogenous variable and the measurement error. In this experiment, there actually is an instrument (m does not equal zero), but the researcher assumes that m equals zero. A useful tool if a researcher doesn’t have access to instruments or available instruments are weak. To test the effect of excluding instruments, I will simulate a model with instruments and compare a particle filter that uses those instruments and one that does not.

