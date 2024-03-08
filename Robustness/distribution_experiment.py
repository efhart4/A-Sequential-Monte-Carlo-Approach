import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_t
from datetime import datetime

from scipy.special import gamma
from numpy.linalg import det, inv
from numpy import pi

"""
TVP model with the following form:
y = TVP * x1 + x2 * beta + measurement_error
x1= m * z + instrumental_error
TVP = TVP(-1) + TVP_error

theta: 
    Measure_StdDev = theta_in[0]
    beta = theta_in[1]
    TVP0 = theta_in[2]
    Transition_StdDev = theta_in[3]
    Instrument_StdDev = theta_in[4]
    correlation_coefficient = theta_in[5]
    m = theta_in[6]
"""


# Extended Particle Filter Function T distribution
#------------------------------------------------------------
def Extended_Particle_Filter_T(y, x1, x2, z,  theta_in):
    """
    Estimate the path of a time varying parameter using a particle filter.

    Args:
        y: output vector of state space model
        x1: covariate on the TVP (potentially endogenous)
        x2: linear covariates
        z: instruments
        theta_in: all model parameters
    
    
    Returns:
        (0) a boolean indicating if the filter converged. 
        (1) the estimated path of the TVP.
        (2) the log likelihood of the model.

    """

    # Initialize the parameters
    num_particles = 500
    Sample = len(y)
    

    # theta_in
    Measure_StdDev = theta_in[0]
    beta = theta_in[1]
    TVP0 = theta_in[2]
    Transition_StdDev = theta_in[3]
    Instrument_StdDev = theta_in[4]
    correlation_coefficient = theta_in[5]
    m = theta_in[6]

       

    # Calculate the covariance between the measurement error and the instrument error
    measure_instrument_cov = correlation_coefficient * Measure_StdDev * Instrument_StdDev
    

    # Create variance-covariance matrix and mean vector for the measurement and instrument error
    Sigma = np.array([[Measure_StdDev**2, measure_instrument_cov], 
                      [measure_instrument_cov, Instrument_StdDev**2]])
    mu = np.array([0, 0])
    

    # Initialize the matrices
    TVP_est_matrix = np.zeros((Sample, num_particles))
    TVP_est_matrix[0, :] = TVP0
    TVP_Path = np.zeros(Sample)
    TVP_Path[0] = TVP0


    # Initialize the log likelihood
    log_lik = 0



    # Loop through the data
    for k in range(1, Sample):

        # Propose a vector of states for the current time step
        variation = np.random.normal(0, Transition_StdDev, num_particles)
        proposed_TVP_vector = TVP_est_matrix[k - 1, :] + variation


        # Find the predicted value of the measure and the error
        prediction = proposed_TVP_vector * x1[k] + beta * x2[k] 
        error_vec = (y[k] - prediction)


        # Find the error of the instrumental equation
        endogenous_prediction = m * z[k] 
        instrument_error = x1[k] - endogenous_prediction


        # Find the likelihood of the proposed states using a joint likelihood method
        likelihood_over_particles = joint_likelihood_T(error_vec, instrument_error, mu, Sigma)
        likelihood_over_particles[np.isnan(likelihood_over_particles)] = 0


        # If all the likelihoods are zero, return false
        if np.sum(likelihood_over_particles) == 0:
            #print('All likelihoods are zero across the particles')
            return [False, None, None]


        # Normalize the likelihoods and choose the states
        probability_over_particles = likelihood_over_particles / np.sum(likelihood_over_particles)
        chosen_states = np.random.choice(num_particles, num_particles, p=probability_over_particles)


        # Save the chosen states 
        chosen_TVP_vector = proposed_TVP_vector[chosen_states]
        TVP_est_matrix[k, :] = chosen_TVP_vector


        # Find the current total log likelihood of the chosen states
        log_pdf_itr = np.log(np.mean(likelihood_over_particles[chosen_states]))
        log_lik += log_pdf_itr


        # Save the current estimate of the TVP
        TVP_Path[k] = np.mean(chosen_TVP_vector)

    return [True, TVP_Path, log_lik]

# Extended Particle Filter Function N distributions
#------------------------------------------------------------
def Extended_Particle_Filter_N(y, x1, x2, z,  theta_in):
    """
    Estimate the path of a time varying parameter using a particle filter.

    Args:
        y: output vector of state space model
        x1: covariate on the TVP (potentially endogenous)
        x2: linear covariates
        z: instruments
        theta_in: all model parameters
    
    
    Returns:
        (0) a boolean indicating if the filter converged. 
        (1) the estimated path of the TVP.
        (2) the log likelihood of the model.

    """

    # Initialize the parameters
    num_particles = 500
    Sample = len(y)
    

    # theta_in
    Measure_StdDev = theta_in[0]
    beta = theta_in[1]
    TVP0 = theta_in[2]
    Transition_StdDev = theta_in[3]
    Instrument_StdDev = theta_in[4]
    correlation_coefficient = theta_in[5]
    m = theta_in[6]

       

    # Calculate the covariance between the measurement error and the instrument error
    measure_instrument_cov = correlation_coefficient * Measure_StdDev * Instrument_StdDev
    

    # Create variance-covariance matrix and mean vector for the measurement and instrument error
    Sigma = np.array([[Measure_StdDev**2, measure_instrument_cov], 
                      [measure_instrument_cov, Instrument_StdDev**2]])
    mu = np.array([0, 0])
    

    # Initialize the matrices
    TVP_est_matrix = np.zeros((Sample, num_particles))
    TVP_est_matrix[0, :] = TVP0
    TVP_Path = np.zeros(Sample)
    TVP_Path[0] = TVP0


    # Initialize the log likelihood
    log_lik = 0



    # Loop through the data
    for k in range(1, Sample):

        # Propose a vector of states for the current time step
        variation = np.random.normal(0, Transition_StdDev, num_particles)
        proposed_TVP_vector = TVP_est_matrix[k - 1, :] + variation


        # Find the predicted value of the measure and the error
        prediction = proposed_TVP_vector * x1[k] + beta * x2[k] 
        error_vec = (y[k] - prediction)


        # Find the error of the instrumental equation
        endogenous_prediction = m * z[k] 
        instrument_error = x1[k] - endogenous_prediction


        # Find the likelihood of the proposed states using a joint likelihood method
        likelihood_over_particles = joint_likelihood_N(error_vec, instrument_error, mu, Sigma)
        likelihood_over_particles[np.isnan(likelihood_over_particles)] = 0


        # If all the likelihoods are zero, return false
        if np.sum(likelihood_over_particles) == 0:
            #print('All likelihoods are zero across the particles')
            return [False, None, None]


        # Normalize the likelihoods and choose the states
        probability_over_particles = likelihood_over_particles / np.sum(likelihood_over_particles)
        chosen_states = np.random.choice(num_particles, num_particles, p=probability_over_particles)


        # Save the chosen states 
        chosen_TVP_vector = proposed_TVP_vector[chosen_states]
        TVP_est_matrix[k, :] = chosen_TVP_vector


        # Find the current total log likelihood of the chosen states
        log_pdf_itr = np.log(np.mean(likelihood_over_particles[chosen_states]))
        log_lik += log_pdf_itr


        # Save the current estimate of the TVP
        TVP_Path[k] = np.mean(chosen_TVP_vector)

    return [True, TVP_Path, log_lik]


# Joint Likelihood Function N distributions
#------------------------------------------------------------
def joint_likelihood_N(error_vec, instrument_error_t, mu, Sigma):
    particle_count = len(error_vec)
    particle_instrument_error_vec = np.column_stack((error_vec, np.full(particle_count, instrument_error_t)))
    likelihood_over_particles = multivariate_normal(mean=mu, cov=Sigma).pdf(particle_instrument_error_vec)
    return likelihood_over_particles


# Joint Likelihood Function T distribution
#------------------------------------------------------------
def joint_likelihood_T(error_vec, instrument_error_t, mu, Sigma):
    particle_count = len(error_vec)
    df = 3
    particle_instrument_error_vec = np.column_stack((error_vec, np.full(particle_count, instrument_error_t)))
    likelihood_over_particles = multivariate_t(mu, Sigma, df).pdf(particle_instrument_error_vec)
    return likelihood_over_particles


# Generate Data Function
#------------------------------------------------------------
def Generate_State_Space_Model_Data(theta):   
    """
    Generate State Space Model Data when there is endogeniety.

    Args:
    -----
        theta: model paramters 
    
    Returns:
    --------
        df : DataFrame
            A DataFrame containing the generated data.
    """
    
    # theta: 
    Measure_StdDev = theta[0]
    beta = theta[1]
    TVP0 = theta[2]
    Transition_StdDev = theta[3]
    Instrument_StdDev = theta[4]
    correlation_coefficient = theta[5]
    m = theta[6]

    # Define the number of samples
    num_samples = 250
    

    # Create Two Exogenous variable
    Exo_mean = [1, 1] # mean vector
    Exo_cov = [[1, 0], [0, 1]]  # covariance matrix
    z, x2 = np.random.multivariate_normal(Exo_mean, Exo_cov, num_samples).T


    # Create State Variable, a time varying parameter that follows a random walk
    Steps_RW_mean = 0    # Mean of steps
    Steps_RW = np.random.normal(Steps_RW_mean, Transition_StdDev, num_samples) # generate from normal distribution
    Steps_RW[0] = TVP0 # replace step 0 with an initial value
    TVP = np.cumsum(Steps_RW) # create a series where each new value is the sum of the previous values plus a new random shock, which is the definition of a random walk.


    # Define the variances for the measurement error and the endogenous variable error
    measurement_error_variance = (Measure_StdDev)**2
    instrument_error_variance = (Instrument_StdDev)**2


    # Calculate the covariance between the two error terms
    error_covariance = correlation_coefficient * Measure_StdDev * Instrument_StdDev

    # Define the covariance matrix for the error terms
    # The diagonal elements are the variances of the error terms
    # The off-diagonal elements are the covariances between the error terms
    error_covariance_matrix = [
        [measurement_error_variance, error_covariance],
        [error_covariance, instrument_error_variance]
    ]


    # Define the mean vector for the error terms
    error_mean_vector = [0, 0] # mean vector for the error terms (measurement error and endogenous variable error)


    # Generate the error terms for the measurement and the endogenous variable
    # The generated error terms follow a multivariate T distribution with the defined mean vector and covariance matrix
    df = 3
    measurement_error, endogenous_variable_error = multivariate_t(error_mean_vector, error_covariance_matrix, df).rvs(size=num_samples).T # np.random.multivariate_normal(error_mean_vector, error_covariance_matrix, num_samples).T

    
    # Create (potentially) Endogenous variable
    x1 =  m * z + endogenous_variable_error


    # Create measurement variable
    y = TVP * x1 + x2 * beta + measurement_error


    # Create a DataFrame from the generated data
    data = np.column_stack([y, x1, x2, TVP, z,measurement_error, endogenous_variable_error])

    # Define the column names and create the DataFrame in one step
    df = pd.DataFrame(data, columns=['y', 'x1', 'x2', 'TVP',  'z', 'measurement_error', 'endogenous_variable_error'])

    return df

# Prior Probability for Extended Particle Filter 
#------------------------------------------------------------
def prior_probability(theta_in, uniform_priors):
    """
    Calculate the log prior probability of the model parameters.

    Args:
    -----
        theta_in: model parameters
        uniform_priors: DataFrame of the parameters of the uniform prior distributions

    Returns:
    --------
        log_prior: float
            The log prior probability of the model parameters
    """

    # theta_in
    Measure_StdDev = theta_in[0]
    beta = theta_in[1]
    TVP0 = theta_in[2]
    Transition_StdDev = theta_in[3]
    Instrument_StdDev = theta_in[4]
    correlation_coefficient = theta_in[5]
    m = theta_in[6]


    # Calculate the prior probability of the initial parameter values
    start_beta = uniform_priors.loc['start', 'beta']
    end_beta = uniform_priors.loc['end', 'beta']
    prob_beta = 1/(end_beta - start_beta) if start_beta <= beta <= end_beta else 0

    start_TVP0 = uniform_priors.loc['start', 'TVP0']
    end_TVP0 = uniform_priors.loc['end', 'TVP0']
    prob_TVP0 = 1/(end_TVP0 - start_TVP0) if start_TVP0 <= TVP0 <= end_TVP0 else 0

    start_Measure_StdDev = uniform_priors.loc['start', 'Measure_StdDev']
    end_Measure_StdDev = uniform_priors.loc['end', 'Measure_StdDev']
    prob_measurement_StdDev = 1/(end_Measure_StdDev - start_Measure_StdDev) if start_Measure_StdDev <= Measure_StdDev <= end_Measure_StdDev else 0
    
    start_TVP_StdDev = uniform_priors.loc['start', 'Transition_StdDev']
    end_TVP_StdDev = uniform_priors.loc['end', 'Transition_StdDev']
    prob_TVP_StdDev = 1/(end_TVP_StdDev - start_TVP_StdDev) if start_TVP_StdDev <= Transition_StdDev <= end_TVP_StdDev else 0

    start_Instrument_StdDev = uniform_priors.loc['start', 'Instrument_StdDev']
    end_Instrument_StdDev = uniform_priors.loc['end', 'Instrument_StdDev']
    prob_instrumental_StdDev = 1/(end_Instrument_StdDev - start_Instrument_StdDev) if start_Instrument_StdDev <= Instrument_StdDev <= end_Instrument_StdDev else 0

    start_correlation_coefficient = uniform_priors.loc['start', 'correlation_coefficient']
    end_correlation_coefficient = uniform_priors.loc['end', 'correlation_coefficient']
    prob_correlation_coefficient = 1/(end_correlation_coefficient - start_correlation_coefficient) if start_correlation_coefficient <= correlation_coefficient <= end_correlation_coefficient else 0

    start_m = uniform_priors.loc['start', 'm']
    end_m = uniform_priors.loc['end', 'm']
    prob_m = 1/(end_m - start_m) if start_m <= m <= end_m else 0



    # Calculate the product of prior probabilities
    prior_product = ( prob_beta * prob_TVP0 * prob_measurement_StdDev * prob_TVP_StdDev 
                     * prob_correlation_coefficient * prob_m * prob_instrumental_StdDev)

    if prior_product == 0:
        zero_values = []
        if prob_TVP0 == 0:
            zero_values.append("prob_TVP0")
        if prob_measurement_StdDev == 0:
            zero_values.append("prob_measurement_StdDev")
        if prob_instrumental_StdDev == 0:
            zero_values.append("prob_instrumental_StdDev")
        if prob_TVP_StdDev == 0:
            zero_values.append("prob_TVP_StdDev")
        if prob_correlation_coefficient == 0:
            zero_values.append("prob_correlation_coefficient")
        if prob_m == 0:
            zero_values.append("prob_m")
        if prob_beta == 0:
            zero_values.append("prob_beta")
        
        #print("Product of prior probabilities is zero.")
        #print("Zero values:", zero_values)
        return False, None

    # Calculate the log of the prior probability
    log_prior = np.log(prior_product)

    return True, log_prior


# Metropolis Hasting Sampler for Extended Particle Filter T distributions
#------------------------------------------------------------
def Metropolis_Hasting_Sampler_T(y, x1, x2, z, true_theta):
    """
    sample model parameters using Metropolis Hasting Sampler
    

    Args:
        y: output vector of state space model
        x1: covariate on the TVP (potentially endogenous)
        x2: linear covariates
        z: instruments
        theta_in: all model parameters


    Outline of the function:
    --------
        1. Define prior parameters for the model
        2. Set the number of draws
        3. Initialize the parameter values for the Metropolis- Hasting Sampler
        4. Calculate the prior probability of the initial parameter values
        5. Start Sampling
        6. Accept/Reject Draws using the prior probability and particle filter
        7. Store the accepted draws
        8. Return the prior and posterior distributions

    Returns:
    --------
        A list of 
        (0) a boolean indicating if the filter converged.
        (1) the estimated paths of the TVP.
        (2) accepted samples of theta
        (3) uniform priors
        (4) acceptance rate
    """


    # Uniform Prior Distribution
    index3 = ['start', 'end']

    # For the variables of interest, define the bounds of the uniform prior distributions 
    data3 = {
        'Measure_StdDev': [0, 2],
        'TVP0': [0, 2],
        'Transition_StdDev': [0, 2],
        'Instrument_StdDev': [0, 2],
        'correlation_coefficient': [-1, 1],
        'm': [0, 2],
        'beta': [0, 2]
    }

    # Create the DataFrame to store the parameters of the inverse gamma prior distributions
    uniform_priors = pd.DataFrame(data3, index=index3)
    # Convert the DataFrame to a LaTeX table with a caption
    latex_table = uniform_priors.to_latex(caption="Metropolis-Hastings Sampler Prior", label="tab:MH Priors")

    # Print the LaTeX table
    #print(latex_table)


    parameter_count = 7



    # Set the number of draws
    burn_in = 1000 # Number of burn in draws
    num_samples = 2000 # Total number of samples
    total_draws = burn_in + num_samples # Total number of draws
    count = 0 # Counter
    sample_count = 0 # Counter for accepted samples
    accept_count = 0 # Counter for accepted draws


    # Data Matrices
    samples = np.zeros((parameter_count, num_samples))
    sample_paths = np.zeros((len(y), num_samples))
    sum_absolute_percent_deviation = np.zeros(parameter_count)


    # Variance covariance matrix of shock to random walk proposal
    R_matrix = np.diag(np.ones(parameter_count)) 

    # Change individual elements so that percent change in theta is close across the values.
    R_matrix[0, 0] = 1 # beta
    R_matrix[1, 1] = 1 # Measure_StdDev
    R_matrix[2, 2] = 1 # TVP0
    R_matrix[3, 3] = 1 # Transition_StdDev
    R_matrix[4, 4] = 1 # m
    R_matrix[5, 5] = 2 # Instrument_StdDev
    R_matrix[6, 6] = 1 # correlation_coefficient


    # Scale the matrix according to the acceptance rate
    R_matrix = R_matrix * 0.000000002 # gives 0.323


    # Initial parameter values for the sampler. All comparisons are made using the true value as the starting value.
    theta_g = true_theta
    
    # find the log prior 
    prior_indicator_g, log_prior_g = prior_probability(theta_g, uniform_priors)

    
    # calculate the likelihood of the data conditoned on the parameters using the extended particle filter
    pf_indicator_g, path_g, pY_theta_g = Extended_Particle_Filter_T(y, x1, x2, z, theta_g)

    if pf_indicator_g == False or prior_indicator_g == False:
        print("The extended particle filter did not converge with your initial values or issue with prior. Ending the function.")
        # print out which failed
        if pf_indicator_g == False:
            print("The particle filter did not converge with your initial values.")
        if prior_indicator_g == False:
            print("The prior probability is not finite with your initial values.")
        return False, None, None, None, None
    


    
    


    # Start MCMC
    for count in range(total_draws):
        theta_star = theta_g + multivariate_normal.rvs(mean=np.zeros(parameter_count), cov=R_matrix)

        # Find the percent deviation from theta_g to theta_star
        #percent_deviation = (theta_star - theta_g) / theta_g
        
        # Add the absolute value of the percent deviation to the running sum
        #sum_absolute_percent_deviation += np.abs(percent_deviation)

        # If the count is divisible by ten print a notification 
        #if (count+1) % 100 == 0:
            # Calculate and print the rolling average of the percent deviation
            #rolling_average = np.around(sum_absolute_percent_deviation / (count + 1), 8)
            #print(f'Rolling average of absolute percent deviation: {rolling_average}')
            # print the count
            #print("Count:", (count+1))

        # find the log prior 
        prior_indicator_star, log_prior_star = prior_probability(theta_star, uniform_priors)
        
        # check if the prior is finite
        if prior_indicator_star == True:

            # calculate the likelihood of the data conditoned on the parameters using the extended particle filter if the prior is finite
            pf_indicator_star, path_star, pY_theta_star = Extended_Particle_Filter_T(y, x1, x2, z, theta_star)

            # check if the particle filter converged
            if pf_indicator_star == True:

                # if the prior is finite and the particle filter converged, accept or reject the draw
                if np.log(np.random.uniform(0, 1)) < (pY_theta_star - pY_theta_g + log_prior_star - log_prior_g):
                    # accept
                    theta_g = theta_star
                    path_g = path_star
                    log_prior_g = log_prior_star
                    pY_theta_g = pY_theta_star
                    accept_count += 1

                if count > burn_in:
                    samples[:, sample_count] = theta_g
                    sample_paths[:, sample_count] = path_g
                    sample_count += 1
    
                # Print the acceptance rate every 100 draws
                #if (count+1) % 500 == 0:
                    #print("Acceptance rate:", accept_count / count) 
                    # also print the count
                    #print("Count:", count)

    return True, sample_paths, samples, uniform_priors, (accept_count / count)


# Metropolis Hasting Sampler for Extended Particle Filter N distributions
#------------------------------------------------------------
def Metropolis_Hasting_Sampler_N(y, x1, x2, z, true_theta):
    """
    sample model parameters using Metropolis Hasting Sampler
    

    Args:
        y: output vector of state space model
        x1: covariate on the TVP (potentially endogenous)
        x2: linear covariates
        z: instruments
        theta_in: all model parameters


    Outline of the function:
    --------
        1. Define prior parameters for the model
        2. Set the number of draws
        3. Initialize the parameter values for the Metropolis- Hasting Sampler
        4. Calculate the prior probability of the initial parameter values
        5. Start Sampling
        6. Accept/Reject Draws using the prior probability and particle filter
        7. Store the accepted draws
        8. Return the prior and posterior distributions

    Returns:
    --------
        A list of 
        (0) a boolean indicating if the filter converged.
        (1) the estimated paths of the TVP.
        (2) accepted samples of theta
        (3) uniform priors
        (4) acceptance rate
    """


    # Uniform Prior Distribution
    index3 = ['start', 'end']

    # For the variables of interest, define the bounds of the uniform prior distributions 
    data3 = {
        'Measure_StdDev': [0, 4],
        'TVP0': [0, 2],
        'Transition_StdDev': [0, 2],
        'Instrument_StdDev': [0, 4],
        'correlation_coefficient': [-1, 1],
        'm': [0, 2],
        'beta': [0, 2]
    }

    # Create the DataFrame to store the parameters of the inverse gamma prior distributions
    uniform_priors = pd.DataFrame(data3, index=index3)
    # Convert the DataFrame to a LaTeX table with a caption
    latex_table = uniform_priors.to_latex(caption="Metropolis-Hastings Sampler Prior", label="tab:MH Priors")

    # Print the LaTeX table
    #print(latex_table)


    parameter_count = 7



    # Set the number of draws
    burn_in = 1000 # Number of burn in draws
    num_samples = 2000 # Total number of samples
    total_draws = burn_in + num_samples # Total number of draws
    count = 0 # Counter
    sample_count = 0 # Counter for accepted samples
    accept_count = 0 # Counter for accepted draws


    # Data Matrices
    samples = np.zeros((parameter_count, num_samples))
    sample_paths = np.zeros((len(y), num_samples))
    sum_percent_deviation = np.zeros(parameter_count)


    # Variance covariance matrix of shock to random walk proposal
    R_matrix = np.diag(np.ones(parameter_count)) 

    # Change individual elements so that percent change in theta is close across the values.
    R_matrix[0, 0] = 1 # beta
    R_matrix[1, 1] = 1 # Measure_StdDev
    R_matrix[2, 2] = 1 # TVP0
    R_matrix[3, 3] = 1 # Transition_StdDev
    R_matrix[4, 4] = 1 # m
    R_matrix[5, 5] = 1 # Instrument_StdDev
    R_matrix[6, 6] = 1 # correlation_coefficient


    # Scale the matrix according to the acceptance rate
    R_matrix = R_matrix * 0.00000001 # 0.00000005 gives .143 # 0.0000001  # gives 0.116

    # Initial parameter values for the sampler. All comparisons are made using the true value as the starting value.
    theta_g = true_theta
    
    # find the log prior 
    prior_indicator_g, log_prior_g = prior_probability(theta_g, uniform_priors)

    
    # calculate the likelihood of the data conditoned on the parameters using the extended particle filter
    pf_indicator_g, path_g, pY_theta_g = Extended_Particle_Filter_N(y, x1, x2, z, theta_g)

    if pf_indicator_g == False or prior_indicator_g == False:
        print("The extended particle filter did not converge with your initial values or issue with prior. Ending the function.")
        # print out which failed
        if pf_indicator_g == False:
            print("The particle filter did not converge with your initial values.")
        if prior_indicator_g == False:
            print("The prior probability is not finite with your initial values.")
        return False, None, None, None, None
    


    
    


    # Start MCMC
    for count in range(total_draws):
        theta_star = theta_g + multivariate_normal.rvs(mean=np.zeros(parameter_count), cov=R_matrix)

        # Find the percent deviation from theta_g to theta_star
        #percent_deviation = (theta_star - theta_g) / theta_g
        
        # Add the absolute value of the percent deviation to the running sum
        #sum_absolute_percent_deviation += np.abs(percent_deviation)

        # If the count is divisible by ten print a notification 
        if (count+1) % 1000 == 0:
            # Calculate and print the rolling average of the percent deviation
            rolling_average = np.around(sum_percent_deviation / (count + 1), 3)
            #print(f'Rolling average of absolute percent deviation: {rolling_average}')
            # print the count
            #print("Count:", (count+1))

        # find the log prior 
        prior_indicator_star, log_prior_star = prior_probability(theta_star, uniform_priors)
        
        # check if the prior is finite
        if prior_indicator_star == True:

            # calculate the likelihood of the data conditoned on the parameters using the extended particle filter if the prior is finite
            pf_indicator_star, path_star, pY_theta_star = Extended_Particle_Filter_N(y, x1, x2, z, theta_star)

            # check if the particle filter converged
            if pf_indicator_star == True:

                # if the prior is finite and the particle filter converged, accept or reject the draw
                if np.log(np.random.uniform(0, 1)) < (pY_theta_star - pY_theta_g + log_prior_star - log_prior_g):
                    # accept
                    theta_g = theta_star
                    path_g = path_star
                    log_prior_g = log_prior_star
                    pY_theta_g = pY_theta_star
                    accept_count += 1

                if count > burn_in:
                    samples[:, sample_count] = theta_g
                    sample_paths[:, sample_count] = path_g
                    sample_count += 1
    
                # Print the acceptance rate every 100 draws
                #if (count+1) % 500 == 0:
                    #print("Acceptance rate:", accept_count / count) 
                    # also print the count
                    #print("Count:", count)

    return True, sample_paths, samples, uniform_priors, (accept_count / count)



# Test Metropolis Hasting Sampler 
#------------------------------------------------------------
def test_MH_sampler():
    """
    Test estimation quality of the Metropolis Hasting sampler that uses a particle filters for the likelihood
    by simulating multiple independent data sets and reviewing the average outcome. 

    Returns:
    --------
        The mean of the completed MH samplers: 
        (0) point estimates
        (1) standard deviations
        (2) acceptance rates
        (3) deviation from the true TVP
    """
    theta = [1, # Measure_StdDev
             1, # beta
             1, # TVP0
             1, # Transition_StdDev
             1, # Instrument_StdDev 
             0.7, # correlation_coefficient
             1] # m


    # Print the start time since the function takes so long to run
    print("Start time:", datetime.now().strftime('%H:%M:%S'))

    # Set the number of iterations
    iterations = 34

    # Create empty arrays to store the results
    samples_MH_T = np.empty((iterations,), dtype=object)
    samples_MH_N = np.empty((iterations,), dtype=object)
    acceptance_MH_T = np.empty((iterations,), dtype=object)
    acceptance_MH_N = np.empty((iterations,), dtype=object)
    collection_squared_error_T = np.empty((iterations,), dtype=object)
    collection_squared_error_N = np.empty((iterations,), dtype=object)

    # Run the Metropolis Hasting Sampler for the number of iterations
    i = 0

    while i < iterations:
        
        # For each iteration, simulate a new set of data
        My_Data = Generate_State_Space_Model_Data(theta) 
        
        # Define the dependent variable
        y = My_Data['y']
        x1 = My_Data['x1']

        # Define the independent variables
        x2 = My_Data['x2']
        z  = My_Data['z']

        # Define the true TVP
        TVP = My_Data['TVP']

        # MH sampler with instruments
        indicator_T, sample_paths_T, samples_T, uniform_priors_T, acceptance_T  =  Metropolis_Hasting_Sampler_T(y, x1, x2, z, theta)
        

        # For each iteration, run the Metropolis Hasting Sampler using the Extended Particle Filter
        # MH sampler without instruments
        indicator_N, sample_paths_N, samples_N, uniform_priors_N, acceptance_N =  Metropolis_Hasting_Sampler_N(y, x1, x2, z, theta)


        # if iteration is divisible by _ print the iteration number
        #if i % 10 == 0:
        print("\n")  
        print(f'Iteration {i+1}')
        print("Time after 1 iteration:", datetime.now().strftime('%H:%M:%S'))     
            

        # If both of the MH samplers produced samples, save data on those samples
        if indicator_T == True and indicator_N == True:

            # print data summary while testing
            print("\n")         
            print("Particle Filter T")
            print("Indicator:", indicator_T)
            print("Acceptance Rate:", acceptance_T)
            print("Median of samples:", np.round(np.median(samples_T, axis=1), decimals=3) )   

            # print data summary while testing
            print("\n")         
            print("Particle Filter N Indicator:", indicator_N)
            print("Acceptance Rate:", acceptance_N)
            print("Median of samples:", np.round(np.median(samples_N, axis=1), decimals=3) )

            # Find the mean over all the paths given by a single iteration for both types of filters,
            mean_sample_path_iteration_T = np.mean(sample_paths_T, axis=1)
            mean_sample_path_iteration_N = np.mean(sample_paths_N, axis=1)
            
            # Now that we have a single path for each iteration, we can find the squared error at each time step
            squared_error_at_each_time_step_N =  (mean_sample_path_iteration_N - TVP) ** 2
            collection_squared_error_N[i] = squared_error_at_each_time_step_N

            squared_error_at_each_time_step_T =  (mean_sample_path_iteration_T - TVP) ** 2
            collection_squared_error_T[i] = squared_error_at_each_time_step_T

            # Save the results at each iteration
            samples_MH_T[i] = np.median(samples_T, axis=1)
            acceptance_MH_T[i] = acceptance_T
            
            samples_MH_N[i] = np.median(samples_N, axis=1)
            acceptance_MH_N[i] = acceptance_N

            i += 1

    # find the mean of the samples
    sample_mean_N =  np.mean(samples_MH_N, axis=0) 
    sample_mean_T = np.mean(samples_MH_T, axis=0) 

    # find the mean of the acceptance rates
    acceptance_mean_N = np.mean(acceptance_MH_N)
    acceptance_mean_T = np.mean(acceptance_MH_T)

    # find the standard deviation of the samples
    sample_std_N = np.std(samples_N, axis=1)
    sample_std_T = np.std(samples_T, axis=1)

    # find the per period mean of the squared error
    mean_MSE_T = np.mean(collection_squared_error_T, axis=0)
    mean_MSE_N = np.mean(collection_squared_error_N, axis=0)

    return sample_mean_N, sample_mean_T, sample_std_N, sample_std_T, acceptance_mean_N, acceptance_mean_T, mean_MSE_T, mean_MSE_N


