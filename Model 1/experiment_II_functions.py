import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal
from datetime import datetime
import matplotlib.pyplot as plt

"""
TVP model with the following form:
Y = (1-TVP)/TVP * x1 + x2 * beta + measurement_error
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

# Particle Filter Function
#------------------------------------------------------------
def Particle_Filter(y, x1, x2, theta_in):
    """
    Estimate the path of a time varying parameter using a particle filter.

    Args:
        y: output vector of state space model
        x1: covariate on the TVP (potentially endogenous)
        x2: linear covariates
        theta_in: all model parameters

    
    
    Returns:
        (0) a boolean indicating if the filter converged. 
        (1) the estimated path of the TVP.
        (2) the log likelihood of the model.
    """


    # Initialize the parameters
    num_particles = 400
    Sample = len(y)
    
    # theta_in:
    Measure_StdDev = theta_in[0]
    beta = theta_in[1]
    TVP0 = theta_in[2]
    Transition_StdDev = theta_in[3]

   
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
        coefficient_on_x1 =    (1-proposed_TVP_vector)/proposed_TVP_vector # Apply coefficient equation to state variable
        prediction_vec = coefficient_on_x1 * x1[k] +    beta * x2[k]
        error_vec = y[k] - prediction_vec


        # Find the likelihood of the proposed states
        likelihood_over_particles = norm.pdf(error_vec, 0, Measure_StdDev)
        likelihood_over_particles[np.isnan(likelihood_over_particles)] = 0

        # If all the likelihoods are zero, return false
        if np.sum(likelihood_over_particles) == 0:
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


# Extended Particle Filter Function
#------------------------------------------------------------
def Extended_Particle_Filter(y, x1, x2, z, theta_in):
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
    num_particles = 400
    Sample = len(y)
    

    # theta_in: 
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
        coefficient_on_x1 =    (1-proposed_TVP_vector)/proposed_TVP_vector # Apply coefficient equation to state variable
        prediction_vec = coefficient_on_x1 * x1[k] +    beta * x2[k]
        error_vec = y[k] - prediction_vec


        # Find the error of the instrument equation
        endogenous_prediction = m * z[k]
        instrument_error = x1[k] - endogenous_prediction


        # Find the likelihood of the proposed states using a joint likelihood method
        likelihood_over_particles = joint_likelihood(error_vec, instrument_error, mu, Sigma)
        likelihood_over_particles[np.isnan(likelihood_over_particles)] = 0


        # If all the likelihoods are zero, return false
        if np.sum(likelihood_over_particles) == 0:
            #print('All likelihoods are zero')
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


# Joint Likelihood Function
#------------------------------------------------------------
def joint_likelihood(error_vec, instrument_error_t, mu, Sigma):
    particle_count = len(error_vec)
    particle_instrument_error_vec = np.column_stack((error_vec, np.full(particle_count, instrument_error_t)))
    likelihood_over_particles = multivariate_normal(mean=mu, cov=Sigma).pdf(particle_instrument_error_vec)
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
    
    # theta
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
    error_mean_vector = [0, 0]


    # Generate the error terms for the measurement and the endogenous variable
    # The generated error terms follow a multivariate normal distribution with the defined mean vector and covariance matrix
    measurement_error, endogenous_variable_error = np.random.multivariate_normal(error_mean_vector, error_covariance_matrix, num_samples).T


    # Create (potentially) Endogenous variable
    x1 = m * z + endogenous_variable_error


    # Apply coefficient equation to state variable
    Endogenous_coefficient = (1-TVP)/TVP 


    # Create measurement variable
    y = Endogenous_coefficient * x1 + beta * x2 + measurement_error


    # Create a DataFrame from the generated data
    data = np.column_stack([y, x1, x2, TVP, z, measurement_error, endogenous_variable_error])


    # Define the column names and create the DataFrame in one step
    df = pd.DataFrame(data, columns=['y', 'x1', 'x2', 'TVP',  'z', 'measurement_error', 'endogenous_variable_error'])


    return df

# Prior Probability for Extended Particle Filter
#------------------------------------------------------------
def prior_probability_EPF(theta_in, uniform_priors):
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



# Prior Probability for Particle Filter
#------------------------------------------------------------
def prior_probability_PF(theta_in, uniform_priors):
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



    # Calculate the product of prior probabilities
    prior_product = ( prob_beta * prob_TVP0 * prob_measurement_StdDev * prob_TVP_StdDev )

    if prior_product == 0:
        zero_values = []
        if prob_TVP0 == 0:
            zero_values.append("prob_TVP0")
        if prob_measurement_StdDev == 0:
            zero_values.append("prob_measurement_StdDev")
        if prob_TVP_StdDev == 0:
            zero_values.append("prob_TVP_StdDev")
        if prob_beta == 0:
            zero_values.append("prob_beta")
        
        #print("Product of prior probabilities is zero.")
        #print("Zero values:", zero_values)
        return False, None

    # Calculate the log of the prior probability
    log_prior = np.log(prior_product)

    return True, log_prior



def Metropolis_Hasting_Sampler_EPF(y, x1, x2, z, true_theta):
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
        A list of:
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
        'TVP0': [3, 5],
        'Transition_StdDev': [0, 0.2],
        'beta': [0, 2],
        'Instrument_StdDev': [0, 2],
        'correlation_coefficient': [-1, 1],
        'm': [0, 2]   
    }

    # Create the DataFrame to store the parameters of the inverse gamma prior distributions
    uniform_priors = pd.DataFrame(data3, index=index3)
    # Convert the DataFrame to a LaTeX table with a caption
    latex_table = uniform_priors.to_latex(caption="Metropolis-Hastings Sampler Prior", label="tab:MH Priors")

    # Print the LaTeX table
    #print(latex_table)


    parameter_count = 7



    # Draw Information
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
    R_matrix = np.diag(np.ones(parameter_count)) 


    # Variance covariance matrix of shock to random walk proposal
    # Change individual elements so that percent change in theta is close across the values.
    R_matrix[0, 0] = 1 # beta
    R_matrix[1, 1] = 1 # Measure_StdDev
    R_matrix[2, 2] = 6 # TVP0
    R_matrix[3, 3] = .04 # Transition_StdDev
    R_matrix[4, 4] = 1 # M
    R_matrix[5, 5] = 0.8 # Instrument_StdDev
    R_matrix[6, 6] = 1 # correlation_coefficient


    # Scale the matrix according to the acceptance rate
    R_matrix = R_matrix *0.0006 #0.0005 #0.00034



    # Initial parameter values for the sampler. All comparisons are made using the true value as the starting value.
    theta_g = true_theta
    
    # find the log prior 
    prior_indicator_g, log_prior_g = prior_probability_EPF(theta_g, uniform_priors)

    
    # calculate the likelihood of the data conditoned on the parameters using the extended particle filter
    pf_indicator_g, path_g, pY_theta_g = Extended_Particle_Filter(y, x1, x2, z, theta_g)

    if pf_indicator_g == False or prior_indicator_g == False:
        # print out which failed
        if pf_indicator_g == False:
            print("The particle filter did not converge with your initial values.")
        if prior_indicator_g == False:
            print("The prior probability is not finite with your initial values.")
        return False, None, None, None, None
    


    # Start MCMC
    for count in range(total_draws):
        theta_star = theta_g + multivariate_normal.rvs(mean=np.zeros(parameter_count), cov=R_matrix)

        """# Find the percent deviation from theta_g to theta_star
        percent_deviation = (theta_star - theta_g) / theta_g
        
        # Add the absolute value of the percent deviation to the running sum
        sum_percent_deviation += np.abs(percent_deviation)

        # If the count is divisible by __ print rolling average of absolute percent deviation
        if (count+1) % 3000 == 0:
            # print the count
            print("Count:", (count+1))
            # Calculate and print the rolling average of the percent deviation
            rolling_average = np.around(sum_percent_deviation / (count + 1), 3)
            print(f'Rolling average of absolute percent deviation: {rolling_average}')"""
            


        # find the log prior 
        prior_indicator_star, log_prior_star = prior_probability_EPF(theta_star, uniform_priors)
        
        # check if the prior is finite
        if prior_indicator_star == True:
            # calculate the likelihood of the data conditoned on the parameters using the extended particle filter if the prior is finite
            pf_indicator_star, path_star, pY_theta_star = Extended_Particle_Filter(y, x1, x2, z, theta_star)

            # check if the particle filter converged
            if pf_indicator_star == True:

                # if the particle filter converged and the prior is finite, accept or reject the draw
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
        """if (count+1) % 3000 == 0:
            print("Acceptance rate:", accept_count / count) """


    return True, sample_paths, samples, uniform_priors, (accept_count / count)


# Metropolis Hasting Sampler for Particle Filter
#------------------------------------------------------------
def Metropolis_Hasting_Sampler_PF(y, x1, x2, true_theta):
    """
    sample model parameters using Metropolis Hasting Sampler
    

    Arguments:
    -----
        Y: output vector of state space model
        x1: covariate on the TVP (potentially endogenous)
        X2: linear covariates


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
        'TVP0': [3, 5],
        'Transition_StdDev': [0, 0.2],
        'beta': [0, 2]
    }

    # Create the DataFrame to store the parameters of the inverse gamma prior distributions
    uniform_priors = pd.DataFrame(data3, index=index3)
    #print(uniform_priors)


    parameter_count = 4



    # Draw Information
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
    R_matrix = np.diag(np.ones(parameter_count)) 


    # Variance covariance matrix of shock to random walk proposal
    # Change individual elements so that percent change in theta is close across the values.
    R_matrix[0, 0] = 1 # beta
    R_matrix[1, 1] = 1 # Measure_StdDev
    R_matrix[2, 2] = 5.5 # TVP0
    R_matrix[3, 3] = .05 # Transition_StdDev


    # Scale the matrix according to the acceptance rate
    R_matrix = R_matrix *0.00012 # 0.00009 # 0.00008 # 0.00007 #0.00005 # 0.000034


    # Initial parameter values for the sampler. All comparisons are made using the true value as the starting value.
    theta_g = true_theta[:4]
    
    # find the log prior 
    prior_indicator_g, log_prior_g = prior_probability_PF(theta_g, uniform_priors)

    
    # calculate the likelihood of the data conditoned on the parameters using the extended particle filter
    pf_indicator_g, path_g, pY_theta_g = Particle_Filter(y, x1, x2, theta_g)

    if pf_indicator_g == False or prior_indicator_g == False:
        print("The particle filter did not converge with your initial values or issue with prior. Ending the function.")
        return False, None, None, None, None
    


    # Start MCMC
    for count in range(total_draws):
        theta_star = theta_g + multivariate_normal.rvs(mean=np.zeros(parameter_count), cov=R_matrix)

        """# Find the percent deviation from theta_g to theta_star
        percent_deviation = (theta_star - theta_g) / theta_g
        
        # Add the absolute value of the percent deviation to the running sum
        sum_percent_deviation += np.abs(percent_deviation)

        # If the count is divisible by __ print rolling average of absolute percent deviation
        if (count+1) % 3000 == 0:
            #print the count
            print("Count:", (count+1))
            # Calculate and print the rolling average of the percent deviation
            rolling_average = np.around(sum_percent_deviation / (count + 1), 3)
            print(f'Rolling average of absolute percent deviation: {rolling_average}')
            """


        # find the log prior 
        prior_indicator_star, log_prior_star = prior_probability_PF(theta_star, uniform_priors)
        
        # check if the prior is finite
        if prior_indicator_star == True:
            # calculate the likelihood of the data conditoned on the parameters using the extended particle filter
            pf_indicator_star, path_star, pY_theta_star = Particle_Filter(y, x1, x2, theta_star)

            # check if the particle filter converged
            if pf_indicator_star == True:
                # print("The particle filter did converge with your new values: theta_star. and prior probability is finite")
                # increment counter
                count += 1
                #print(theta_star[3])

                # now decide to replace theta_g with theta_star, if we reject theta_star, theta_g stays the same. 
                if np.log(np.random.uniform(0, 1)) < (pY_theta_star - pY_theta_g + log_prior_star - log_prior_g):
                    # accept
                    theta_g = theta_star
                    path_g = path_star
                    log_prior_g = log_prior_star
                    pY_theta_g = pY_theta_star
                    accept_count += 1

        # If we have finished our burn in count, store the accepted draws
        if count > burn_in:
            samples[:, sample_count] = theta_g
            sample_paths[:, sample_count] = path_g
            sample_count += 1

        """# Print the acceptance rate every 100 draws
        if (count+1) % 3000 == 0:
            print("Acceptance rate:", accept_count / count) 
            # also print the count
            #print("Count:", count)"""


    # return sample paths, samples, proposals, and prior distributions
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
        (4) MSE of the deviation from the true TVP
    """
    
    theta = [1, # Measure_StdDev
             1, # beta
             4, # TVP0
             .1, # Transition_StdDev
             1, # Instrument_StdDev
             0, # correlation_coefficient
             1] # m
    

    # Print the start time since the function takes so long to run
    print("Start time:", datetime.now().strftime('%H:%M:%S'))

    # Set the number of iterations
    iterations = 34

    # Create empty arrays to store the results
    MH_sample_medians_PF = np.empty((iterations,), dtype=object)
    MH_sample_medians_EPF = np.empty((iterations,), dtype=object)
    MH_sample_std_EPF = np.empty((iterations,), dtype=object)
    MH_sample_std_PF = np.empty((iterations,), dtype=object)
    MH_acceptance_rates_PF = np.empty((iterations,), dtype=object)
    MH_acceptance_rates_EPF = np.empty((iterations,), dtype=object)
    deviation_paths_PF = np.empty((iterations,), dtype=object)
    deviation_paths_EPF = np.empty((iterations,), dtype=object)

    # Run the Metropolis Hasting Sampler for the number of iterations
    i = 0
    while i < iterations:
        My_Data = Generate_State_Space_Model_Data(theta)

        # Define the dependent variable
        y = My_Data['y']
        x1 = My_Data['x1']

        # Define the independent variables
        x2 = My_Data['x2']
        z  = My_Data['z']

        # Define the true TVP
        TVP = My_Data['TVP']


        # Run the Metropolis Hasting Sampler using the Extended Particle Filter
        indicator_EPF, sample_paths_EPF, samples_EPF, uniform_priors_EPF, acceptance_EPF =  Metropolis_Hasting_Sampler_EPF(y, x1, x2, z, theta)

        # print indicator_EPF
        print(f'EPF Indicator: {indicator_EPF}, Time: {datetime.now().strftime("%H:%M:%S")}, Iteration: {i+1}.')

        # Run the Metropolis Hasting Sampler using the Particle Filter
        indicator_PF, sample_paths_PF, samples_PF, uniform_priors_EPF, acceptance_PF  =  Metropolis_Hasting_Sampler_PF(y, x1, x2, theta)

        # print indicator and time
        print(f'PF Indicator: {indicator_PF}, Time: {datetime.now().strftime("%H:%M:%S")}, Iteration: {i+1}.')
        

            

        # If both of the filters converged, save the difference between the estimated TVP and the true TVP
        if indicator_PF == True and indicator_EPF == True:
            print(f'Iteration {i+1} completed at {datetime.now().strftime("%H:%M:%S")}.')
            
            

            # print data from the particle filter
            print("\n")         # print a line break
            print("Particle Filter")
            print("Indicator:", indicator_PF)
            print("Acceptance Rate:", acceptance_PF)
            median_PF = np.median(samples_PF, axis=1)
            rounded_median_PF = np.round(median_PF, decimals=3)
            print("Median of samples:", rounded_median_PF)
            print("\n")         # print a line break
            
            # print data from the extended particle filter
            print("\n")         # print a line break
            print("Extended Particle Filter")
            print("Indicator:", indicator_EPF)
            print("Acceptance Rate:", acceptance_EPF)
            median_EPF = np.median(samples_EPF, axis=1)
            rounded_median_EPF = np.round(median_EPF, decimals=3)
            print("Median of samples:", rounded_median_EPF)
            print("\n")         # print a line break

            # Find the mean over all the paths (you could find confidence intervals here too)
            sample_path_PF = np.mean(sample_paths_PF, axis=1)
            sample_path_EPF = np.mean(sample_paths_EPF, axis=1)
        

            MH_sample_medians_PF[i] = np.median(samples_PF, axis=1)
            MH_sample_medians_EPF[i] = np.median(samples_EPF, axis=1)

            MH_sample_std_EPF[i] = np.std(samples_EPF, axis=1)
            MH_sample_std_PF[i] = np.std(samples_PF, axis=1)

            MH_acceptance_rates_PF[i] = acceptance_PF
            MH_acceptance_rates_EPF[i] = acceptance_EPF
            

            # Also save the difference between the estimated TVP and the true TVP
            difference_EPF =  sample_path_EPF - TVP 
            deviation_paths_EPF[i] = difference_EPF
            difference_PF = sample_path_PF - TVP
            deviation_paths_PF[i]  = difference_PF
            i += 1
        else:
            print("One of the filters did not converge. Repeat iteration.")

    # find the mean across iterations
    mean_median_EPF =  np.mean(MH_sample_medians_EPF, axis=0) 
    mean_median_PF = np.mean(MH_sample_medians_PF, axis=0) 

    mean_std_EPF = np.mean(MH_sample_std_EPF, axis=0)
    mean_std_PF = np.mean(MH_sample_std_PF, axis=0)

    mean_acceptance_MH_EPF = np.mean(MH_acceptance_rates_EPF, axis=0)
    mean_acceptance_MH_PF = np.mean(MH_acceptance_rates_PF, axis=0)

    # find the average deviation for paths
    mean_deviation_PF = np.mean(deviation_paths_PF, axis=0)
    mean_deviation_EPF = np.mean(deviation_paths_EPF, axis=0)
    
    return mean_median_EPF, mean_median_PF,mean_std_EPF, mean_std_PF, mean_acceptance_MH_EPF, mean_acceptance_MH_PF, mean_deviation_PF, mean_deviation_EPF 



"""# Run the sampler and get the results
result = test_MH_sampler()
# save the results
import pickle

with open('/Users/modelt/Documents/Research/Extended Particle FIlter/Results of Experiments/model10.npy', 'wb') as f:
    pickle.dump(result, f)


import pickle
# Load the results
with open('/Users/modelt/Documents/Research/Extended Particle FIlter/Results of Experiments/model10.npy', 'rb') as f:
    result0 = pickle.load(f)

result07 = result0

# Handle result0
sample_mean_EPF_0 = result0[0] # np.mean(result0[0], axis=0)
sample_mean_PF_0 = result0[1] # np.mean(result0[1], axis=0)
sample_std_EPF_0 = result0[2] # np.mean(result0[2], axis=0)
sample_std_PF_0 = result0[3] #np.mean(result0[3], axis=0)
acceptance_mean_EPF_0 = result0[4]# np.mean(result0[4], axis=0)
acceptance_mean_PF_0 = result0[5]# np.mean(result0[5], axis=0)
mean_deviation_PF_0 = result0[6]# np.mean(result0[6], axis=0)
mean_deviation_EPF_0 = result0[7] #  np.mean(result0[7], axis=0)

# Handle result07
sample_mean_EPF_07 =result07[0] #  np.mean(result07[0], axis=0)
sample_mean_PF_07 = result07[1] # np.mean(result07[1], axis=0)
sample_std_EPF_07 = result07[2] # np.mean(result07[2], axis=0)
sample_std_PF_07 = result07[3] # np.mean(result07[3], axis=0)
acceptance_mean_EPF_07 = result07[4] # np.mean(result07[4], axis=0)
acceptance_mean_PF_07 = result07[5] #np.mean(result07[5], axis=0)
mean_deviation_PF_07 = result07[6] #np.mean(result07[6], axis=0)
mean_deviation_EPF_07 = result07[7] #np.mean(result07[7], axis=0)

# Create Table 1
variable_names = ['$sigma_nu$ = 1', '$beta$ =1', '$lambda_0$ =4', '$sigma_omega$ =0.1', '$sigma_xi$ =1', '$rho$ ', 'm=1',   'Acceptance Rate']

# Create a DataFrame with the results for result0
df0 = pd.DataFrame({
    'Extended Particle Filter': [f"{sample_mean_EPF_0[i]:.3f} ({sample_std_EPF_0[i]:.3f})" for i in range(7)] + [f"{acceptance_mean_EPF_0:.3f}"],
    'Particle Filter': [f"{sample_mean_PF_0[i]:.3f} ({sample_std_PF_0[i]:.3f})" if i < len(sample_mean_PF_0) else '-' for i in range(7)] + [f"{acceptance_mean_PF_0:.3f}"]
}, index = variable_names)

# Create a DataFrame with the results for result07
df07 = pd.DataFrame({
    'Extended Particle Filter': [f"{sample_mean_EPF_07[i]:.3f} ({sample_std_EPF_07[i]:.3f})" for i in range(7)] + [f"{acceptance_mean_EPF_07:.3f}"],
    'Particle Filter': [f"{sample_mean_PF_07[i]:.3f} ({sample_std_PF_07[i]:.3f})" if i < len(sample_mean_PF_07) else '-' for i in range(7)] + [f"{acceptance_mean_PF_07:.3f}"]
}, index = variable_names)

# Convert df0 to LaTeX
latex0 = df0.to_latex(caption="Model I: Metropolis-Hastings Sampler Results for 0")
print(latex0)

# Convert df07 to LaTeX
latex07 = df07.to_latex(caption="Model I: Metropolis-Hastings Sampler Results for 07")
print(latex07)


# Create Figure 2
plt.figure()

# Create the first subplot for correlation coefficient of 0
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
plt.plot(mean_deviation_PF_0, label='Particle Filter')
plt.plot(mean_deviation_EPF_0, label='Extended Particle Filter')
plt.title(f'Correlation Coefficient: 0')
plt.legend(loc='upper left')



# Create the second subplot for correlation coefficient of 0.7
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
plt.plot(mean_deviation_PF_07, label='Particle Filter')
plt.plot(mean_deviation_EPF_07, label='Extended Particle Filter')
plt.title(f'Correlation Coefficient: 0.7')
plt.legend(loc='upper left')


# Show the figure with both subplots
plt.tight_layout()
plt.show()"""