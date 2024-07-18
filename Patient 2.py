#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Given data
gfr_values = np.array([7.76, 35.75, 48.12, 44.19, 67.53, 61.32, 57.59, 56.82, 62.95, 50.54, 48.21, 70.41, 52.10, 56.04])
times = np.array([9, 42, 120, 183, 309, 828, 1059, 1455, 1717, 2152, 2465, 2848, 3249, 3523])
dt = 0.1
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
Q = np.array([[576, 0.0],[0.0, 576]])
R = 243.55
m_0 = np.array([7.76, 0.848485])
P_0 = np.array([[1, 0.],
                [0., 1]])

# Function to perform Extended Kalman Filter
def extended_kalman_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, observations):
    n = m_0.shape[-1]
    steps = observations.shape[0]
    
    ekf_states = np.empty((steps, n))
    ekf_P = np.empty((steps, n, n))
    
    m = m_0.copy()
    P = P_0.copy()
    
    for i in range(steps):
        y = observations[i]
        
        # Jacobian of the dynamic model function
        F = np.array([[1., dt], 
                      [-dt * (3 * a * m[0]**2 + 2 * b * m[0] + gamma), 1 - lmbda * dt]])
        
        # Predicted state distribution
        m = np.array([m[0] + dt * m[1],
                      m[1] - lmbda * dt * m[1] - (a * m[0]**3 + b * m[0]**2 + gamma * m[0] + delta) * dt])
        #print('m',m)
        P = F @ P @ F.T + Q
        
        # Predicted observation
        h = m[0] 
        H = np.array([[1., 0.]])
        S = H.dot(P.dot(H.T)) + R
        
        # Gain
        K = linalg.solve(S, H @ P, assume_a="pos").T 
        m = m + K @ np.atleast_1d(y - h)
        P = P - K @ S @ K.T
        
        ekf_states[i] = m
        ekf_P[i] = P
    
    return ekf_states, ekf_P

# Run the Extended Kalman Filter using real GFR data
ekf_states, ekf_P = extended_kalman_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, gfr_values)
#print('ekf_states',ekf_states)

# Plotting
plt.figure(figsize=(10, 6))

plt.scatter(times, gfr_values, label='Measurements', marker='o', color='black', s=10)  # Use times as x-values
plt.plot(times, ekf_states[:, 0], label='Estimated GFR (EKF)', color='green', linestyle='-', linewidth=1)

plt.xlabel('Time [days]',  fontsize=20)
plt.ylabel('GFR [ml/min]', fontsize=20 )
plt.title('Estimated GFR (EKF) for Patient 2', fontsize=20 )
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE
def rmse(x, y):
    return np.sqrt(np.mean(np.sum(np.square(x - y), -1)))

rmse_ekf = rmse(gfr_values, ekf_states[:, 0])
print(f"EKF RMSE: {rmse_ekf}")


# In[ ]:





# In[14]:


import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Given data
observations = np.array([7.76, 35.75, 48.12, 44.19, 67.53, 61.32, 57.59, 56.82, 62.95, 50.54, 48.21, 70.41, 52.10, 56.04])
times = np.array([9, 42, 120, 183, 309, 828, 1059, 1455, 1717, 2152, 2465, 2848, 3249, 3523])
dt = 0.1
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
Q = np.array([[576, 0.0], [0.0, 576]])
R = 243.55
m_0 = np.array([7.76,0.848485])
P_0 = np.array([[1, 0.],
                [0., 1]])


def ukf_weights(alpha, beta, lamda, n):
    wm = np.full(2 * n + 1, 1 / (2 * (n + lamda)))
    wc = wm[:]
    
    wm[0] = lamda / (n + lamda)
    wc[0] = lamda / (n + lamda) + (1 - alpha ** 2 + beta)
    return wm, wc
def unscented_kalman_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, observations):
    n = m_0.shape[-1]
    

    # UKF parameters 
    alpha = 1
    beta = 0
    kappa = 3 - n
    lamda = alpha ** 2 * (n + kappa) - n  
    wm, wc = ukf_weights(alpha, beta, lamda, n) 

    # Initialize
    steps = observations.shape[0]
    
    ukf_m = np.empty((steps, n))
    ukf_P = np.empty((steps, n, n))
    
    m = m_0[:]
    P = P_0[:]
    
    zeros = np.zeros((n, 1))
    
    for i in range(steps):
        y = observations[i]
        
        # Compute the sigma points for the dynamics
        L = np.linalg.cholesky(P) * np.sqrt(n + lamda)
        sigma_points = np.concatenate([zeros, L, -L], axis=1) + m.reshape(-1, 1)
        
        # Progagate through the dynamics
        sigma_points[0, :], sigma_points[1, :] = sigma_points[0, :] + dt * sigma_points[1, :], sigma_points[1, :] - lmbda * dt * sigma_points[0, :] -dt*(a*(sigma_points[0, :])**3+b*(sigma_points[0, :])**2)+ gamma*(sigma_points[0, :])+ delta    
        
        # Predicted state distribution
        m = np.dot(sigma_points, wm)
        P = np.dot(wc.reshape(1, -1) * (sigma_points - m.reshape(-1, 1)), (sigma_points - m.reshape(-1, 1)).T) + Q

        # Compute the sigma points for the observation
        L = np.linalg.cholesky(P) * np.sqrt(n + lamda)
        sigma_points = np.concatenate([zeros, L, -L], axis=1) + m.reshape(-1, 1)
        
        # Progagate through the measurement model
        sigma_observations = sigma_points[0, :]
        
        # sigma points measurement mean and covariance
        predicted_mu = np.dot(sigma_observations, wm)
        predicted_cov = np.dot(wc * (sigma_observations - predicted_mu), sigma_observations - predicted_mu) + R
        cross_cov = np.dot(sigma_points - m.reshape(-1, 1), wc * (sigma_observations - predicted_mu))
        
        
        # Gain
        K = cross_cov / predicted_cov # Works only when predicted_cov is scalar
        m = m + K * (y - predicted_mu)
        P = P - predicted_cov * K.reshape(-1, 1) * K
        
        ukf_m[i] = m
        ukf_P[i] = P
    return ukf_m, ukf_P

ukf_m, ukf_P = unscented_kalman_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, observations)


# Plotting
plt.figure(figsize=(10, 6))
#plt.scatter(times, gfr_values, label='Measurements', marker='o', color='black', s=10) 

plt.scatter(times, observations, label='True GFR', color='black', marker='o',s=10)
plt.plot(times, ukf_m[:, 0], label='Estimated GFR (UKF)', color='red', linestyle='-', linewidth=1)
plt.xlabel('Time[days]', fontsize=20 )
plt.ylabel('GFR[ml/min]',  fontsize=20)
plt.title('True GFR vs Estimated GFR (UKF) for Patient 2',  fontsize=20)
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE
def rmse(x, y):
   
    return np.sqrt(np.mean(np.sum(np.square(x - y), -1)))

rmse_ukf = rmse(observations, ukf_m[:, 0])
print(f"UKF RMSE: {rmse_ukf}")




# In[ ]:





# In[15]:


import numpy as np
from scipy import linalg
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Given data
observations = np.array([7.76, 35.75, 48.12, 44.19, 67.53, 61.32, 57.59, 56.82, 62.95, 50.54, 48.21, 70.41, 52.10, 56.04])
times = np.array([9, 42, 120, 183, 309, 828, 1059, 1455, 1717, 2152, 2465, 2848, 3249, 3523])
dt = 0.1
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
Q = np.array([[576, 0.0], [0.0, 576]])
R = 243.55
m_0 = np.array([7.76,0.848485])
P_0 = np.array([[1, 0.],
                [0., 1]])

def ukf_weights(alpha, beta, lamda, n):
    wm = np.full(2 * n + 1, 1 / (2 * (n + lamda)))
    wc = wm[:]
    
    wm[0] = lamda / (n + lamda)
    wc[0] = lamda / (n + lamda) + (1 - alpha ** 2 + beta)
    return wm, wc

def unscented_kalman_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, observations):
    n = m_0.shape[-1]

    # UKF parameters
    alpha = 1
    beta = 0
    kappa = 3 - n
    lamda = alpha ** 2 * (n + kappa) - n
    wm, wc = ukf_weights(alpha, beta, lamda, n) 

    # Initialize
    steps = observations.shape[0]
    
    ukf_m = np.empty((steps, n))
    ukf_P = np.empty((steps, n, n))
    
    m = m_0[:]
    P = P_0[:]
    
    zeros = np.zeros((n, 1))
    
    for i in range(steps):
        y = observations[i]
        
        # Compute the sigma points for the dynamics
        L = np.linalg.cholesky(P) * np.sqrt(n + lamda)
        sigma_points = np.concatenate([zeros, L, -L], axis=1) + m.reshape(-1, 1)
        
        # Propagate through the dynamics
        sigma_points[0, :], sigma_points[1, :] = sigma_points[0, :] + dt * sigma_points[1, :], sigma_points[1, :] - lmbda * dt * sigma_points[0, :] -dt*(a*(sigma_points[0, :])**3+b*(sigma_points[0, :])**2)+ gamma*(sigma_points[0, :])+ delta    
        
        # Predicted state distribution
        m = np.dot(sigma_points, wm)
        P = np.dot(wc.reshape(1, -1) * (sigma_points - m.reshape(-1, 1)), (sigma_points - m.reshape(-1, 1)).T) + Q

        # Compute the sigma points for the observation
        L = np.linalg.cholesky(P) * np.sqrt(n + lamda)
        sigma_points = np.concatenate([zeros, L, -L], axis=1) + m.reshape(-1, 1)
        
        # Propagate through the measurement model
        sigma_observations = sigma_points[0, :]
        
        # Sigma points measurement mean and covariance
        predicted_mu = np.dot(sigma_observations, wm)
        predicted_cov = np.dot(wc * (sigma_observations - predicted_mu), sigma_observations - predicted_mu) + R
        cross_cov = np.dot(sigma_points - m.reshape(-1, 1), wc * (sigma_observations - predicted_mu))
        
        # Kalman gain
        K = cross_cov / predicted_cov 
        m = m + K * (y - predicted_mu)
        P = P - predicted_cov * K.reshape(-1, 1) * K
        
        ukf_m[i] = m
        ukf_P[i] = P
    
    return ukf_m, ukf_P

# Perform cubic spline interpolation
cs = CubicSpline(times, observations)

# Calculate the number of additional points needed for interpolation
additional_points = 988

# Interpolate the observations at desired times with additional points
interpolated_times = np.linspace(times[0], times[-1], num=len(observations) + additional_points)
interpolated_observations = cs(interpolated_times)

# Apply UKF with interpolated observations
ukf_m, ukf_P = unscented_kalman_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, interpolated_observations)
print('ukf_m', ukf_m[:5])
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(interpolated_times, ukf_m[:, 0], label=' UKF estimate for interpolated GFR', color='red', linestyle='--', linewidth=2)
plt.plot(interpolated_times, interpolated_observations, label='Interpolated GFR being estimated', color='green', linestyle='-', linewidth=1)

plt.xlabel('Time [days]',  fontsize=20)
plt.ylabel('GFR [ml/min]',  fontsize=20)
plt.title('True GFR vs Estimated GFR (UKF) for Patient 2', fontsize=20)
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE
def rmse(x, y):
    return np.sqrt(np.mean(np.sum(np.square(x - y), -1)))

rmse_ukf = rmse(interpolated_observations, ukf_m[:, 0])
print(f"UKF RMSE: {rmse_ukf}")

# Length of original GFR values
original_num_observations = len(observations)

# Length of interpolated GFR values
interpolated_num_observations = len(interpolated_observations)

# Increase in the number of observations
increase_in_observations = interpolated_num_observations - original_num_observations

print(f"Increased observations after interpolation: {increase_in_observations}")


# In[ ]:





# In[9]:


import numpy as np
import matplotlib.pyplot as plt
# Seed the random number generator for reproducibility
np.random.seed(5)

# Given data
gfr_values = np.array([7.76, 35.75, 48.12, 44.19, 67.53, 61.32, 57.59, 56.82, 62.95, 50.54, 48.21, 70.41, 52.10, 56.04])
times = np.array([9, 42, 120, 183, 309, 828, 1059, 1455, 1717, 2152, 2465, 2848, 3249, 3523])
dt = 0.1
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
Q = np.array([[576, 0.0], [0.0, 576]])
R = 243.55
num_particles = 10000
m_0 = np.array([7.76,0.848485])
P_0 = np.array([[1, 0.], [0., 1]])

# Particle Filter function with systematic resampling
def particle_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, observations, num_particles):
    n = m_0.shape[-1]
    steps = observations.shape[0]

    particles = np.zeros((num_particles, n))
    weights = np.ones(num_particles) / num_particles

    pf_states = np.empty((steps, n))

    for i in range(steps):
        y = observations[i]

        # Define the state transition matrix F
        F = np.array([[1, dt],
                      [-dt * (3*a*m_0[0]**2 + 2*b*m_0[0] + gamma), 1 - lmbda * dt]])

        # Prediction
        for j in range(num_particles):
            # Predict state based on the state transition model with process noise
            particles[j] = np.dot(F, particles[j]) + np.random.multivariate_normal(np.zeros(n), Q)
            

        # Update (Correction)
        for j in range(num_particles):
            # Calculate likelihood (measurement model)
            measurement_likelihood = 1 / np.sqrt(2 * np.pi * R) * np.exp(-0.5 * (y - particles[j, 0])**2 / R)
            # Update particle weight
            weights[j] *= measurement_likelihood

        # Normalize weights
        weights /= np.sum(weights)

        # Systematic Resampling
        cumulative_weights = np.cumsum(weights)
        u = (np.random.rand() + np.arange(num_particles)) / num_particles
        indices = np.zeros(num_particles, dtype=int)
        j = 0
        for k in range(num_particles):
            while u[k] > cumulative_weights[j]:
                j += 1
            indices[k] = j

        particles = particles[indices]
        weights.fill(1.0 / num_particles)

        # Estimate state based on weighted average of particles
        estimated_state = np.average(particles, weights=weights, axis=0)
        pf_states[i] = estimated_state

    return pf_states

# Run the Particle Filter with systematic resampling using real GFR data
pf_states = particle_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, gfr_values, num_particles)


# Plotting
plt.figure(figsize=(10, 6))

plt.scatter(times, gfr_values, label='Measurements', color='black', marker='o', s=10)
plt.plot(times, pf_states[:, 0], label='Estimated GFR (PF)', color='blue', linestyle='-', linewidth=1)
plt.xlabel('Time[days]', fontsize=20)
plt.ylabel('GFR[ml/min]', fontsize=20 )
plt.title('True GFR vs Estimated GFR (PF) for Patient 2 with Systematic Resampling',  fontsize=15)
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE
def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))

rmse_pf = rmse(gfr_values, pf_states[:, 0])
print(f"PF RMSE with Systematic Resampling: {rmse_pf}")


# In[ ]:





# In[ ]:





# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Seed the random number generator for reproducibility
np.random.seed(5)

# New data
gfr_values = np.array([7.76, 35.75, 48.12, 44.19, 67.53, 61.32, 57.59, 56.82, 62.95, 50.54, 48.21, 70.41, 52.10, 56.04])
times = np.array([9, 42, 120, 183, 309, 828, 1059, 1455, 1717, 2152, 2465, 2848, 3249, 3523])
dt = 0.1
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
Q = np.array([[576, 0.0], [0.0, 576]])
R = 243.55
num_particles = 10000
m_0 = np.array([7.76, 0.848485])
P_0 = np.array([[1, 0.], [0., 1]])

# Perform spline interpolation on the given data
cs = CubicSpline(times, gfr_values)

# Interpolate at regular intervals
interpolated_times = np.linspace(times[0], times[-1], num=1000)
interpolated_gfr_values = cs(interpolated_times)

# Particle Filter function with stratified resampling
def particle_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, observations, num_particles):
    n = m_0.shape[-1]
    steps = observations.shape[0]

    particles = np.zeros((num_particles, n))
    weights = np.ones(num_particles) / num_particles

    pf_states = np.empty((steps, n))

    for i in range(steps):
        y = observations[i]

        # Define the state transition matrix F
        F = np.array([[1, dt],
                      [-dt * (3*a*m_0[0]**2 + 2*b*m_0[0] + gamma), 1 - lmbda * dt]])

        # Prediction
        for j in range(num_particles):
            # Predict state based on the state transition model with process noise
            particles[j] = np.dot(F, particles[j]) + np.random.multivariate_normal(np.zeros(n), Q)

        # Update (Correction)
        for j in range(num_particles):
            # Calculate likelihood (measurement model)
            measurement_likelihood = 1 / np.sqrt(2 * np.pi * R) * np.exp(-0.5 * (y - particles[j, 0])**2 / R)
            # Update particle weight
            weights[j] *= measurement_likelihood

        # Normalize weights
        weights /= np.sum(weights)

        # Stratified Resampling
        cumulative_weights = np.cumsum(weights)
        u = (np.random.rand() + np.arange(num_particles)) / num_particles
        indices = np.zeros(num_particles, dtype=int)
        j = 0
        for k in range(num_particles):
            while u[k] > cumulative_weights[j]:
                j += 1
            indices[k] = j

        particles = particles[indices]
        weights.fill(1.0 / num_particles)

        # Estimate state based on weighted average of particles
        estimated_state = np.average(particles, weights=weights, axis=0)
        pf_states[i] = estimated_state

    return pf_states

# Run the Particle Filter with stratified resampling using interpolated GFR data
pf_states = particle_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, interpolated_gfr_values, num_particles)

# Plotting
plt.figure(figsize=(10, 6))

# Plot estimated GFR using Particle Filter
plt.subplot(1, 2, 2)
plt.plot(interpolated_times, interpolated_gfr_values, label='Interpolated GFR', color='blue', linestyle='-', linewidth=1)
plt.plot(interpolated_times, pf_states[:, 0], label='Estimated GFR (PF)', color='green', linestyle='--', linewidth=1)
plt.xlabel('Time [days]')
plt.ylabel('GFR [ml/min]')
plt.title('PF with Systematic Resampling for Patient 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate RMSE
def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))

rmse_pf = rmse(interpolated_gfr_values, pf_states[:, 0])
print(f"PF RMSE with Systematic Resampling: {rmse_pf}")

# Length of original GFR values
original_num_observations = len(gfr_values)

# Length of interpolated GFR values
interpolated_num_observations = len(interpolated_gfr_values)

# Increase in the number of observations
increase_in_observations = interpolated_num_observations - original_num_observations

print(f"Increased observations after interpolation: {increase_in_observations}")


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Seed the random number generator for reproducibility
np.random.seed(5)

# Given data
gfr_values = np.array([7.76, 35.75, 48.12, 44.19, 67.53, 61.32, 57.59, 56.82, 62.95, 50.54, 48.21, 70.41, 52.10, 56.04])
times = np.array([9, 42, 120, 183, 309, 828, 1059, 1455, 1717, 2152, 2465, 2848, 3249, 3523])
dt = 0.1
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
Q = np.array([[576, 0.0], [0.0, 576]])
R = 243.55
num_particles = 10000
m_0 = np.array([7.76, 0.848485])
P_0 = np.array([[1, 0.], [0., 1]])

# Particle Filter function with systematic resampling
def systematic_resampling(weights, num_particles, random_state):
    cumulative_weights = np.cumsum(weights)
    u = (random_state.rand() + np.arange(num_particles)) / num_particles
    indices = np.zeros(num_particles, dtype=int)
    j = 0
    for k in range(num_particles):
        while u[k] > cumulative_weights[j]:
            j += 1
        indices[k] = j
    return indices

def particle_filter_sys_resampling(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, observations, num_particles):
    n = m_0.shape[-1]
    steps = observations.shape[0]

    particles = np.zeros((num_particles, n))
    weights = np.ones(num_particles) / num_particles

    pf_states = np.empty((steps, n))

    for i in range(steps):
        y = observations[i]

        # Define the state transition matrix F
        F = np.array([[1, dt],
                      [-dt * (3*a*m_0[0]**2 + 2*b*m_0[0] + gamma), 1 - lmbda * dt]])

        # Prediction
        for j in range(num_particles):
            # Predict state based on the state transition model with process noise
            particles[j] = np.dot(F, particles[j]) + np.random.multivariate_normal(np.zeros(n), Q)

        # Update (Correction)
        for j in range(num_particles):
            # Calculate likelihood (measurement model)
            measurement_likelihood = 1 / np.sqrt(2 * np.pi * R) * np.exp(-0.5 * (y - particles[j, 0])**2 / R)
            # Update particle weight
            weights[j] *= measurement_likelihood

        # Normalize weights
        weights /= np.sum(weights)

        # Systematic Resampling
        indices = systematic_resampling(weights, num_particles, np.random)
        particles = particles[indices]
        weights.fill(1.0 / num_particles)

        # Estimate state based on weighted average of particles
        estimated_state = np.average(particles, weights=weights, axis=0)
        pf_states[i] = estimated_state

    return pf_states

# Particle Filter function with stratified resampling
def stratified_resampling(weights, num_particles, random_state):
    u = (random_state.rand(num_particles) + np.arange(num_particles)) / num_particles
    cumulative_weights = np.cumsum(weights)
    indices = np.zeros(num_particles, dtype=int)
    j = 0
    for k in range(num_particles):
        while u[k] > cumulative_weights[j]:
            j += 1
        indices[k] = j
    return indices

def particle_filter_strat_resampling(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, observations, num_particles):
    n = m_0.shape[-1]
    steps = observations.shape[0]

    particles = np.zeros((num_particles, n))
    weights = np.ones(num_particles) / num_particles

    pf_states = np.empty((steps, n))

    for i in range(steps):
        y = observations[i]

        # Define the state transition matrix F
        F = np.array([[1, dt],
                      [-dt * (3*a*m_0[0]**2 + 2*b*m_0[0] + gamma), 1 - lmbda * dt]])

        # Prediction
        for j in range(num_particles):
            # Predict state based on the state transition model with process noise
            particles[j] = np.dot(F, particles[j]) + np.random.multivariate_normal(np.zeros(n), Q)

        # Update (Correction)
        for j in range(num_particles):
            # Calculate likelihood (measurement model)
            measurement_likelihood = 1 / np.sqrt(2 * np.pi * R) * np.exp(-0.5 * (y - particles[j, 0])**2 / R)
            # Update particle weight
            weights[j] *= measurement_likelihood

        # Normalize weights
        weights /= np.sum(weights)

        # Stratified Resampling
        indices = stratified_resampling(weights, num_particles, np.random)

        particles = particles[indices]
        weights.fill(1.0 / num_particles)

        # Estimate state based on weighted average of particles
        estimated_state = np.average(particles, weights=weights, axis=0)
        pf_states[i] = estimated_state

    return pf_states

# Run both particle filters with real GFR data
pf_states_sys_resampling = particle_filter_sys_resampling(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, gfr_values, num_particles)
pf_states_strat_resampling = particle_filter_strat_resampling(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, gfr_values, num_particles)

# Plotting both results side by side
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plotting with systematic resampling
axs[0].scatter(times, gfr_values, label='Measurements', color='black', marker='o', s=10)
axs[0].plot(times, pf_states_sys_resampling[:, 0], label='Estimated GFR (PF)', color='blue', linestyle='-', linewidth=1)
axs[0].set_xlabel('Time [days]',  fontsize=20)
axs[0].set_ylabel('GFR [ml/min]',  fontsize=20)
axs[0].set_title('PF Estimate with Systematic Resampling for Patient 2', fontsize=15)
axs[0].legend()
axs[0].grid(True)

# Plotting with stratified resampling
axs[1].scatter(times, gfr_values, label='Measurements', color='black', marker='o', s=10)
axs[1].plot(times, pf_states_strat_resampling[:, 0], label='Estimated GFR (PF)', color='magenta', linestyle='-', linewidth=1)
axs[1].set_xlabel('Time [days]',  fontsize=20)
axs[1].set_ylabel('GFR [ml/min]',  fontsize=20)
axs[1].set_title('PF Estimate with Stratified Resampling for Patient 2', fontsize=15 )
axs[1].legend()
axs[1].grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# Calculate RMSE for both methods
def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))

rmse_pf_sys_resampling = rmse(gfr_values, pf_states_sys_resampling[:, 0])
rmse_pf_strat_resampling = rmse(gfr_values, pf_states_strat_resampling[:, 0])

print(f"PF RMSE with Systematic Resampling: {rmse_pf_sys_resampling}")
print(f"PF RMSE with Stratified Resampling: {rmse_pf_strat_resampling}")


# In[ ]:





# In[6]:


import numpy as np
import matplotlib.pyplot as plt

# Seed the random number generator for reproducibility
np.random.seed(5)

# Given data
gfr_values = np.array([7.76, 35.75, 48.12, 44.19, 67.53, 61.32, 57.59, 56.82, 62.95, 50.54, 48.21, 70.41, 52.10, 56.04])
times = np.array([9, 42, 120, 183, 309, 828, 1059, 1455, 1717, 2152, 2465, 2848, 3249, 3523])
dt = 0.1
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
Q = np.array([[576, 0.0], [0.0, 576]])
R = 243.55
num_particles = 10000
m_0 = np.array([7.76, 0.848485])
P_0 = np.array([[1, 0.], [0., 1]])

# Particle Filter function with systematic resampling
def particle_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, observations, num_particles):
    n = m_0.shape[-1]
    steps = observations.shape[0]

    particles = np.random.multivariate_normal(m_0, P_0, num_particles)
    weights = np.ones(num_particles) / num_particles

    pf_states = np.empty((steps, n))

    predicted_gfr_values = []
    estimated_gfr_values = []

    for i in range(steps):
        y = observations[i]

        # Define the state transition matrix F
        F = np.array([[1, dt],
                      [-dt * (3*a*m_0[0]**2 + 2*b*m_0[0] + gamma), 1 - lmbda * dt]])

        # Prediction
        for j in range(num_particles):
            # Predict state based on the state transition model with process noise
            particles[j] = np.dot(F, particles[j]) + np.random.multivariate_normal(np.zeros(n), Q)

        # Calculate the mean predicted GFR value
        mean_predicted_gfr = np.mean(particles[:, 0])

        # Update (Correction)
        for j in range(num_particles):
            # Calculate likelihood (measurement model)
            measurement_likelihood = 1 / np.sqrt(2 * np.pi * R) * np.exp(-0.5 * (y - particles[j, 0])**2 / R)
            # Update particle weight
            weights[j] *= measurement_likelihood

        # Normalize weights
        weights /= np.sum(weights)

        # Systematic Resampling
        cumulative_weights = np.cumsum(weights)
        u = (np.random.rand() + np.arange(num_particles)) / num_particles
        indices = np.zeros(num_particles, dtype=int)
        j = 0
        for k in range(num_particles):
            while u[k] > cumulative_weights[j]:
                j += 1
            indices[k] = j

        particles = particles[indices]
        weights.fill(1.0 / num_particles)

        # Estimate state based on weighted average of particles
        estimated_state = np.average(particles, weights=weights, axis=0)
        pf_states[i] = estimated_state

        # Store the mean predicted and estimated GFR values
        predicted_gfr_values.append(mean_predicted_gfr)
        estimated_gfr_values.append(estimated_state[0])

    return pf_states, predicted_gfr_values, estimated_gfr_values

# Run the Particle Filter with systematic resampling using real GFR data
pf_states, predicted_gfr_values, estimated_gfr_values = particle_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, gfr_values, num_particles)

# Plotting
plt.figure(figsize=(10, 6))

plt.scatter(times, gfr_values, label='Measurements', color='black', marker='o', s=10)
plt.plot(times, pf_states[:, 0], label='Estimated GFR (PF)', color='blue', linestyle='-', linewidth=1)
plt.xlabel('Time[days]', fontsize=20)
plt.ylabel('GFR[ml/min]', fontsize=20)
plt.title('True GFR vs Estimated GFR (PF) for Patient 2 with Systematic Resampling', fontsize=15)
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE
def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))

rmse_pf = rmse(gfr_values, pf_states[:, 0])
print(f"PF RMSE with Systematic Resampling: {rmse_pf}")

# Print the dictionary with measured, predicted, and estimated GFR values
#data = {
    #'Measured GFR (ml/min)': gfr_values.tolist(),
    #'Predicted GFR (ml/min)': predicted_gfr_values,
   # 'Estimated GFR (ml/min)': estimated_gfr_values
#}

#print(data)


# In[ ]:




