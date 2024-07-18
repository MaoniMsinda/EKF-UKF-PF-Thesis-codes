#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
deltaTime = 0.1
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
x0 = np.array([7.76, 0.848485])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector = np.arange(0, simulationSteps * deltaTime, deltaTime)

# State space model
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration
solution_euler = euler_integration(x0, deltaTime, simulationSteps)

# Plot results
plt.plot(totalSimulationTimeVector, solution_euler[:, 0], 'b', label='gfr for patient2')
#plt.plot(totalSimulationTimeVector, solution_euler[:, 1], 'g', label='x2 (Euler)')
plt.legend(loc='best')
plt.xlabel('time (days)')
plt.ylabel('x1(t)')
plt.title('Model simulation using Euler method')
plt.grid()
plt.savefig('simulation_euler.png', dpi=600)
plt.show()


# In[ ]:





# In[18]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
deltaTime = 0.1
a = 8.7E-07
b = -3.3E-05
lmbda = 0.0378
gamma = 3.1E-04
delta = 0
x0 = np.array([9.23, 0.468889])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector = np.arange(0, simulationSteps * deltaTime, deltaTime)

# State space model
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration
solution_euler = euler_integration(x0, deltaTime, simulationSteps)

# Plot results
plt.plot(totalSimulationTimeVector, solution_euler[:, 0], 'b', label='gfr for patient9')
#plt.plot(totalSimulationTimeVector, solution_euler[:, 1], 'g', label='x2 (Euler)')
plt.legend(loc='best')
plt.xlabel('time (days)')
plt.ylabel('x1(t)')
plt.grid()
plt.title('Model simulation using Euler method')
plt.savefig('simulation_euler.png', dpi=600)
plt.show()


# In[ ]:





# In[16]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
deltaTime = 0.1
a = 4.83E-07
b = -2.8E-05
lmbda = 0.0061
gamma = 3.1E-04
delta = 0
x0 = np.array([9.81, 0.301818])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector = np.arange(0, simulationSteps * deltaTime, deltaTime)

# State space model
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration
solution_euler = euler_integration(x0, deltaTime, simulationSteps)

# Plot results
plt.plot(totalSimulationTimeVector, solution_euler[:, 0], 'b', label='gfr for patient 15')
#plt.plot(totalSimulationTimeVector, solution_euler[:, 1], 'g', label='x2 (Euler)')
plt.legend(loc='best')
plt.xlabel('time (days)')
plt.ylabel('x1(t)')
plt.title('Model simulation using Euler method')
plt.grid()
plt.savefig('simulation_euler.png', dpi=600)
plt.show()


# In[ ]:





# In[ ]:





# In[13]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
deltaTime = 0.1
a = 5.58E-07
b = -2.88E-05
lmbda = 0.1166
gamma = 3.1E-04
delta = 0
x0 = np.array([52.57, 0.094694])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector = np.arange(0, simulationSteps * deltaTime, deltaTime)

# State space model
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration
solution_euler = euler_integration(x0, deltaTime, simulationSteps)

# Plot results
plt.plot(totalSimulationTimeVector, solution_euler[:, 0], 'b', label='gfr for patient 59')
#plt.plot(totalSimulationTimeVector, solution_euler[:, 1], 'g', label='x2 (Euler)')
plt.legend(loc='best')
plt.xlabel('time (days)')
plt.ylabel('x1(t)')
plt.title('Model simulation using Euler method')
plt.grid()
plt.savefig('simulation_euler.png', dpi=600)
plt.show()


# In[ ]:





# In[2]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Parameters for deterministic simulation
deltaTime = 0.1
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
x0_det = np.array([7.76, 0.848485])

# Parameters for stochastic simulation
delta_t = 0.1  # Discretization step
Q = np.array([[1.0E-04, 0.0], [0.0, 1.0E-04]])
x0_stoch = np.array([7.76, 0.848485])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector_det = np.arange(0, simulationSteps * deltaTime, deltaTime)
totalSimulationTimeVector_stoch = np.arange(0, simulationSteps * delta_t, delta_t)

# State space model for deterministic simulation
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method for deterministic simulation
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration for deterministic simulation
solution_euler_det = euler_integration(x0_det, deltaTime, simulationSteps)

# Discrete-time state-space model using Euler-Maruyama method for stochastic simulation
def discrete_state_space_model(x, dt):
    q_k = np.random.multivariate_normal([0, 0], Q)
    x1_k1 = x[0] + dt * x[1]
    x2_k1 = x[1] - lmbda * dt * x[1] - dt * (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta) + q_k[1]
    return np.array([x1_k1, x2_k1])

# Perform simulation using Euler-Maruyama method for stochastic simulation
def euler_maruyama_simulation(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = discrete_state_space_model(x[i-1], dt)
    return x

# Perform simulation for stochastic simulation
solution_euler_maruyama = euler_maruyama_simulation(x0_stoch, delta_t, simulationSteps)

# Plot results on the same figure
plt.plot(totalSimulationTimeVector_det, solution_euler_det[:, 0], 'blue', label='Deterministic using Euler Method', linestyle='-', linewidth=1)
plt.plot(totalSimulationTimeVector_stoch, solution_euler_maruyama[:, 0], 'magenta', label='Stochastic using Euler-Maruyama method', linestyle='-', linewidth=1)
plt.legend(loc='best')
plt.xlabel('Time (days)')
plt.ylabel('GFR')
plt.title('Model simulation with patient 2 parameters')
plt.grid()
plt.savefig('simulation_comparison.png', dpi=600)
plt.show()


# In[ ]:





# In[6]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Parameters for deterministic simulation
deltaTime = 0.1
a = 8.7E-07
b = -3.3E-05
lmbda = 0.0378
gamma = 3.1E-04
delta = 0
x0_det = np.array([9.23, 0.468889])

# Parameters for stochastic simulation
delta_t = 0.1  # Discretization step
Q = np.array([[1.0E-07, 0.0], [0.0, 1.0E-07]])
x0_stoch = np.array([9.23, 0.468889])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector_det = np.arange(0, simulationSteps * deltaTime, deltaTime)
totalSimulationTimeVector_stoch = np.arange(0, simulationSteps * delta_t, delta_t)

# State space model for deterministic simulation
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method for deterministic simulation
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration for deterministic simulation
solution_euler_det = euler_integration(x0_det, deltaTime, simulationSteps)

# Discrete-time state-space model using Euler-Maruyama method for stochastic simulation
def discrete_state_space_model(x, dt):
    q_k = np.random.multivariate_normal([0, 0], Q)
    x1_k1 = x[0] + dt * x[1]
    x2_k1 = x[1] - lmbda * dt * x[1] - dt * (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta) + q_k[1]
    return np.array([x1_k1, x2_k1])

# Perform simulation using Euler-Maruyama method for stochastic simulation
def euler_maruyama_simulation(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = discrete_state_space_model(x[i-1], dt)
    return x

# Perform simulation for stochastic simulation
solution_euler_maruyama = euler_maruyama_simulation(x0_stoch, delta_t, simulationSteps)

# Plot results on the same figure
plt.plot(totalSimulationTimeVector_det, solution_euler_det[:, 0], 'blue', label='Deterministic using Euler Method', linestyle='-', linewidth=1)
plt.plot(totalSimulationTimeVector_stoch, solution_euler_maruyama[:, 0], 'magenta', label='Stochastic using Euler-Maruyama method', linestyle='-', linewidth=1)
plt.legend(loc='best')
plt.xlabel('Time (days)')
plt.ylabel('GFR')
plt.title('Model simulation with patient 9 parameters')
plt.grid()
plt.savefig('simulation_comparison.png', dpi=600)
plt.show()


# In[ ]:





# In[3]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Parameters for deterministic simulation
deltaTime = 0.1
a = 2.32E-07
b = -2.39E-05
lmbda = 0.0323
gamma = 3.1E-04
delta = 0
x0_det = np.array([91.13, -0.18540])

# Parameters for stochastic simulation
delta_t = 0.1  # Discretization step
Q = np.array([[1.0E-06, 0.0], [0.0, 1.0E-6]])
x0_stoch = np.array([91.13, -0.18540])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector_det = np.arange(0, simulationSteps * deltaTime, deltaTime)
totalSimulationTimeVector_stoch = np.arange(0, simulationSteps * delta_t, delta_t)

# State space model for deterministic simulation
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method for deterministic simulation
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration for deterministic simulation
solution_euler_det = euler_integration(x0_det, deltaTime, simulationSteps)

# Discrete-time state-space model using Euler-Maruyama method for stochastic simulation
def discrete_state_space_model(x, dt):
    q_k = np.random.multivariate_normal([0, 0], Q)
    x1_k1 = x[0] + dt * x[1]
    x2_k1 = x[1] - lmbda * dt * x[1] - dt * (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta) + q_k[1]
    return np.array([x1_k1, x2_k1])

# Perform simulation using Euler-Maruyama method for stochastic simulation
def euler_maruyama_simulation(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = discrete_state_space_model(x[i-1], dt)
    return x

# Perform simulation for stochastic simulation
solution_euler_maruyama = euler_maruyama_simulation(x0_stoch, delta_t, simulationSteps)

# Plot results on the same figure
plt.plot(totalSimulationTimeVector_det, solution_euler_det[:, 0], 'blue', label='Deterministic using Euler Method', linestyle='-', linewidth=1)
plt.plot(totalSimulationTimeVector_stoch, solution_euler_maruyama[:, 0], 'magenta', label='Stochastic using Euler-Maruyama Method',linestyle='-', linewidth=1 )
plt.legend(loc='best')
plt.xlabel('Time (days)')
plt.ylabel('GFR')
plt.title('Model simulation with Patient 123 parameters')
plt.grid()
plt.savefig('simulation_comparison.png', dpi=600)
plt.show()


# In[ ]:





# In[4]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Parameters for deterministic simulation
deltaTime = 0.1
a = 4.83E-07
b = -2.8E-05
lmbda = 0.0061
gamma = 3.1E-04
delta = 0
x0_det = np.array([9.81, 0.301818])

# Parameters for stochastic simulation
delta_t = 0.1  # Discretization step
Q = np.array([[1.0E-05, 0.0], [0.0, 1.0E-05]])
x0_stoch = np.array([9.81, 0.301818])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector_det = np.arange(0, simulationSteps * deltaTime, deltaTime)
totalSimulationTimeVector_stoch = np.arange(0, simulationSteps * delta_t, delta_t)

# State space model for deterministic simulation
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method for deterministic simulation
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration for deterministic simulation
solution_euler_det = euler_integration(x0_det, deltaTime, simulationSteps)

# Discrete-time state-space model using Euler-Maruyama method for stochastic simulation
def discrete_state_space_model(x, dt):
    q_k = np.random.multivariate_normal([0, 0], Q)
    x1_k1 = x[0] + dt * x[1]
    x2_k1 = x[1] - lmbda * dt * x[1] - dt * (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta) + q_k[1]
    return np.array([x1_k1, x2_k1])

# Perform simulation using Euler-Maruyama method for stochastic simulation
def euler_maruyama_simulation(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = discrete_state_space_model(x[i-1], dt)
    return x

# Perform simulation for stochastic simulation
solution_euler_maruyama = euler_maruyama_simulation(x0_stoch, delta_t, simulationSteps)

# Plot results on the same figure
plt.plot(totalSimulationTimeVector_det, solution_euler_det[:, 0], 'blue', label='Deterministic using Euler Method', linestyle='-', linewidth=1 )
plt.plot(totalSimulationTimeVector_stoch, solution_euler_maruyama[:, 0], 'magenta', label='Stochastic using Euler-Maruyama Method', linestyle='-', linewidth=1 )
plt.legend(loc='best')
plt.xlabel('Time (days)')
plt.ylabel('GFR')
plt.title('Model simulation with patient 15 parameters')
plt.grid()
plt.savefig('simulation_comparison.png', dpi=600)
plt.show()


# In[ ]:





# In[5]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
# Parameters for deterministic simulation
deltaTime = 0.1
a = 5.58E-07
b = -2.88E-05
lmbda = 0.1166
gamma = 3.1E-04
delta = 0
x0_det = np.array([52.57, 0.094694])

# Parameters for stochastic simulation
delta_t = 0.1  # Discretization step
Q = np.array([[1.0E-05, 0.0], [0.0, 1.0E-05]])
x0_stoch = np.array([52.57, 0.094694])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector_det = np.arange(0, simulationSteps * deltaTime, deltaTime)
totalSimulationTimeVector_stoch = np.arange(0, simulationSteps * delta_t, delta_t)

# State space model for deterministic simulation
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method for deterministic simulation
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration for deterministic simulation
solution_euler_det = euler_integration(x0_det, deltaTime, simulationSteps)

# Discrete-time state-space model using Euler-Maruyama method for stochastic simulation
def discrete_state_space_model(x, dt):
    q_k = np.random.multivariate_normal([0, 0], Q)
    x1_k1 = x[0] + dt * x[1]
    x2_k1 = x[1] - lmbda * dt * x[1] - dt * (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta) + q_k[1]
    return np.array([x1_k1, x2_k1])

# Perform simulation using Euler-Maruyama method for stochastic simulation
def euler_maruyama_simulation(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = discrete_state_space_model(x[i-1], dt)
    return x

# Perform simulation for stochastic simulation
solution_euler_maruyama = euler_maruyama_simulation(x0_stoch, delta_t, simulationSteps)

# Plot results on the same figure
plt.plot(totalSimulationTimeVector_det, solution_euler_det[:, 0], 'b', label='Deterministic using Euler Method', linestyle='-', linewidth=1)
plt.plot(totalSimulationTimeVector_stoch, solution_euler_maruyama[:, 0], 'magenta', label='Stochastic using Euler-Maruyama Method', linestyle='-', linewidth=1)
plt.legend(loc='best')
plt.xlabel('Time (days)')
plt.ylabel('GFR')
plt.title('Model simulation with Patient 59 parameters')
plt.grid()
plt.savefig('simulation_comparison_patient_59.png', dpi=600)
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
# Parameters for deterministic simulation
deltaTime = 0.1
a = 3.52E-07
b = -2.6E-05
lmbda = 0.0269
gamma = 3.1E-04
delta = 0
x0_det = np.array([50.43, 0.103421])

# Parameters for stochastic simulation
delta_t = 0.1  # Discretization step
Q = np.array([[1.0E-05, 0.0], [0.0, 1.0E-05]])
x0_stoch = np.array([50.43, 0.103421])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector_det = np.arange(0, simulationSteps * deltaTime, deltaTime)
totalSimulationTimeVector_stoch = np.arange(0, simulationSteps * delta_t, delta_t)

# State space model for deterministic simulation
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method for deterministic simulation
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration for deterministic simulation
solution_euler_det = euler_integration(x0_det, deltaTime, simulationSteps)

# Discrete-time state-space model using Euler-Maruyama method for stochastic simulation
def discrete_state_space_model(x, dt):
    q_k = np.random.multivariate_normal([0, 0], Q)
    x1_k1 = x[0] + dt * x[1]
    x2_k1 = x[1] - lmbda * dt * x[1] - dt * (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta) + q_k[1]
    return np.array([x1_k1, x2_k1])

# Perform simulation using Euler-Maruyama method for stochastic simulation
def euler_maruyama_simulation(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = discrete_state_space_model(x[i-1], dt)
    return x

# Perform simulation for stochastic simulation
solution_euler_maruyama = euler_maruyama_simulation(x0_stoch, delta_t, simulationSteps)

# Plot results on the same figure
plt.plot(totalSimulationTimeVector_det, solution_euler_det[:, 0], 'b', label='Deterministic using Euler Method', linestyle='-', linewidth=1)
plt.plot(totalSimulationTimeVector_stoch, solution_euler_maruyama[:, 0], 'magenta', label='Stochastic using Euler-Maruyama Method', linestyle='-', linewidth=1)
plt.legend(loc='best')
plt.xlabel('Time (days)')
plt.ylabel('GFR')
plt.title('Model simulation with Patient 88 parameters')
plt.grid()
plt.savefig('simulation_comparison_patient_88.png', dpi=600)
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Parameters for deterministic simulation
deltaTime = 0.1
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
x0_det = np.array([7.76, 0.848485])

# Parameters for stochastic simulation
delta_t = 0.1  # Discretization step
Q = np.array([[1.0E-04, 0.0], [0.0, 1.0E-04]])
x0_stoch = np.array([7.76, 0.848485])

# Time vector
simulationSteps = 36000  
totalSimulationTimeVector_det = np.arange(0, simulationSteps * deltaTime, deltaTime)
totalSimulationTimeVector_stoch = np.arange(0, simulationSteps * delta_t, delta_t)

# State space model for deterministic simulation
def stateSpaceModel(x):
    dxdt = np.array([
        x[1],
        -lmbda * x[1] - (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta)
    ])
    return dxdt

# Integrate the dynamics using Euler's method for deterministic simulation
def euler_integration(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i-1] + dt * stateSpaceModel(x[i-1])
    return x

# Perform Euler integration for deterministic simulation
solution_euler_det = euler_integration(x0_det, deltaTime, simulationSteps)

# Discrete-time state-space model using Euler-Maruyama method for stochastic simulation
def discrete_state_space_model(x, dt):
    q_k = np.random.multivariate_normal([0, 0], Q)
    x1_k1 = x[0] + dt * x[1]
    x2_k1 = x[1] - lmbda * dt * x[1] - dt * (a * x[0]**3 + b * x[0]**2 + gamma * x[0] + delta) + q_k[1]
    return np.array([x1_k1, x2_k1])

# Perform simulation using Euler-Maruyama method for stochastic simulation
def euler_maruyama_simulation(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        x[i] = discrete_state_space_model(x[i-1], dt)
    return x

# Perform simulation for stochastic simulation
solution_euler_maruyama = euler_maruyama_simulation(x0_stoch, delta_t, simulationSteps)

# Plot results on the same figure
plt.plot(totalSimulationTimeVector_det, solution_euler_det[:, 0], 'blue', label='Deterministic using Euler Method', linestyle='-', linewidth=1)
plt.plot(totalSimulationTimeVector_stoch, solution_euler_maruyama[:, 0], 'magenta', label='Stochastic using Euler-Maruyama method', linestyle='-', linewidth=1)
plt.legend(loc='best')
plt.xlabel('Time (days)')
plt.ylabel('GFR')
plt.title('Model simulation with patient 2 parameters')
plt.grid()
plt.savefig('simulation_comparison.png', dpi=600)
plt.show()


# In[ ]:





# In[5]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Define parameters
np.random.seed(0)
delta_t = 0.1  # Discretization step
a = 3.02E-07
b = -2.50E-05
lmbda = 0.0373
gamma = 3.1E-04
delta = 0
Q = np.array([[0.0001, 0.0], [0.0, 0.0001]])  # Process noise covariance matrix
R = 16  # Measurement noise covariance
m_0 = np.array([7.76, 0.848485])  # Initial state estimate
P_0 = np.array([[1, 0.], [0., 1]])  # Initial error covariance

# Function to generate true trajectory with process noise
def generate_true_trajectory(x0, dt, steps):
    x = np.zeros((steps, len(x0)))
    x[0] = x0
    for i in range(1, steps):
        q_k = np.random.multivariate_normal([0, 0], Q)
        x1_k1 = x[i-1, 0] + dt * x[i-1, 1]
        x2_k1 = x[i-1, 1] - lmbda * dt * x[i-1, 1] - dt * (a * x[i-1, 0]**3 + b * x[i-1, 0]**2 + gamma * x[i-1, 0] + delta) + q_k[1]
        x[i] = np.array([x1_k1, x2_k1])
    return x

# Generate true trajectory with process noise
simulation_steps = 36000
true_states = generate_true_trajectory(m_0, delta_t, simulation_steps)

# Generate noisy measurements
measurements = true_states[:, 0] + np.random.normal(0, np.sqrt(R), simulation_steps)

# Function to perform Extended Kalman Filter
def extended_kalman_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, dt, R, observations):
    n = m_0.shape[-1]
    steps = observations.shape[0]
    
    ekf_states = np.empty((steps, n))
    ekf_P = np.empty((steps, n, n))
    
    m = m_0[:]
    P = P_0[:]
    
    for i in range(steps):
        y = observations[i]
        
        # Jacobian of the dynamic model function
        F = np.array([[1., dt], 
                      [-dt * (3*a*m[0]**2 + 2*b*m[0] + gamma), 1 - lmbda * dt]])
        
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
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ np.atleast_1d(y - h)
        P = P - K @ S @ K.T
        ekf_states[i] = m
        ekf_P[i] = P
    return ekf_states, ekf_P

# Run the Extended Kalman Filter using noisy measurements
ekf_states, _ = extended_kalman_filter(m_0, P_0, lmbda, a, b, gamma, delta, Q, delta_t, R, measurements)
#print('ekf_states',ekf_states)
# Compute RMSE
ekf_rmse = np.sqrt(np.mean((ekf_states[:, 0] - true_states[:, 0])**2))
print("Root Mean Square Error (RMSE):", ekf_rmse)


# Plot true GFR, measurements, and filter estimates
plt.figure(figsize=(10, 6))
plt.plot(true_states[:, 0], label='True trajectory', linestyle='-', color='blue',linewidth=3)
plt.scatter(np.arange(simulation_steps), measurements, label='Measurements', marker='o', color='black', s=1)
plt.plot(ekf_states[:, 0], label='Filter Estimate', linestyle='-', color='red', linewidth=1)

plt.xlabel('Time Step')
plt.ylabel('GFR')
plt.legend()
plt.grid(True)
plt.title('Extended Kalman Filter for Patient 2 GFR Estimation')
plt.show()




# In[ ]:





# In[ ]:




