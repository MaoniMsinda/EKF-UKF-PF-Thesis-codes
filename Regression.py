#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Data from the table
data = {
    'Measured GFR (ml/min)': [9.23, 17.67, 18.6, 22.7, 23.46, 23.46, 25.63, 24.3, 17.49, 14.28, 19.97, 16.35],
    'Predicted GFR (ml/min)': [8.780223370997067, 17.367674863494642, 18.55230177003754, 22.568142892664167, 
                               23.515298485307355, 23.377776807650445, 25.550667779682374, 24.366857003081194, 
                               17.746146501390754, 14.389260064667296, 19.745473201147263, 16.353286786465667],
    'Estimated GFR (ml/min)': [8.78022337, 17.36767486, 18.55230177, 22.56814289, 23.51529849, 23.37777681, 
                               25.55066778, 24.366857, 17.7461465, 14.38926006, 19.7454732, 16.35328679]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the style of the visualization
sns.set(style="whitegrid")

# Create regression plots
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Measured vs Predicted GFR
sns.regplot(x='Measured GFR (ml/min)', y='Predicted GFR (ml/min)', data=df, color= 'black', ax=axs[0])
axs[0].set_title('Regression Based on PF for Patient 9', fontsize=20)
axs[0].set_xlabel('Measured GFR (ml/min)', fontsize=20 )
axs[0].set_ylabel('Predicted GFR (ml/min)', fontsize=20)

# Calculate regression details
slope, intercept, r_value, p_value, std_err = linregress(df['Measured GFR (ml/min)'], df['Predicted GFR (ml/min)'])
axs[0].plot(df['Measured GFR (ml/min)'], intercept + slope * df['Measured GFR (ml/min)'], 'r', label=f'Linear fit: y={slope:.2f}x+{intercept:.2f}')
axs[0].legend()

# Measured vs Estimated GFR
sns.regplot(x='Measured GFR (ml/min)', y='Estimated GFR (ml/min)', data=df, color='black', ax=axs[1])
axs[1].set_title('Regression Based on PF for Patient 9', fontsize=20)
axs[1].set_xlabel('Measured GFR (ml/min)',  fontsize=20)
axs[1].set_ylabel('Estimated GFR (ml/min)', fontsize=20 )

# Calculate regression details
slope, intercept, r_value, p_value, std_err = linregress(df['Measured GFR (ml/min)'], df['Estimated GFR (ml/min)'])
axs[1].plot(df['Measured GFR (ml/min)'], intercept + slope * df['Measured GFR (ml/min)'], 'r', label=f'Linear fit: y={slope:.2f}x+{intercept:.2f}')
axs[1].legend()

# Display the plots
plt.tight_layout()
plt.show()


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Given data for predicted GFR
times_pred = np.array([9, 42, 120, 183, 309, 828, 1059, 1455, 1717, 2152, 2465, 2848, 3249, 3523])
gfr_measurements_pred = np.array([7.76, 35.75, 48.12, 44.19, 67.53, 61.32, 57.59, 56.82, 62.95, 50.54, 48.21, 70.41, 52.10, 56.04])
predicted_gfr = np.array([5.54, 28.33, 43.48, 44.30, 61.99, 61.56, 58.51, 57.36, 61.74, 53.02, 49.52, 65.63, 55.16, 56.01])

# Given data for estimated GFR
times_est = np.array([9, 42, 120, 183, 309, 828, 1059, 1455, 1717, 2152, 2465, 2848, 3249, 3523])
gfr_measurements_est = np.array([7.76, 35.75, 48.12, 44.19, 67.53, 61.32, 57.59, 56.82, 62.95, 50.54, 48.21, 70.41, 52.10, 56.04])
estimated_gfr = np.array([5.53998278, 28.32521797, 43.48031164, 44.29586337, 61.99052492, 
                          61.56358176, 58.51409448, 57.36149825, 61.73733891, 53.02096082, 
                          49.51817475, 65.63397083, 55.16150706, 56.0113568])

# Create DataFrames
data_pred = {
    'Time (days)': times_pred,
    'GFR Measurements (ml/min)': gfr_measurements_pred,
    'Predicted GFR (ml/min)': predicted_gfr
}

data_est = {
    'Time (days)': times_est,
    'GFR Measurements (ml/min)': gfr_measurements_est,
    'Estimated GFR (ml/min)': estimated_gfr
}

df_pred = pd.DataFrame(data_pred)
df_est = pd.DataFrame(data_est)

# Set the style of the visualization
sns.set(style="whitegrid")

# Create the regression plots
plt.figure(figsize=(15, 6))

# Measured vs Predicted GFR
plt.subplot(1, 2, 1)
sns.regplot(x='GFR Measurements (ml/min)', y='Predicted GFR (ml/min)', data=df_pred,color='black',  ci=95)
plt.title('Regression Based on PF for Patient 2',  fontsize=20)
plt.xlabel('Measured GFR (ml/min)',  fontsize=20)
plt.ylabel('Predicted GFR (ml/min)',  fontsize=20)

# Calculate regression details for predicted GFR
slope_pred, intercept_pred, _, _, _ = linregress(df_pred['GFR Measurements (ml/min)'], df_pred['Predicted GFR (ml/min)'])
plt.plot(df_pred['GFR Measurements (ml/min)'], intercept_pred + slope_pred * df_pred['GFR Measurements (ml/min)'], 'r', label=f'Linear fit: y={slope_pred:.2f}x+{intercept_pred:.2f}')
plt.legend()

# Measured vs Estimated GFR
plt.subplot(1, 2, 2)
sns.regplot(x='GFR Measurements (ml/min)', y='Estimated GFR (ml/min)', data=df_est, color='black', ci=95)
plt.title('Regression Based on PF for Patient 2', fontsize=20 )
plt.xlabel('Measured GFR (ml/min)', fontsize=20 )
plt.ylabel('Estimated GFR (ml/min)',  fontsize=20)

# Calculate regression details for estimated GFR
slope_est, intercept_est, _, _, _ = linregress(df_est['GFR Measurements (ml/min)'], df_est['Estimated GFR (ml/min)'])
plt.plot(df_est['GFR Measurements (ml/min)'], intercept_est + slope_est * df_est['GFR Measurements (ml/min)'], 'r', label=f'Linear fit: y={slope_est:.2f}x+{intercept_est:.2f}')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Data from the table
data = {
    'Time (days)': [9, 42, 120, 183, 309, 828, 1059, 1455, 1717, 2152, 2465, 2848, 3249, 3523],
    'GFR Measurements (ml/min)': [7.76, 35.75, 48.12, 44.19, 67.53, 61.32, 57.59, 56.82, 62.95, 50.54, 48.21, 70.41, 52.10, 56.04],
    'Predicted GFR (ml/min)': [7.84, 7.87, 29.18, 44.07, 44.65, 63.23, 62.77, 59.60, 58.15, 62.77, 53.61, 49.44, 66.87, 55.73],
    'Estimated GFR (ml/min)': [7.79, 28.94, 43.59, 44.16, 62.16, 61.76, 58.79, 57.46, 61.86, 53.32, 49.43, 65.69, 55.41, 55.97]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the style of the visualization
sns.set(style="whitegrid")

# Create regression plots
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Measured vs Predicted GFR
sns.regplot(x='GFR Measurements (ml/min)', y='Predicted GFR (ml/min)', data=df, color='black', ax=axs[0])
axs[0].set_title('Regression Based on EKF for Patient 2', fontsize=20 )
axs[0].set_xlabel('Measured GFR (ml/min)', fontsize=20)
axs[0].set_ylabel('Predicted GFR (ml/min)', fontsize=20)

# Calculate regression details
slope, intercept, r_value, p_value, std_err = linregress(df['GFR Measurements (ml/min)'], df['Predicted GFR (ml/min)'])
axs[0].plot(df['GFR Measurements (ml/min)'], intercept + slope * df['GFR Measurements (ml/min)'], 'r', label=f'Linear fit: y={slope:.2f}x+{intercept:.2f}')
axs[0].legend()

# Measured vs Estimated GFR
sns.regplot(x='GFR Measurements (ml/min)', y='Estimated GFR (ml/min)', data=df, color='black', ax=axs[1])
axs[1].set_title('Regression Based on EKF for Patient 2', fontsize=20 )
axs[1].set_xlabel('Measured GFR (ml/min)', fontsize=20 )
axs[1].set_ylabel('Estimated GFR (ml/min)',  fontsize=20)

# Calculate regression details
slope, intercept, r_value, p_value, std_err = linregress(df['GFR Measurements (ml/min)'], df['Estimated GFR (ml/min)'])
axs[1].plot(df['GFR Measurements (ml/min)'], intercept + slope * df['GFR Measurements (ml/min)'], 'r', label=f'Linear fit: y={slope:.2f}x+{intercept:.2f}')
axs[1].legend()

# Display the plots
plt.tight_layout()
plt.show()


# In[ ]:




