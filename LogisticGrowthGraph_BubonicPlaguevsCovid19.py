# SIR model parameters for comparison of bubonic plague in Denver population vs COVID-19 spread. 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp

# Total population of Denver
N = 729000

# Initial conditions: one infected, rest susceptible, no recovered initially
I0 = 1      
S0 = N - I0
R0 = 0

# Parameters (beta: transmission rate, gamma: recovery rate)
# Bubonic plague parameters (modern estimates)
beta_plague = 0.3
gamma_plague = 0.1

# COVID-19 parameters (high transmission scenario)
beta_covid = 0.5
gamma_covid = 0.1

# SIR model differential equations
def sir_model(t, y, beta, gamma):
    S, I, R =y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt] 
  
# Time points (e.g., 160 days)
t_span = [0, 160]
t_eval = np.linspace(t_span[0], t_span[1], 160)

# Solve the ODE system for both scenarios
sol_plague = solve_ivp(sir_model, t_span, [S0, I0, R0], args=(beta_plague, gamma_plague), t_eval=t_eval)
sol_covid = solve_ivp(sir_model, t_span, [S0, I0, R0], args=(beta_covid, gamma_covid), t_eval=t_eval)

# Plotting (width and height in inches)    
plt.figure(figsize=(12,6))

# Bubonic Plague left side
plt.subplot(1, 2, 1)
plt.plot(sol_plague.t, sol_plague.y[0], 'b', label='Susceptible')
plt.plot(sol_plague.t, sol_plague.y[1], 'r', label='Infected')
plt.plot(sol_plague.t, sol_plague.y[2], 'g', label='Recovered')
plt.plot(sol_plague.t, sol_plague.y[1] * 0.1, 'k--', label='Estimated Deaths (10%)')  # Modern day estimated deaths (was about 60% when it first spread)
plt.title('Bubonic Plague SIR Model in Denver Population')      
plt.xlabel('Days')
plt.ylabel('Number of People')      
plt.legend()
plt.grid(True)

# COVID-19 right side
plt.subplot(1, 2, 2)
plt.plot(sol_covid.t, sol_covid.y[0], 'b', label='Susceptible')
plt.plot(sol_covid.t, sol_covid.y[1], 'r', label='Infected')
plt.plot(sol_covid.t, sol_covid.y[2], 'g', label='Recovered')
plt.plot(sol_covid.t, sol_covid.y[1] * 0.014, 'k--', label='Estimated Deaths (1.4%)')  # Estimated deaths line
plt.title('COVID-19 SIR Model in Denver Population')        
plt.xlabel('Days')
plt.ylabel('Number of People')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final numbers after simulation
# The -1 index is a Python shortcut for the last element along that axis, so you get values at the final simulation time. S starts at 0. Initial starts at 1 because 1 was infected first. 
S_final_plague = sol_plague.y[0, -1]
I_final_plague = sol_plague.y[1, -1]
R_final_plague = sol_plague.y[1, -1]    

# Final numbers for COVID-19
S_final_covid = sol_covid.y[0, -1]
I_final_covid = sol_covid.y[1, -1]  
R_final_covid = sol_covid.y[2, -1]

# Total infected during the bubonic plague
# Total population (N) minus susceptible at end
total_infected_plague = N - S_final_plague

# Total infected during COVID-19
# Total population (N) minus susceptible at end
total_infected_covid = N - S_final_covid  

# Total deaths assuming 10% fatality rate for bubonic plague  
deaths_plague = total_infected_plague * 0.1

# Total deaths assuming 1.4% fatality rate for COVID-19
deaths_covid = total_infected_covid * 0.014

# Print at what day the peak infection occurs for both diseases
# Use .t to get the time points
# Finds the index where the infected series is largest. 
peak_day_plague = sol_plague.t[np.argmax(sol_plague.y[1])]
peak_day_covid = sol_covid.t[np.argmax(sol_covid.y[1])]

# Print the amount infected at peak for both diseases  
# Use the .y attribute to access the solution values
# Finds the index where the infected series is largest. 
peak_infected_plague = sol_plague.y[1][np.argmax(sol_plague.y[1])]
peak_infected_covid = sol_covid.y[1][np.argmax(sol_covid.y[1])]

# Total dead at peak
deaths_peak_plague = peak_infected_plague * 0.1
deaths_peak_covid = peak_infected_covid * 0.014
  
  
# Print final results:Bubonic Plague
# 0f is used to round to nearest whole number 
print(f"Bubonic Plague Final Results:") 
print(f"Total Infected: {total_infected_plague:.0f}")
print(f"Total Deaths (10% fatality): {deaths_plague:.0f}")  
print(f"Peak Infection Day: {peak_day_plague:.0f}")    
print(f"Peak Infected: {peak_infected_plague:.0f}")  
print(f"Deaths at Peak: {deaths_peak_plague:.0f}")

# Print final results: COVID-19
# 0f is used to round to nearest whole number 
print(f"\nCOVID-19 Final Results:")
print(f"Total Infected: {total_infected_covid:.0f}")
print(f"Total Deaths (1.4% fatality): {deaths_covid:.0f}")
print(f"Peak Infection Day: {peak_day_covid:.0f}")   
print(f"Peak Infected: {peak_infected_covid:.0f}")
print(f"Deaths at Peak: {deaths_peak_covid:.0f}")
print(f"")