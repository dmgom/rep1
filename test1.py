# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:46:26 2023

@author: mauri
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
n_years = 20
n_age_groups = 4
population_H = np.zeros((n_years, n_age_groups))
population_M = np.zeros((n_years, n_age_groups))

# Datos iniciales
initial_population_total = 51520000
 
initial_population_H = np.array([0.1847 * initial_population_total, 0.1607 * initial_population_total, 0.0917 * initial_population_total, 0.0512 * initial_population_total])
initial_population_M = np.array([0.1752 * initial_population_total, 0.1402 * initial_population_total, 0.1094 * initial_population_total, 0.0871 * initial_population_total])
population_H[0, :] = initial_population_H
population_M[0, :] = initial_population_M

# Parámetros adicionales
birth_rate = 0.14  #tasa de natalidad anual
unemployment_rate = 0.112 #porcentaje de la población que está desempleada
informality_rate = 0.58  #porcentaje de la población que trabaja en la economía informal
average_salary = 19240596  #salario promedio en la economía
inflation_rate = 0.07  #tasa de inflación anual
E_initial = 22  #edad inicial de la población cotizando
E_jubilacion_M = 57  #edad de jubilación para mujeres
E_jubilacion_H = 62  #edad de jubilación para hombres
A = 0.4  #porcentaje de salario que se aporta al fondo
B = 0.7 #porcentaje del salario pagado a los jubilados
esperanza_vida = 75  #esperanza de vida al nacer

# Tasas de mortalidad por grupo etario (estimaciones ficticias)
mortality_rates = np.array([0.008, 0.008, 0.012, 0.0])

# Funciones para el modelo
def update_population(pop, birth_rate, mortality_rates, E_initial, E_jubilacion, esperanza_vida):
    new_pop = np.zeros_like(pop)
   
    # Aplicar tasas de mortalidad
    pop_after_mortality = pop * (1 - mortality_rates)
   
    new_pop[0] = pop_after_mortality[0] - pop_after_mortality[0] * (1 / (E_initial - 1)) + (birth_rate * (pop_after_mortality[1]))
    new_pop[1] = pop_after_mortality[1] - pop_after_mortality[1] * (1 / ((E_jubilacion - E_initial)/2)) + pop_after_mortality[0] * (1 / (E_initial - 1))
    new_pop[2] = pop_after_mortality[2] - pop_after_mortality[2] * (1 / ((E_jubilacion - E_initial)/2)) + pop_after_mortality[1] * (1 / ((E_jubilacion - E_initial)/2))
    new_pop[3] = pop_after_mortality[3] + pop_after_mortality[2] * (1 / ((E_jubilacion - E_initial)/2)) - pop_after_mortality[3] * (1 / (esperanza_vida - E_jubilacion))
    #Ajuste poblacion pensionados con tasaa de mortalidad
    #new_pop[3] = pop_after_mortality[3] + pop_after_mortality[2] * (1 / ((E_jubilacion - E_initial)/2))
   
    return new_pop

def update_fund(pop, unemp_rate, inf_rate, inform_rate, avg_salary, A, E_initial, E_jubilacion):
    contributors = (pop[1] + pop[2]) * (1 - unemp_rate) * (1 - inform_rate)  
    retirees = pop[3] * (1 - inform_rate) * (1 - unemp_rate)
   
    income = contributors * A * avg_salary * (1 + inf_rate)
    expenses = retirees * avg_salary * B * (1 + inf_rate)
   
    return income - expenses

# Simulación del modelo
fund_balance = np.zeros(n_years)
fund_balance[0] = 100000000000  # Saldo inicial del fondo

for t in range(1, n_years):
    population_H[t, :] = update_population(population_H[t-1, :], birth_rate, mortality_rates, E_initial, E_jubilacion_H, esperanza_vida)
    fund_balance[t] = fund_balance[t-1] + update_fund(population_H[t, :], unemployment_rate, inflation_rate, informality_rate,average_salary, A, E_initial, E_jubilacion_H)
    population_M[t, :] = update_population(population_M[t-1, :], birth_rate, mortality_rates, E_initial, E_jubilacion_M, esperanza_vida)
    fund_balance[t] += fund_balance[t-1] + update_fund(population_M[t, :], unemployment_rate, inflation_rate, informality_rate,average_salary, A, E_initial, E_jubilacion_M)

# Gráficos
time = np.arange(n_years)

# Gráfico de la población por grupos etarios
plt.figure()
plt.plot(time, (population_H + population_M)/1000000)
plt.xlabel('Años')
plt.ylabel('Población (en millones)')
plt.title('Población por grupos etarios a lo largo del tiempo')
plt.legend(['No cotizantes', 'Cotizantes en edad temprana', 'Cotizantes en edad avanzada', 'Pensionados'])
plt.grid(True)
plt.show()

# Gráfico del saldo del fondo de pensiones
'''plt.figure()
plt.plot(time, fund_balance)
plt.xlabel('Años')
plt.ylabel('Saldo del fondo de pensiones')
plt.title('Saldo del fondo de pensiones a lo largo del tiempo')
plt.grid(True)
plt.show()

'''

# Grafico del saldo del fondo de pensiones en miles de millones de pesos
plt.figure()
plt.plot(time, fund_balance / 1000000000)
plt.xlabel('Años')
plt.ylabel('Saldo del fondo de pensiones (en miles de millones de pesos)')
plt.title('Saldo del fondo de pensiones a lo largo del tiempo')
plt.grid(True)
plt.show()

#analisis de sensibilidad

vecA=np.arange(7,22,1)
vecIndicador=np.zeros(vecA.shape)

r_minimo=E_jubilacion_M*(1-0.10)
r_maximo=E_jubilacion_M*(1+0.11)
r_pasos=E_jubilacion_M*(0.02)
k_minimo=E_jubilacion_H*(1-0.10)
k_maximo=E_jubilacion_H*(1+0.11)
k_pasos=E_jubilacion_H*(0.02)


vecr=np.arange(r_minimo,r_maximo,r_pasos)
veck=np.arange(k_minimo,k_maximo,k_pasos)

vecIndicador = np.zeros((len(vecr),len(veck)))

def model_sensibilidad(vecr,veck):
   
    #recibimos variables
    #N_o=parameter
    #N=(N_o*k*mt.exp(r*t))/(k-N_o+(N_o*mt.exp(r*t)))
    for i in range(len(vecr)):
        for j in range(len(veck)):
            #print(vecIndicador[i,j], simu_temp[-1,0])
           # vecIndicador[i,j] = update_fund(pop, unemp_rate, inf_rate, inform_rate, avg_salary, A, E_initial, E_jubilacion)
            for t in range(1, n_years):
                population_H[t, :] = update_population(population_H[t-1, :], birth_rate, mortality_rates, E_initial, vecr[i], esperanza_vida)
                fund_balance[t] = fund_balance[t-1] + update_fund(population_H[t, :], unemployment_rate, inflation_rate, informality_rate,average_salary, A, E_initial, vecr[i])
                population_M[t, :] = update_population(population_M[t-1, :], birth_rate, mortality_rates, E_initial, veck[j], esperanza_vida)
                fund_balance[t] += fund_balance[t-1] + update_fund(population_M[t, :], unemployment_rate, inflation_rate, informality_rate,average_salary, A, E_initial, veck[j])
                vecIndicador[i,j]=fund_balance[t]
            # Gráficos    
    #Calculamos derivadas
    return vecIndicador

vecIndicador=model_sensibilidad(vecr, veck)

# for i in range(len(vecr)):
#     for j in range(len(veck)):
       
#         simu_temp=odeint(model_sensibilidad, N_o, vecTime, args=(vecr[i],veck[j]))
#         #print(vecIndicador[i,j], simu_temp[-1,0])
#         vecIndicador[i,j] = simu_temp[-1,0]
# library
import seaborn as sns
import pandas as pd
import numpy as np

# Create a dataset

df = pd.DataFrame(vecIndicador, index=list(vecr), columns=list(veck))

# Default heatmap
plt.figure(figsize = (15,15))
p1 = sns.heatmap(df,linewidths=0)

plt.legend("Analisis de sensiblidad de k y r")
plt.xlabel('Sensibilidad k')
plt.ylabel('Sensibilidad r')
plt.show()