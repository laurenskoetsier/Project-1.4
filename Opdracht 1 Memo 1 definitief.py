import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Locatie van de bestanden posities_1_Team_12 en posities_2_Team_12.
os.chdir("/Users/laurenskoetsier/Desktop/")

# Functie om gegevens in te lezen en te converteren naar zwevendekommagetallen
def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            values = line.strip().split()
            data.append([float(value) for value in values])
        return pd.DataFrame(data, columns=["t", "p"])

# Data inlezen van de twee bestanden. 
data1 = read_data("posities_1_Team_12.txt")
data2 = read_data("posities_2_Team_12.txt")

# Posities en tijden uit de twee bestanden halen.
x1 = data1["p"]
x2 = data2["p"]
tijd = data1["t"]

# Snelheden berekenen uit de posities en de tijd. 
v1 = np.gradient(x1, tijd)
v2 = np.gradient(x2, tijd)

# Versnelling berekenen met behulp van de snelheden en de tijd.
a1 = np.gradient(v1, tijd)
a2 = np.gradient(v2, tijd)

# Parameters uit AccelerometerParameters_Team_12.txt-bestand.
k = 32
m = 0.000001476
c = 0.034363  

# Differentiaalvergelijking voor data1
def dydt1(y, a1_val):
    dy_dt1 = np.array([0., 0])
    dy_dt1[0] = y[1]
    dy_dt1[1] = a1_val - ((c * y[1]) / m) - ((k * y[0]) / m)  # Differentiaal vergelijking
    return dy_dt1

# Functie om de respons van data1 te berekenen
def Form(dydt1_func, a1_data):
    y = np.zeros([len(tijd), 2])
    y[0, :] = np.array([0, 0])
    # Integreren van de differentiaal vergelijking met behulp van Euler's methode
    for i, tval in enumerate(tijd[:-1]):
        y[i + 1, :] = y[i, :] + (tijd[i + 1] - tijd[i]) * dydt1_func(y[i, :], a1_data[i])
    return y

# Berekenen van de respons van data1
response1 = Form(dydt1, a1)

# Differentiaalvergelijking voor data2
def dydt2(y, a2_val):
    dy_dt2 = np.array([0., 0])
    dy_dt2[0] = y[1]
    dy_dt2[1] = a2_val - ((c * y[1]) / m) - ((k * y[0]) / m)  # Differentiaal vergelijking
    return dy_dt2

# Functie om de respons van data2 te berekenen
def Form2(dydt2_func, a2_data):
    y = np.zeros([len(tijd), 2])
    y[0, :] = np.array([0, 0])
    # Integreren van de differentiaal vergelijking met behulp van Euler methode
    for i, tval in enumerate(tijd[:-1]):
        y[i + 1, :] = y[i, :] + (tijd[i + 1] - tijd[i]) * dydt2_func(y[i, :], a2_data[i])
    return y

# Berekenen van de respons van data2
response2 = Form2(dydt2, a2)

# Grafiek plotten voor positie 1 
fig, ax1 = plt.subplots()
plt.title('Dataset posities 1 respons versus ideale respons.')
color = '#1f77b4'  # Blauwe kleurcode 
plt.xlabel('Tijd (s)')
plt.ylabel('Versnelling (m/s)')
line1, = plt.plot(tijd, a1, color=color, label="respons")

ax2 = plt.twinx(ax1)  # Dezelfde x-as verkrijgen
color = '#ff7f0e'  # Oranje kleurcode
plt.ylabel('respons',)
line2, = plt.plot(tijd, response1[:, 0], color=color, label="ideale respons")
fig.tight_layout()
plt.legend([line1, line2], ["respons", "ideale respons"], loc="upper right")

plt.show()

# Grafiek plotten voor positie 2
fig, ax1 = plt.subplots()
plt.title('Dataset posities 2 respons versus ideale respons.')
color = '#1f77b4'  # Blauwe kleurcode.
plt.xlabel('Tijd (s)')
plt.ylabel('Versnelling (m/s)')
line1, = plt.plot(tijd, a2, color=color, label="respons")

ax2 = plt.twinx(ax1)  # Dezelfde x-as verkrijgen
color = '#ff7f0e'  # Oranje kleurcode.
plt.ylabel('respons')
line2, = plt.plot(tijd, response2[:, 0], color=color, label="ideale Respons")
fig.tight_layout()
plt.legend([line1, line2], ["respons", "ideale response"], loc="lower right")

plt.show()


