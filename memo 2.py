#MEMO 2
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
#OPDRACHT 2.1

# Parameters
k = 32  # Veerconstante (N/m)
m = 0.000001476  # Massa (kg)
original_b = 0.034363  # Originele demping (kg/s)
b_values = [original_b * 0.5, original_b * 2, original_b * 5, original_b * 10]  # Variërende demping (kg/s)
t = np.linspace(0, 0.002, 10000)  # Tijd (s)

# Functie voor de versnelling (afgeleide van snelheid)
def acceleration(x, v, b):
    return (-k * x - b * v) / m

# Simulatie en plotten van de beweging voor elke waarde van b
plt.figure(figsize=(10, 8))

for i, b in enumerate(b_values):
    x = np.zeros_like(t)  # Positie
    v = np.zeros_like(t)  # Snelheid
    x[0] = 0.001  # Initiele positie
    v[0] = 0  # Initiele snelheid

    for j in range(1, len(t)):
        dt = t[j] - t[j - 1]
        a = acceleration(x[j - 1], v[j - 1], b)
        v[j] = v[j - 1] + a * dt
        x[j] = x[j - 1] + v[j] * dt

    plt.subplot(3, 2, i + 1)
    plt.plot(t, x)
    plt.title(f'Demping (b) = {b:.4f} kg/s')
    plt.xlabel('Tijd (s)')
    plt.ylabel('Positie (m)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Originele demping
plt.subplot(3, 2, 5)
x_original = np.zeros_like(t)
v_original = np.zeros_like(t)
x_original[0] = 0.001
v_original[0] = 0

for j in range(1, len(t)):
    dt = t[j] - t[j - 1]
    a = acceleration(x_original[j - 1], v_original[j - 1], original_b)
    v_original[j] = v_original[j - 1] + a * dt
    x_original[j] = x_original[j - 1] + v_original[j] * dt

plt.plot(t, x_original)
plt.title(f'Originele demping (b = {original_b:.4f} kg/s)')
plt.xlabel('Tijd (s)')
plt.ylabel('Positie (m)')
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

plt.tight_layout()
plt.savefig('accelerometer_simulatie.png')
plt.show()


#OPDRACHT 2.3

from scipy.signal import lti, step

# Constanten
m = 0.000001476  # massa in kg
originele_k = 32  # originele veerconstante in N/m
originele_b = 0.034363  # originele dempingscoëfficiënt in kg/s

#tijdinstelling berekenen
def insteltijd(t, y):
    return t[np.argmax(y >= 0.98 * y[-1])]

# Origineel systeem
origineel_systeem = lti([1], [m, originele_b, originele_k])
t_origineel, y_origineel = step(origineel_systeem)
originele_insteltijd = insteltijd(t_origineel, y_origineel)
print("Oorspronkelijke insteltijd:", originele_insteltijd, "seconden")

# Doel (minstens twee keer zo snel)
doel_insteltijd = originele_insteltijd / 2

# Bereik van k- en b-waarden om te testen
k_waarden = np.linspace(originele_k, 4 * originele_k, 10)  # variërend k van origineel tot 4 keer
b_waarden = np.linspace(originele_b, originele_b / 3, 5)  # variërend b van origineel tot ongeveer 1/3


plt.figure(figsize=(9, 5), facecolor='white')

# kleurcodes
kleuren = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Bereken en plot responstijden voor elke (k, b) combinatie
for i, b in enumerate(b_waarden):
    geselecteerde_k = None
    for k in k_waarden:
        systeem = lti([1], [m, b, k])
        t, y = step(systeem)
        it = insteltijd(t, y)
        if it <= doel_insteltijd:
            geselecteerde_k = k
            break
    if geselecteerde_k:
        systeem = lti([1], [m, b, geselecteerde_k])
        t, y = step(systeem)
        plt.plot(t, y, label=f'b={b:.5f}, k={geselecteerde_k:.5f}', color=kleuren[i])


plt.plot(t_origineel, y_origineel, label='Origineel', linestyle='--', color='black')

plt.xlabel('Tijd (s)')
plt.ylabel('Respons')
plt.title('Respons van de versnellingssensor voor verschillende combinaties van k en b')
plt.legend()
plt.grid(True)
plt.show()

#OPDRACHT 2.5

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
    
    for i, tval in enumerate(tijd[:-1]):
        y[i + 1, :] = y[i, :] + (tijd[i + 1] - tijd[i]) * dydt2_func(y[i, :], a2_data[i])
    return y

# Berekenen van de respons van data2
response2 = Form2(dydt2, a2)

# Spanning van de output van de MEMS-accelerometer
voltage_output = 6.83 * 10**(-18)

# Functie om spanning te berekenen uit respons
def calculate_voltage(response, voltage_output):
    return response[:, 0] * voltage_output

# Spanning berekenen voor data1 en data2
voltage1 = calculate_voltage(response1, voltage_output)
voltage2 = calculate_voltage(response2, voltage_output)

# Factor om het spanningssignaal zichtbaar te maken
scaling_factor = 10**18

# Schalen van de spanning voor visualisatie
scaled_voltage1 = voltage1 * scaling_factor
scaled_voltage2 = voltage2 * scaling_factor

# Figuren plotten en opslaan 1 PNG
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8))

# Grafiek plotten voor positie 1 
ax1.set_title('Dataset posities 1: Ideale respons vs spanning output ')
color = '#1f77b4'  # Blauwe kleurcode 
ax1.set_xlabel('Tijd (s)')
ax1.set_ylabel('Versnelling (m/s^2)')
line1, = ax1.plot(tijd, a1, color=color, label="Ideale respons")

ax2 = ax1.twinx()  # Dezelfde x-as verkrijgen
color = '#ff7f0e'  # Oranje kleurcode
ax2.set_ylabel('Spanning (V)')
line2, = ax2.plot(tijd, scaled_voltage1, color=color, label="Spanning output")
ax1.legend([line1, line2], ["Ideale respons", "Spanning output "], loc="upper right")

# Grafiek plotten voor positie 2
ax3.set_title('Dataset posities 2: Ideale respons vs spanning output ')
color = '#1f77b4'  # Blauwe kleurcode
ax3.set_xlabel('Tijd (s)')
ax3.set_ylabel('Versnelling (m/s^2)')
line3, = ax3.plot(tijd, a2, color=color, label="Ideale respons")

ax4 = ax3.twinx()  # Dezelfde x-as verkrijgen
color = '#ff7f0e'  # Oranje kleurcode
ax4.set_ylabel('Spanning (V) ')
line4, = ax4.plot(tijd, scaled_voltage2, color=color, label="Spanning output ")
ax3.legend([line3, line4], ["Ideale respons", "Spanning output "], loc="upper right")


fig.tight_layout()
plt.savefig('versnelling_vs_spanning_output.png')
plt.show()




            









