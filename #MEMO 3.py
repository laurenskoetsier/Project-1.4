#MEMO 3 codes 

# @laurens Koetsier 


#---PUNT-4------------------------------------------------------------------------------------------------------

# code voor het maken van de x,t-grafiek van Massabeweging onder Ocillerende kracht F(t) + ingezoomde grafiek tijdsinterval 0.0-0.01s.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

# Gegeven parameters uit bestand
k = 0.3820  # N/m
m = 4.3e-9  # kg
b = 8.1053e-7  # kg/s
Fmax = 60e-9  # N

# Resonantiefrequentie
omega_0 = np.sqrt(k / m)
f_0 = omega_0 / (2 * np.pi)  # Hz
print(f'Resonantiefrequentie (f_0): {f_0:.2f} Hz')

# Tijd array
tmax = 0.1  
num_steps = 25000  
t = np.linspace(0, tmax, num_steps)

# Oscillerende kracht bij resonantie
def driving_force(t):
    return Fmax * np.cos(omega_0 * t)

# Differentiaalvergelijking
def mass_spring_damper(t, y):
    x, v = y
    dxdt = v
    dvdt = (driving_force(t) - b * v - k * x) / m
    return [dxdt, dvdt]

# Beginvoorwaarden: x(0) = 0, v(0) = 0
y0 = [0, 0]

# Oplossen differentiaalvergelijking
sol = solve_ivp(mass_spring_damper, [0, tmax], y0, t_eval=t, method='RK45')

# Resultaten plotten
fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

# grafiek plotten
ax.plot(sol.t, sol.y[0])
ax.set_title('x,t-grafiek van massabeweging onder oscillerende kracht F(t)')
ax.set_xlabel('t (s)')
ax.set_ylabel('x (m)')
ax.grid(True)

# Ingezoomde plot buiten de plot
zoom_start = 0.0
zoom_end = 0.01
ax_zoom.plot(sol.t, sol.y[0])
ax_zoom.set_xlim(zoom_start, zoom_end)
ax_zoom.set_title(f'Ingezoomd: {zoom_start}s - {zoom_end}s')
ax_zoom.set_xlabel('t (s)')
ax_zoom.set_ylabel(' x (m)')
ax_zoom.grid(True)

# Verbindingslijnen ingezoomde plot
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from matplotlib.patches import ConnectionPatch

# Ingezoomd gebied in de plot markeren
ax.axvline(x=zoom_start, color='navy', linestyle='--')
ax.axvline(x=zoom_end, color='navy', linestyle='--')

# Verbindingslijnen toevoegen tussen de originele en ingezoomde plot
xy1 = (zoom_start, 0)
xy2 = (zoom_start, 0.6)  
con = ConnectionPatch(xyA=xy1, coordsA=ax.transData, xyB=(0, 0), coordsB=ax_zoom.transAxes, color='orange')
fig.add_artist(con)

xy1 = (zoom_end, 0)
xy2 = (zoom_end, 0.6)  
con = ConnectionPatch(xyA=xy1, coordsA=ax.transData, xyB=(1, 0), coordsB=ax_zoom.transAxes, color='orange')
fig.add_artist(con)

plt.tight_layout()
plt.show()

#---PUNT-5------------------------------------------------------------------------------------------------------

#De waarde van de amplitude  wordt  in de terminal geprint

amplitude_interval = (sol.t >= 0.09) & (sol.t <= 0.1)
amplitude = np.max(np.abs(sol.y[0][amplitude_interval]))
print(f"Amplitude na stabilisatie: {amplitude:.6e} m")

#---PUNT-6------------------------------------------------------------------------------------------------------
# code voor het maken van de tuning-kromme.

# Parameters
k = 0.3820  # N/m
m = 4.3e-9  # kg
Q = 50.1  # Q-factor

# Berekeningen
omega0 = np.sqrt(k / m)
b = omega0 / Q
Fmax = 60e-9  # N

# Frequentiebereik in Hz berekenen voor x-as 
f0 = omega0 / (2 * np.pi)
f = np.linspace(f0 * 0.9, f0 * 1.1, 1000)
omega = 2 * np.pi * f

# Berekening amplitude
A = (Fmax / m) / np.sqrt((omega0**2 - omega**2)**2 + (2 * b * omega)**2)

# Bereken van de maximale amplitude en halve amplitude
maximale_amplitude = max(A)
halve_amplitude = maximale_amplitude / 2

# Eigenschappen van de grafiek (hoe het eruit moet komen te zien qua maat, kleuren, etc...)
plt.figure(figsize=(10, 6))
plot_amplitude, = plt.plot(f, A, label='Tuning-kromme')
plt.scatter(f[np.argmax(A)], maximale_amplitude, color='black', label='Maximale Amplitude', zorder=5)
plt.axhline(y=halve_amplitude, color='g', linestyle='--', label='Halve Maximale Amplitude')

# Bijschrift voor de maximale amplitude
plt.annotate(f'Maximale Amplitude: {maximale_amplitude:.2e} m', 
             xy=(f[np.argmax(A)], maximale_amplitude), 
             xytext=(1435, 3.47e-6),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='right', verticalalignment='top')

# Bijschrift voor de halve amplitude
plt.annotate(f'Halve Amplitude: {halve_amplitude:.2e} m',
             xy=(1450, 1e-6),
             xytext=(1450, 1.1e-6),
             horizontalalignment='center', verticalalignment='top', color='green')

plt.annotate('', xy=(1500, halve_amplitude), xytext=(1450, 1.1e-6),
             arrowprops=dict(arrowstyle='->', color='green'))

# Berekening van FWHM
indices = np.where(A >= halve_amplitude)[0]
FWHM = f[indices[-1]] - f[indices[0]]

# Bijschrift voor FWHM
plt.annotate('FWHM', 
             xy=(1500, 2.2e-6),
             xytext=(1500, 2.01e-6 * 1.05),
             color='blue', fontsize=10,
             horizontalalignment='center', verticalalignment='bottom', 
             arrowprops=dict(arrowstyle='-', color='blue'))

plt.annotate('', xy=(f[indices[0]], halve_amplitude), xytext=(f[indices[-1]], halve_amplitude),
             arrowprops=dict(arrowstyle='<->', color='blue'))

plt.axvline(x=f[indices[0]], color='blue', linestyle='--', label=f'FWHM = {FWHM:.2f} Hz')
plt.axvline(x=f[indices[-1]], color='blue', linestyle='--')

# Bijschrift voor FWHM Start en Eind
plt.annotate(f'FWHM Start: {f[indices[0]]:.2f} Hz', 
             xy=(1448, 2.00e-6), 
             xytext=(1400, 2.220e-6),
             arrowprops=dict(facecolor='blue', arrowstyle='->'),
             horizontalalignment='right', verticalalignment='bottom', color='blue')

plt.annotate(f'FWHM Eind: {f[indices[-1]]:.2f} Hz', 
             xy=(1549.6, 2.00e-6), 
             xytext=(1600, 2.220e-6),
             arrowprops=dict(facecolor='blue', arrowstyle='->'),
             horizontalalignment='left', verticalalignment='bottom', color='blue')

# bijschrift voor resonantiefrequentie
plt.axvline(x=f0, color='steelblue', label=f'Resonantiefrequentie $f_0$ = {f0:.2f} Hz')
plt.text(1501, 1.96e-6, 'f0', color='steelblue', verticalalignment='top', horizontalalignment='left')

# Titels assen
plt.title('Tuning-kromme')
plt.xlabel('Frequentie (Hz)')
plt.ylabel('Amplitude (m)')
plt.legend()

plt.grid(True)
plt.tight_layout()  
plt.show()

#---PUNT-8------------------------------------------------------------------------------------------------------
# code voor de variatie van de b waarde, zelf *0.80 of *1.20 invullen bij b.

# Gegeven parameters uit bestand
k = 0.3820  # N/m
m = 4.3e-9  # kg
b = 8.1053e-7*(1.20) # kg/s
Fmax = 60e-9  # N
x_statisch = 1.5712e-7  # m

# Resonantiefrequentie
omega_0 = np.sqrt(k / m)

# Tijd array
tmax = 0.1  
num_steps = 25000  
t = np.linspace(0, tmax, num_steps)

# Oscillerende kracht bij resonantie
def driving_force(t):
    return Fmax * np.cos(omega_0 * t)

# Differentiaalvergelijking
def mass_spring_damper(t, y):
    x, v = y
    dxdt = v
    dvdt = (driving_force(t) - b * v - k * x) / m
    return [dxdt, dvdt]

# Beginvoorwaarden: x(0) = 0, v(0) = 0
y0 = [0, 0]

# Oplossen differentiaalvergelijking
sol = solve_ivp(mass_spring_damper, [0, tmax], y0, t_eval=t, method='RK45')

# Amplitude in het tijdsinterval 0,09-0,1s
amplitude_interval = (sol.t >= 0.09) & (sol.t <= 0.1)
amplitude = np.max(np.abs(sol.y[0][amplitude_interval]))
print(f"Amplitude in het tijdsinterval 0,09-0,1s: {amplitude:.6e} m")

# Berekening van de Q-factor
Q = amplitude / x_statisch
print(f"Q-factor: {Q:.2f}")

# Bijgewerkte b-waarde met de berekende Q-factor
b = omega_0 / Q

# Frequentiebereik in Hz berekenen voor x-as
f0 = omega_0 / (2 * np.pi)
f = np.linspace(f0 * 0.9, f0 * 1.1, 1000)
omega = 2 * np.pi * f

# Berekening amplitude
A = (Fmax / m) / np.sqrt((omega_0**2 - omega**2)**2 + (2 * b * omega)**2)

# Berekening van de maximale amplitude en halve amplitude
maximale_amplitude = max(A)
halve_amplitude = maximale_amplitude / 2

# Berekening van FWHM
indices = np.where(A >= halve_amplitude)[0]
FWHM = f[indices[-1]] - f[indices[0]]

# Waardes worden geprint in terminal
print("FWHM:", FWHM, "Hz")
