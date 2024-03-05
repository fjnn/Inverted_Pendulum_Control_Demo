# import sympy as sym

# theta = sym.symbols('theta') 
# A = sym.sin(theta)
# print(A)
# A_numeric = A.subs('theta', 0.1)
# print(A_numeric)


# sin_theta = sym.symbols('s_theta')
# sin_theta_num = 0.1


import numpy as np
import matplotlib.pyplot as plt

# Define the PDFs of two independent random variables
def pdf_X(x):
    # Gaussian (Normal) PDF with mean 0 and standard deviation 1
    return 1/np.sqrt(2*np.pi) * np.exp(-0.5*x**2)

def pdf_Y(y):
    # Uniform PDF over the range [-2, 2]
    return np.where(np.logical_and(y >= -2, y <= 2), 0.25, 0)

# Define the range of values for the random variables
x_values = np.linspace(-5, 5, 1000)
y_values = np.linspace(-5, 5, 1000)

# Compute the PDFs of X and Y
pdf_X_values = pdf_X(x_values)
pdf_Y_values = pdf_Y(y_values)

# Compute the convolution of the PDFs using FFT (Fast Fourier Transform)
convolution = np.fft.ifft(np.fft.fft(pdf_X_values) * np.fft.fft(pdf_Y_values)).real

# Plot the PDFs and their convolution
plt.figure(figsize=(10, 6))
plt.plot(x_values, pdf_X_values, label='PDF of X (Gaussian)')
plt.plot(y_values, pdf_Y_values, label='PDF of Y (Uniform)')
# plt.plot(x_values, convolution[:len(x_values)], label='Convolution (PDF of X + Y)', linestyle='--')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('PDFs and Convolution')
plt.legend()
plt.grid(True)
plt.show()
