import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('D:/Code_MetnumReg/Student_Performance (1).csv')

# Selecting relevant columns
data = data[['Sample Question Papers Practiced', 'Performance Index']].dropna()

# Rename columns for clarity
data.columns = ['Jumlah_latihan_soal', 'Nilai_ujian']

# Splitting data into train and test sets
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Linear regression model
X_train = train_data['Jumlah_latihan_soal'].values.reshape(-1, 1)
y_train = train_data['Nilai_ujian'].values
X_test = test_data['Jumlah_latihan_soal'].values.reshape(-1, 1)
y_test = test_data['Nilai_ujian'].values

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_train_pred = linear_model.predict(X_train)
linear_test_pred = linear_model.predict(X_test)

# Calculate RMS error for linear regression
linear_train_mse = mean_squared_error(y_train, linear_train_pred)
linear_train_rms_error = np.sqrt(linear_train_mse)
linear_test_mse = mean_squared_error(y_test, linear_test_pred)
linear_test_rms_error = np.sqrt(linear_test_mse)

# Simple power model (power regression)
X_train_power = np.log1p(X_train)
y_train_power = np.log1p(y_train)
X_test_power = np.log1p(X_test)
y_test_power = np.log1p(y_test)

power_model = LinearRegression()
power_model.fit(X_train_power, y_train_power)
power_train_pred = np.expm1(power_model.predict(X_train_power))
power_test_pred = np.expm1(power_model.predict(X_test_power))

# Calculate RMS error for power regression
power_train_mse = mean_squared_error(y_train, power_train_pred)
power_train_rms_error = np.sqrt(power_train_mse)
power_test_mse = mean_squared_error(y_test, power_test_pred)
power_test_rms_error = np.sqrt(power_test_mse)

# Plotting linear regression
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(train_data['Jumlah_latihan_soal'], train_data['Nilai_ujian'], color='blue', label='Training data')
plt.scatter(test_data['Jumlah_latihan_soal'], test_data['Nilai_ujian'], color='orange', label='Test data')
plt.plot(train_data['Jumlah_latihan_soal'], linear_train_pred, color='red', label=f'Training regression (Train RMS Error: {linear_train_rms_error:.2f}, Test RMS Error: {linear_test_rms_error:.2f})')
plt.plot(test_data['Jumlah_latihan_soal'], linear_test_pred, color='green')
plt.xlabel('Jumlah Latihan Soal')
plt.ylabel('Nilai Ujian')
plt.title('Regresi Linear')
plt.legend()

# Plotting power regression
plt.subplot(1, 2, 2)
plt.scatter(train_data['Jumlah_latihan_soal'], train_data['Nilai_ujian'], color='blue', label='Training data')
plt.scatter(test_data['Jumlah_latihan_soal'], test_data['Nilai_ujian'], color='orange', label='Test data')
plt.plot(train_data['Jumlah_latihan_soal'], power_train_pred, color='red', label=f'Training regression (Train RMS Error: {power_train_rms_error:.2f}, Test RMS Error: {power_test_rms_error:.2f})')
plt.plot(test_data['Jumlah_latihan_soal'], power_test_pred, color='green')
plt.xlabel('Jumlah Latihan Soal')
plt.ylabel('Nilai Ujian')
plt.title('Regresi Pangkat Sederhana')
plt.legend()

# Show plots
plt.tight_layout()
plt.show()
