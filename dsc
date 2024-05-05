import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import csv

data = pd.read_csv('Age_Income.csv')

summary_stats = data.groupby('Age')['Income'].describe()
print(summary_stats)

age_numeric_encoding = {
    'Young': 1,
    'Middle Age': 2,
    'Old': 3
}

data['Age_numeric'] = data['Age'].map(age_numeric_encoding)

income_np = np.array(data['Income'])
income_df = pd.Series(data['Income'])

#Formula
mean_income = sum(data['Income'])/len(data['Income'])
print(mean_income)

#statistics library function
mean_income=statistics.mean(data['Income'])
print(mean_income)

#  Using NumPy function
mean_income = np.mean(income_np)
print(mean_income)

#Using Pandas function
mean_income = income_df.mean()
print(mean_income)


#formula
n = len(data['Income'])
if n % 2:
    median_income = sorted(data['Income'])[round(0.5*(n-1))]
else:
    x_ord, index = sorted(data['Income']), round(0.5 * n)
    median_income = 0.5 * (x_ord[index-1] + x_ord[index])

print(median_income)


# Using Statistics Library function
median_income = statistics.median(data['Income'])
print(median_income)

# Using NumPy function
median_income = np.median(income_np)
print(median_income)

# Using Pandas function
median_income = income_df.median()
print(median_income)


# Finding the mode without any library
income_counts = {}
for item in data['Income']:
    if item in income_counts:
        income_counts[item] += 1
    else:
        income_counts[item] = 1

mode_income = max(income_counts, key=income_counts.get)
print(mode_income)

# Using Statistics Library function
mode_income = statistics.mode(data['Income'])
print(mode_income)



# Calculating Variance using Formula (without libraries)
n = len(scores)
score_mean = sum(scores) / n
score_var = sum((item - score_mean)**2 for item in scores) / (n - 1)
print(score_var)

# Finding Variance using Libraries
# Using Statistics library function
score_var = statistics.variance(scores)
print(score_var)

#std without library
income_std = income_var**0.5
print(income_std)

# Using Statistics library function
income_std = statistics.stdev(data['Income'])
print(income_std)

# Using NumPy library function
income_std = np.std(income_np, ddof=1) #Here the ddof stands for delta degrees of freedom. This parameter allows the proper calculation of 𝑠², with (𝑛 − 1) in the denominator instead of 𝑛.
print(income_std)

# Using Pandas Library function
income_std = income_df.std(ddof=1)
print(income_std)

# Calculating Skewness using formula (without libraries)
n = len(data['Income'])
income_mean = sum(data['Income']) / n
income_var = sum((item - income_mean)**2 for item in data['Income']) / (n - 1)
income_std = income_var ** 0.5
income_skew = (sum((item - income_mean)**3 for item in data['Income'])* n / ((n - 1) * (n - 2) * income_std**3))
print(income_skew)

# Using Scipy library function
income_skew=scipy.stats.skew(income_np, bias=False) #Here the parameter bias is set to False to enable the corrections for statistical bias.
print(income_skew)
# Using Pandas Library function
income_df.skew()
print(income_skew)

norm = income_df
norm.plot(kind = 'density')
print('This distribution has skew', norm.skew())
print('This distribution has kurtosis', norm.kurt())





IRIS

iris = pd.read_csv('Iris.csv')
iris['Species'].unique()
# Function to calculate mean
def calculate_mean(data):
    return sum(data) / len(data)

# Function to calculate mode
def calculate_mode(data):
    frequency_dict = {}
    for value in data:
        frequency_dict[value] = frequency_dict.get(value, 0) + 1
    mode_frequency = max(frequency_dict.values())
    mode = [key for key, value in frequency_dict.items() if value == mode_frequency]
    return mode

# Function to calculate median
def calculate_median(data):
    sorted_data = sorted(data)
    n = len(data)
    if n % 2 == 0:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        return sorted_data[n // 2]


# Function to calculate variance
def calculate_variance(data):
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance

# Function to calculate standard deviation
def calculate_standard_deviation(data):
    variance = calculate_variance(data)
    return variance ** 0.5

# Function to calculate skewness
def calculate_skewness(data):
    mean = calculate_mean(data)
    std_dev = calculate_standard_deviation(data)
    skewness = sum((x - mean) ** 3 for x in data) / (len(data) * std_dev ** 3)
    return skewness

iris.columns

# Define a function to calculate statistics for a given column
def calculate_statistics(data):
    mean = calculate_mean(data)
    mode = calculate_mode(data)
    median = calculate_median(data)
    variance = calculate_variance(data)
    std_dev = calculate_standard_deviation(data)
    skewness = calculate_skewness(data)
    return mean, mode, median, variance, std_dev, skewness

# Filter the dataset for 'Iris-setosa' and 'Iris-versicolor'
filtered_data = iris[(iris['Species'] == 'Iris-setosa') | (iris['Species'] == 'Iris-versicolor')]

# Get data for each column except 'Id' and 'Species'
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for column in columns:
    column_data = filtered_data[column].tolist()
    mean, mode, median, variance, std_dev, skewness = calculate_statistics(column_data)
    print(f"Statistics for {column}:", "\nMean:", mean, "\nMode:", mode, "\nMedian:", median, "\nVariance:", variance, "\nStandard Deviation:", std_dev, "\nSkewness:", skewness, "\n")
