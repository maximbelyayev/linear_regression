#!/usr/bin/env python
# coding: utf-8

# In[266]:


import pandas as pd
import numpy as np
import math


# In[267]:


print("Please enter the full filepath of the .csv file that contains variable data:")
file = input()


# In[268]:


data = pd.read_csv(file)
data = data.dropna()

t_score = pd.read_csv("t_score (two-tailed).csv")


# In[269]:


with open(file, 'r') as f:
    lines = f.readlines()
    headers = lines[0].split(",")
    new_headers = [header.replace("\n", "") for header in headers]


# In[270]:


print("Found the following headers in {}:".format(file))
for header in new_headers:
    print(header)
print("\n")

variable_lst = []
responses = ["yes", "y", "no", "n", "quit", "exit"]
yes_no = 'yes'

print("Please enter the name of the dependent variable to be used in the regression:")
dependent = input()
while dependent not in new_headers:
    print("Not a valid entry. Please re-enter the name of the dependent variable to be used in the regression:")
    dependent = input()
variable_lst.append(dependent)

while yes_no in responses[:2]:
    if len(variable_lst) == 1:
        print("Please enter the name of the first explanatory variable to be used in the regression:")
    else:
        print("Please enter the name of the next explanatory variable to be used in the regression:")
    explanatory = input()
    
    while explanatory in variable_lst or explanatory not in new_headers:
        if explanatory in variable_lst:
            print("Variable already included in the regression. Please choose another variable.")
        else:
            print("Not a valid entry. Please carefully re-enter the name of the explanatory variable to be used in the regression:")
        explanatory = input()
    variable_lst.append(explanatory)
    print("\n")
    
    print("Current dependent variable: {}\nCurrent explanatory variable(s): {}".format(dependent, variable_lst[1:]))
    print("\n")
    
    reg_string_lst = []
    reg_string_count = 1
    for explanatory in variable_lst[1:]:
        reg_string_lst.append(" + B{}{}".format(reg_string_count, explanatory))
        reg_string_count +=1
    reg_string = ''.join(reg_string_lst)
    print("Regression equation:\n{} = B0{}".format(dependent, reg_string))
    print("\n")
    
    print("Would you like to add another explanatory variable? (yes/no)")
    yes_no = input().lower()
    print("\n")
    
    while yes_no not in responses:
        print('Not a valid entry. Would you like to add another explanatory variable? Please respond with either "yes" or "no".')
        yes_no = input().lower()
        
print("-" * 100)
print("\n")
        
### CALCULATIONS

y = data[[dependent]]
x = data[data.columns.intersection(variable_lst[1:])]
x.insert(0, 'Intercept', 1)

x_t = x.T
beta = np.linalg.inv(x_t @ x) @ x_t @ y
coef = beta[dependent].tolist()
coef = [round(x, 3) for x in coef]

reg_string_lst = []
reg_string_count = 1
for i in range(1, len(variable_lst)):
    reg_string_lst.append(" + {}{}".format(coef[i], variable_lst[i]))
    reg_string_count +=1
reg_string = ''.join(reg_string_lst)
print("Linear regression:\n{} = {}{}".format(dependent, coef[0], reg_string))
print("\n")


# R-square
yhat = np.matmul(x, np.asarray(beta))
y_bar = data[dependent].mean()
sst = ((data[dependent] - y_bar)**2).sum()
yhat = yhat.squeeze()
sse = ((data[dependent] - yhat.squeeze())**2).sum()
ssr = sst - sse
r_square = ssr/sst
print("R-square:\n{}".format(r_square))
print("\n")


# Confidence Interval (95%)
df = data.shape[0]-(len(variable_lst[1:])+1)
variance = sse/df
c = np.linalg.inv(x_t @ x)

t_critical = t_score["0.05"][df - 1]

standard_error_lst = []
lower_95 = []
upper_95 = []
i = 0

for header in x.columns.values.tolist():
    lower_95.append(coef[i] - (t_critical * math.sqrt(variance * c[i][i])))
    upper_95.append(coef[i] + (t_critical * math.sqrt(variance * c[i][i])))
    i += 1
    
confidence_intervals = list(zip(lower_95, upper_95))

i = 0
for header in x.columns.values.tolist():
    print("95% Confidence Interval for {}:\n{}\n".format(header, confidence_intervals[i]))
    i+= 1


# In[ ]:




