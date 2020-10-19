import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing data 
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(25)
Data imported successfully
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
10	7.7	85
11	5.9	62
12	4.5	41
13	3.3	42
14	1.1	17
15	8.9	95
16	2.5	30
17	1.9	24
18	6.1	67
19	7.4	69
20	2.7	30
21	4.8	54
22	3.8	35
23	6.9	76
24	7.8	86
# information abot the data
s_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 464.0 bytes
# information about columns
s_data.columns
Index(['Hours', 'Scores'], dtype='object')
#data shape
s_data.shape
(25, 2)
# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

***from the above digram we can see that there is a positive linear relationship between the two variables

X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
​
print("Training complete.")
​
Training complete.
x = s_data['Hours'].values
y = s_data['Scores'].values
n = len(x)
numerator = 0
denominator = 0
x_mean = np.mean(x)
y_mean = np.mean(y)
for i in range(n):
    numerator = numerator + (x[i]-x_mean)*(y[i]-y_mean)
    denominator = denominator + (x[i]-x_mean)**2
m = numerator/denominator
c = y_mean - (m*x_mean)
print(m)
print(c)
9.775803390787475
2.4836734053731746
line = m*x + c
plt.scatter(x,y)
plt.plot(x,line)
plt.show()

print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 
Actual	Predicted
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033
# main question: if a student study for 9.25 hours what is the predicted Score?
prediction = m*9.25 + c
print("Predicted Score is",prediction)
Predicted Score is 92.90985477015732
