4. მოცემულია IT საგნებზე სტუდენთა ცხრილი:

name_of_class	students

Python	33
Statistics	27
Machine Learning	25
Data Science	39
Big Data	32


გამოსახეთ bar დიაგრამით, დაიტანეთ y ღერზე Number of students, დიაგრამას გაუკეთეთ სათაურად Subjects enrolled by students და შენი სახელი და გვარი.
ამოხსნა:

import matplotlib.pyplot as plt    
name_of_class = ['Python', 'Statistics', 'Machine Learning', 'Data Science', 'Big Data']  
students = [33,27,25,39,32]  
plt.bar(name_of_class,students)  
plt.ylabel('Number of students')  
plt.title('Subjects enrolled by students\nAleksandre Chakhvadze')  
plt.show()


5. მოცემულია 2 სტუდენტის საგნების შედეგების ცხრილი:
		Jack	John
Maths	95	85
Statistics  	85	82
Python	74	64
Data Science	75	70
English	80	82
გამოსახეთ bar დიაგრამით, დაიტანეთ y ღერზე Scores, დიაგრამას გაუკეთეთ სათაურად Scores by Students და Exam ML. Jack-ის სვეტის ფერი იყოს ლურჯი, ხოლო John-ის წითელი. სვეტის სიგანე იყოს 0.4 .
ამოხსნა:

import  numpy as np  
import  matplotlib.pyplot as plt  
scores_Jack = ( 95 ,  85 ,  74 ,  75 ,  80 )  
scores_John = ( 85 ,  82 ,  64 ,  70 ,  82 )  
fig, ax = plt.subplots ()  
indexes = np.arange (len (scores_Jack))  
bar_width =  0.4  
data1 = plt.bar (indexes, scores_Jack, bar_width, color = 'b' , label = 'Jack' )  
data2 = plt.bar (indexes + bar_width, scores_John, bar_width, color = 'r' , label = 'John' )  
plt.ylabel ( 'Scores' )  
plt.title ( 'Scores by Students\nExam ML' )  
plt.xticks (indexes + bar_width / 2 , ( 'Maths' ,  'Statistics' ,  'Python' ,  'Data Science' ,  'English' ))  
plt.legend ()  
plt.tight_layout ()  
plt.show ()  


6. მოცემულია 3 სტუდენტის საგნების შედეგების ცხრილი:
		Jack	John	Lado
Maths			95	85	80 
Statistics		85	82	81 
Python			74	64	74 
Data Science	75	70	74 
English			80	82	88

გამოსახეთ bar დიაგრამით, დაიტანეთ y ღერზე Scores, დიაგრამას გაუკეთეთ სათაურად Scores by Students და Exam ML BTU. Jack-ის სვეტის ფერი იყოს ლურჯი, John-ის წითელი, ხოლო Lado-სი კი ყვითელი. სვეტის სიგანე იყოს 0.3 .
ამოხსნა:

import  numpy as np  
import  matplotlib.pyplot as plt  
scores_Jack = ( 95 ,  85 ,  74 ,  75 ,  80 )  
scores_John = ( 85 ,  82 ,  64 ,  70 ,  82 )
scores_Lado = ( 80 ,  81 ,  74 ,  74 ,  88 )
fig, ax = plt.subplots ()  
indexes = np.arange (len (scores_Jack))  
bar_width =  0.3  
data1 = plt.bar (indexes, scores_Jack, bar_width, color = 'b' , label = 'Jack' )  
data2 = plt.bar (indexes + bar_width, scores_John, bar_width, color = 'r' , label = 'John' )
data3 = plt.bar (indexes + 2*bar_width, scores_Lado, bar_width, color = 'y' , label = 'Lado' )
plt.ylabel ( 'Scores' )  
plt.title ( 'Scores by Students\nExam ML BTU' )  
plt.xticks (indexes + bar_width  , ( 'Maths' ,  'Statistics' ,  'Python' ,  'Data Science' ,  'English' ))  
plt.legend ()  
plt.tight_layout ()  
plt.show () 

7. მოცემულია ცხრილში ამ თვეში გაყიდული ამტომანქანები:

cars	numbers_cars
FORD	13
TESLA	26
JAGUAR	39
AUDI	13
BMW	39
MERCEDES	78

გამოსახეთ pie დიაგრამით პროცენტული წილების მითითებით. დიაგრამის ზომა იყოს (4;4).
ამოხსნა:

from matplotlib import pyplot as plt   
import numpy as np   
cars = ['FORD', 'TESLA', 'JAGUAR','AUDI', 'BMW',  'MERCEDES']   
numbers_cars = [13, 26, 39, 13, 39, 78]   
fig = plt.figure(figsize =(4, 4))   
plt.pie(numbers_cars, labels=cars, autopct='%1.1f%%') 
plt.title ( 'Sales_BTU' )    
plt.show()

8. მოცემულია ცხრილში ინფლაციის მაჩვენებელელი წლების განმავლობაში:

Year	inflation_rate
2000	2.8
2001	3.2
2002	4 
2003	3.7
2004	1.2 
2005	6.9
2006	7
2007	6.5
2008	6.23
2009	4.5

გამოსახეთ plot დიაგრამით. დაიტანეთ x ღერძზე Year, დაიტანეთ y ღერძზე Inflation Rate, გაუკეთეთ სათაური Inflation Rate Vs Year. ფონტის ზომები იყოს 14. ჩანდეს უკანაფონზე ბადე. წირის ფერი იყოს წითელი. 
ამოხსნა:

import matplotlib.pyplot as plt  
Year = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]  
inflation_rate = [2.8, 3.2, 4, 3.7, 1.2, 6.9, 7, 6.5, 6.23, 4.5]  
plt.plot(Year, inflation_rate, color='red', marker='o')  
plt.title('Inflation Rate Vs Year', fontsize=14)  
plt.xlabel('Year', fontsize=14)  
plt.ylabel('Inflation Rate', fontsize=14)  
plt.grid(True)  
plt.show()

9. შემთხვევითი 1000 რიცხვის საშუალებით ააგეთ განაწილებითი ჰისტოგრამა. ფერი წითელი, ფონტის ზომა 12,  დიაგრამის ზომა (8;6). დაიტანეთ x ღერძზე Value, დაიტანეთ y ღერძზე Frequency, გაუკეთეთ სათაური Normal Distribution Histogram.
ამოხსნა:

import numpy as np  
import matplotlib.pyplot as plt
randomNumbers = np.random.normal(size=1000)    
plt.figure(figsize=[8,6])  
plt.hist(randomNumbers, width = 0.5, color='r',alpha=1)  
plt.grid(axis='y', alpha=0.5)  
plt.xlabel('Value',fontsize=12)  
plt.ylabel('Frequency',fontsize=12)  
plt.title('Normal Distribution Histogram',fontsize=12)  
plt.show()

10. დაწერეთ კოდი რომელიც შექმნის seaborn-ის dataset diamonds-ზე დაყრდნობით შემდეგ მოდელებს: წრფივი რეგრესია, k უახლოესი მეზობლის რეგრესია, random forest და svm.
დამოკიდებულ ცვლადად აიღეთ ფასი, ხოლო სხვა დანარჩენი კი დამოუკიდებელ ფაქტორებად. მოახდინეთ კატეგორიული სვეტების გადაყვანა რიცხვითში. დაყავით მოდელი 80% სატრენინგოდ დანარჩენი 20% კი სატესტოდ. მოახდინეთ მონაცემების მაშტაბირება და დაასტანდარტულეთ.
მიღებული მოდელები შეაფასეთ შემდეგი მეტრიკების მეშვეობით R2, საშუალო აბსოლუტური გადახრა, საშუალო კვადრატული გადახრა და ფესვი საშუალო კვადრატული გადახრიდან.
ამოხსნა:
#general for all models
import pandas as pd
import numpy as np
import seaborn as sns
diamonds_df = sns.load_dataset("diamonds")
X = diamonds_df.drop(['price'], axis=1)
y = diamonds_df["price"]
numerical = X.drop(['cut', 'color', 'clarity'], axis = 1)
categorical = X.filter(['cut', 'color', 'clarity'])
cat_numerical = pd.get_dummies(categorical,drop_first=True)
X = pd.concat([numerical, cat_numerical], axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.20, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

# LinearRegression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
regressor = lin_reg.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics
print('LinearRegression:')
print('R^2:', metrics.r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# KNeighborsRegressor Model
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
regressor = knn_reg.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics
print('KNeighborsRegressor:')
print('R^2:', metrics.r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# RandomForestRegressor Model
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=42, n_estimators=500)
regressor = rf_reg.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


from sklearn import metrics

print('RandomForestRegressor:')
print('R^2:', metrics.r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# SVR Model
from sklearn import svm
svm_reg = svm.SVR()
regressor = svm_reg.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics
print('SVM Regressor:')
print('R^2:', metrics.r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


11. დაწერეთ კოდი რომელიც შექმნის seaborn-ის dataset diamonds-ზე დაყრდნობით შემდეგ წრფივი რეგრესიის მოდელი.
დამოკიდებულ ცვლადად აიღეთ ფასი, ხოლო სხვა დანარჩენი კი დამოუკიდებელ ფაქტორებად. მოახდინეთ კატეგორიული სვეტების გადაყვანა რიცხვითში. დაყავით მოდელი 80% სატრენინგოდ დანარჩენი 20% კი სატესტოდ. მოახდინეთ მონაცემების მასშტაბირება და დაასტანდარტულეთ. მოახდინეთ K fold-ირება.
ამოხსნა:
import pandas as pd
import numpy as np
import seaborn as sns
diamonds_df = sns.load_dataset("diamonds")
X = diamonds_df.drop(['price'], axis=1)
y = diamonds_df["price"]
numerical = X.drop(['cut', 'color', 'clarity'], axis = 1)
categorical = X.filter(['cut', 'color', 'clarity'])
cat_numerical = pd.get_dummies(categorical,drop_first=True)
X = pd.concat([numerical, cat_numerical], axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.20, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
regressor = lin_reg.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn import metrics
print('LinearRegression:')
print('R^2:', metrics.r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.model_selection import cross_val_score
print(cross_val_score(regressor, X, y, cv=5, scoring ="neg_mean_absolute_error"))

12. ამ საკითხში იქნება ასაგები ორფაქტორიანი მოდელი შესწავლილი რომელიმე linear ალგორითმის მიხედვით. ყურადღებით დააკვირდით რომელი ალგორითმით გთხოვენ გამოანგარიშებას ფინალურის ბილეთის ვარიანტში. აქ ყველაა წარმოდგენილი!!!!

მოცემული ცხრილის მიხედვით ააგეთ ორ ფაქტორიანი წრფივი მოდელი
X1	X2	Y 
0.1	0.5	3.7
0.5	0.2	4.2
1.2	2.4	7.9
3.1	4.2	12.9
გამოიანგარიშეთ დეტერმინაციის კოეფიციენტი და ასევე მოდელის ყველა კოეფიციენტი.
ა) მოცემული ცხრილის მიხედვით ააგეთ ორფაქტორიანი მოდელი სტანდარტული წრფივი ალგორითმით.
ამოხსნა:

#sadziebelia aseti funqcia Y=k0+k1x1+k2x2
import numpy  as np
from sklearn.linear_model import LinearRegression
x =np.array([
  [0.1,0.5],[0.5,0.2],[1.2,2.4],[3.1,4.2]
])
y =np.array([3.7,4.2,7.9,12.9])
model = LinearRegression()
model.fit(x,y)
r_sq =model.score(x,y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred =model.predict(x)
print('predicted y is :' , y_pred)

შედეგი:
coefficient of determination: 0.9993731429272054
intercept: 3.0889785870324564
slope: [1.6198727  1.15160403]
predicted y is : [ 3.82676787  4.12923574  7.7966755  12.94732088]

ბ) მოცემული ცხრილის მიხედვით ააგეთ ორფაქტორიანი მოდელი Ridge ალგორითმით თუ ალფა იქნება 1.4
  
ამოხსნა:
import numpy  as np
from sklearn.linear_model import Ridge
x =np.array([
  [0.1,0.5],[0.5,0.2],[1.2,2.4],[3.1,4.2]
])
y =np.array([3.7,4.2,7.9,12.9])

model = Ridge(alpha=1.4)
model.fit(x,y)
r_sq =model.score(x,y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred =model.predict(x)
print('predicted y is :' , y_pred)

გ) მოცემული ცხრილის მიხედვით ააგეთ ორფაქტორიანი მოდელი Lasso ალგორითმით თუ ალფა იქნება 0.5
  
ამოხსნა:
import numpy  as np
from sklearn.linear_model import Lasso
x =np.array([
  [0.1,0.5],[0.5,0.2],[1.2,2.4],[3.1,4.2]
])
y =np.array([3.7,4.2,7.9,12.9])

model =Lasso(alpha=0.5)
model.fit(x,y)
r_sq =model.score(x,y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred =model.predict(x)
print('predicted y is :' , y_pred)

დ) მოცემული ცხრილის მიხედვით ააგეთ ორფაქტორიანი მოდელი ElasticNet ალგორითმით თუ ალფა იქნება 1.0, ხოლო l1_ratio იქნება 0.5
  
ამოხსნა:
import numpy  as np
from sklearn.linear_model import ElasticNet
x =np.array([
  [0.1,0.5],[0.5,0.2],[1.2,2.4],[3.1,4.2]
])
y =np.array([3.7,4.2,7.9,12.9])

model =ElasticNet(alpha=1.0,l1_ratio=0.5)
model.fit(x,y)
r_sq =model.score(x,y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred =model.predict(x)
print('predicted y is :' , y_pred)

მე13 საკითხში აქ statmodels ბიბლიოთეკის გამოყენებით მოდელის აგება და შეფასებაა. გამოცდაზე შემფასებელი პარამეტრები და რეპორტი ეს სიტყვები მიგანიშნებს რო ეს გჭირდება.

13. ვთქვათ მოცემული გაქვსთ შემდეგი მასივები ააგეთ უმცირეს კვადრატთა მეთოდით წრფივი რეგრესია და გამოიტანეთ მოდელის შემფასებელი პარამეტრები და რეპორტი. 
x =np.array([
  [0.1,0.5],[0.5,0.2],[1.2,2.4],[3.1,4.2],[4.4,5.3],[1.3,4.5],[2.3,5.6],
[7.7,7.9]
])
y =np.array([3.7,4.2,7.9,12.9,14.8,19.0,21.6,34.5])

ამოხსნა:
import numpy  as np
import statsmodels.api as sm
x =np.array([
  [0.1,0.5],[0.5,0.2],[1.2,2.4],[3.1,4.2],[4.4,5.3],[1.3,4.5],[2.3,5.6],[7.7,7.9]
])
y =np.array([3.7,4.2,7.9,12.9,14.8,19.0,21.6,34.5])
x =sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
