import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

##########################################################################
#1
train_file = pd.read_csv("train.csv");
print('#' * 100 + "\n\t\t\t\t\t1st-task")
group_object = train_file.groupby(['Pclass', 'Sex'])
res_first = group_object['Survived'].value_counts().unstack()

print('статистика погибших/выживших')
print(res_first)
##########################################################################
#2
print('#' * 100 + "\n\t\t\t\t\t2st-task")
statistic_male = train_file[train_file['Sex'] == 'male'].describe()
statistic_female = train_file[train_file['Sex'] == 'female'].describe()
print("\t\t\t\t\t\t\t\tMale")
print(statistic_male)
print("\t\t\t\t\t\t\t\tFemale")
print(statistic_female)
##########################################################################
#3
print('#' * 100 + "\n\t\t\t\t\t3st-task")
groups_by_Embarked = train_file.groupby(['Embarked'])
res_third = groups_by_Embarked['Survived'].value_counts().unstack()

res = res_third['Percentage_of_survived'] = res_third[1] / (res_third[0] + res_third[1]) * 100
print(res_third)
##########################################################################
#4
print('#' * 100 + "\n\t\t\t\t\t4st-task")
names = train_file["Name"]
surname = names.apply(lambda x: x.split(',')[0])
first_names = names.apply(lambda x: x.split(',')[1].split('.')[1])
print('*' * 100 + '\n' + "\t\t\t\t\tSurname top")
print(surname.value_counts().iloc[0:10])
print('*' * 100 + '\n' + "\t\t\t\t\tFirst name top")
print(first_names.value_counts().iloc[0:10])
##########################################################################
#5
print('#' * 100 + "\n\t\t\t\t\t5st-task")
numeric_columns = train_file.select_dtypes(include='number')
for colum_name in numeric_columns:
    train_file[colum_name].fillna(train_file[colum_name].median)
train_file.to_csv("train_filled", index=False);
print("test_filled is ready")
##########################################################################
#6
print("#"*75, "Задание-6", "#"*75)
test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')
numeric_columns = train_data.select_dtypes(include=['number'])
train_data[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.median())
test_data[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.median())
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = train_data[features]
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(test_data[features])
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
output.to_csv('predictions.csv', index=False)
print("Предсказания сохранены в файл predictions.csv.")
