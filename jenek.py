import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#task1
print("#"*75, "Задание-1", "#"*75)
df = pd.read_csv('train.csv')
print(df)
statistics = df.groupby(['Sex', 'Pclass'])['Survived'].value_counts().unstack()

print("Статистика погибших/выживших для мужчин и женщин в каждом классе:")
print(statistics)
#task2
print("#"*75, "Задание-2", "#"*75)
statistics_male = df[df['Sex'] == 'male'].describe()
statistics_female = df[df['Sex'] == 'female'].describe()
print("Статистика для мужчин:")
print(statistics_male)
print("\nСтатистика для женщин:")
print(statistics_female)
#task3
print("#"*75, "Задание-3", "#"*75)
survival_by_embarked = df.pivot_table(index='Embarked', values='Survived', aggfunc='mean')
print(survival_by_embarked)
#task4
print("#"*75, "Задание-4", "#"*75)
names = df['Name'].apply(lambda x: x.split(',')[1].split('.')[1].strip())
top_10_names = names.value_counts().head(10)
print("Топ-10 популярных имен:")
print(top_10_names)
surnames = df['Name'].apply(lambda x: x.split(',')[0].strip())
top_10_surnames = surnames.value_counts().head(10)
print("\nТоп-10 популярных фамилий:")
print(top_10_surnames)
#task5
print("#"*75, "Задание-5", "#"*75)
numeric_columns = df.select_dtypes(include=['number'])
df[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.median())
df.to_csv('train_filled.csv', index=False)
print("Отсутствующие значения в числовых столбцах были заполнены медианой и сохранены в файл train_filled.csv.")
#task65
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
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(test_data[features])
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
output.to_csv('predictions.csv', index=False)
print("Предсказания сохранены в файл predictions.csv.")

survived = df[df['Survived'] == 1]['Age'].dropna()
not_survived = df[df['Survived'] == 0]['Age'].dropna()
#task8
print("#"*75, "Задание-8", "#"*75)
# Создание гистограммы
plt.figure(figsize=(10, 6))
plt.hist([survived, not_survived], bins=30, color=['g', 'r'], label=['Выжившие', 'Погибшие'])
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.legend()
plt.title('Гистограмма выживаемости в зависимости от возраста')
plt.show()