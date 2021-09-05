import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_excel(r'D:\gripinternship\data1.xlsx')
data.head(10)

print(data.corr())

# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

regression = LinearRegression()
regression.fit(train_X, train_y)
print("Model has been trained")



pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction

compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))

