import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df =pd.read_csv("Housing.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

categorical_cols = df.select_dtypes(include='object').columns

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print(df.head())   

x=df.drop("price",axis=1)
y=df['price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("training samples:",len(x_train))
print("testing samples:",len(x_test))

model = LinearRegression()
model.fit(x_train,y_train)
print("Model Trained Successfully!")


y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("R2 Score :",r2)

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.plot(
    [y_test.min(),y_test.max()],
    [y_test.min(),y_test.max()],
    'r--'
)
plt.savefig("scatter.png")
plt.show()

coefficient = pd.DataFrame({
    "Feature":x.columns,
    "Coefficient" : model.coef_
})

print(coefficient.sort_values(by="Coefficient",ascending=False))