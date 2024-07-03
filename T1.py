import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
data = {
"Age": [30, 25, 40],
"Income": [60000, 45000, 80000],
"Score": [0.75, 0.60, 0.90],
"Gender": ["Male", "Female", "Male"],
"City": ["New York", "Los Angeles", "Chicago"]
}
df = pd.DataFrame(data)
df.drop_duplicates(inplace=True) # Remove duplicates
df.dropna(inplace=True) # Remove rows with missing values
scaler = MinMaxScaler()
df[["Age", "Income", "Score"]] = scaler.fit_transform(df[["Age", "Income", "Score"]])
encoder = OneHotEncoder(sparse=False, drop="first")
encoded_gender = pd.DataFrame(encoder.fit_transform(df[["Gender"]]), columns=["Gender_Male"])
encoded_city = pd.get_dummies(df["City"], prefix="City")
df_encoded = pd.concat([df, encoded_gender, encoded_city], axis=1)
df_encoded.drop(["Gender", "City"], axis=1, inplace=True)
df_encoded["Wealth_Index"] = df_encoded["Age"] * df_encoded["Income"]
X = df_encoded.drop("Score", axis=1)
y = df_encoded["Score"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Preprocessed dataset:")
print(X_train)