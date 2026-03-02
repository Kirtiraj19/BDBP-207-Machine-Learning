# Implement a linear regression model using scikit-learn for the simulated dataset
#-simulated_data_multiple_linear_regression_for_ML.csv - to predict the “disease_score_fluct”
# from multiple clinical parameters.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1)Load Data Set           #pd.read_csv - it will convert CSV into dataframe(table) because ML models need data in table form.
df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')

X=df.iloc[:,0:5]   #all rows and columns 0-4
y=df.iloc[:,6]     #last column
print(X)           #X is input features
print(y)           #y is output value (disease_score_fluct)
#2) Divide it into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

#print(X_train.shape)
#print(X_test.shape)

def main():

    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)

    #4 Initialization the model
    model = LinearRegression()

    #5 Training the model
    model.fit(X_train_scaled, y_train)

    #6 Test the model
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    print("R2 score for disease_score_fluct:",r2)


    print('Done !')
if __name__ == '__main__':
    main()



