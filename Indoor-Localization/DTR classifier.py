import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

data_frame = pd.read_csv("Filtered_Training_Data.csv") 


input_data = data_frame[['RSSI1', 'RSSI2', 'RSSI3']]
label_x = data_frame[['X']]
label_y = data_frame[['Y']]


# Split data into train and test to verify accuracy after fitting the model. 
input_x_train, input_x_test, label_x_train, label_x_test = train_test_split(input_data, label_x, test_size=0.2, random_state=5, shuffle=True)
input_y_train, input_y_test, label_y_train, label_y_test = train_test_split(input_data, label_y, test_size=0.2, random_state=5, shuffle=True)


# DTR model
DTR_x_model = DecisionTreeRegressor(max_depth=25)
DTR_y_model = DecisionTreeRegressor(max_depth=25)

# Training
DTR_x_model.fit(input_x_train, label_x_train)
DTR_y_model.fit(input_y_train, label_y_train)

# Save model
pickle.dump(DTR_x_model, open("Models/DTR/DTR_x_model.sav", 'wb'))
pickle.dump(DTR_y_model, open("Models/DTR/DTR_y_model.sav", 'wb'))

# Prediction
predict_x_train = DTR_x_model.predict(input_x_train)
predict_x_test = DTR_x_model.predict(input_x_test)

predict_y_train = DTR_y_model.predict(input_y_train)
predict_y_test = DTR_y_model.predict(input_y_test)


# Training and testing accuraciss
print('Training MSE X', mean_squared_error(label_x_train, predict_x_train))
print('Testing MSE X', mean_squared_error(label_x_test, predict_x_test))

print('Training MSE Y', mean_squared_error(label_y_train, predict_y_train))
print('Testing MSE Y', mean_squared_error(label_y_test, predict_y_test))


# Dataset accuracy
x_prediction = DTR_x_model.predict(input_data)
y_prediction = DTR_y_model.predict(input_data)

print('------------------------------------------------------------------------------------------')
print('X MSE: ', mean_squared_error(label_x, x_prediction))
print('Y MSE: ', mean_squared_error(label_y, y_prediction))

print('X r2 ', r2_score(label_x, x_prediction))
print('Y r2 ', r2_score(label_y, y_prediction))

results = {'DTR_prediction_X':list(x_prediction), 'DTR_prediction_Y': list(y_prediction)}
PR_results_df = pd.DataFrame(results, columns=['DTR_prediction_X', 'DTR_prediction_Y'])

PR_results_df.to_csv('DTR_results.csv')