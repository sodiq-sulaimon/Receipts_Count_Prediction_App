from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create flask app
app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_date = request.form['startDate']
    end_date = request.form['endDate']

    # Generate pandas dataframe for the dates
    pred_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['date'])

    # Feature Engineering
    pred_df['day_of_week'] = pred_df['date'].dt.dayofweek
    pred_df['month'] = pred_df['date'].dt.month
    pred_df['day_of_month'] = pred_df['date'].dt.day

    # Perform cyclical encoding for the days and month
    pred_df['day_of_week_sin'] = np.sin(2 * np.pi * pred_df['day_of_week'] / 7)
    pred_df['day_of_week_cos'] = np.cos(2 * np.pi * pred_df['day_of_week'] / 7)

    pred_df['day_of_month_sin'] = np.sin(2 * np.pi * pred_df['day_of_month'] / 31)
    pred_df['day_of_month_cos'] = np.cos(2 * np.pi * pred_df['day_of_month'] / 31)

    pred_df['month_sin'] = np.sin(2 * np.pi * pred_df['month'] / 7)
    pred_df['month_cos'] = np.cos(2 * np.pi * pred_df['month'] / 7)

    # Drop original day, month columns
    date = pred_df['date']
    pred_df = pred_df.drop(['date', 'day_of_week', 'day_of_month', 'month'], axis=1)

    # Prediction function
    def predict(X, w, b):
        """
        Predict using linear regression
        :param X(ndarray): training examples with multiple features, shape (n,)
        :param w(ndarray): receipt_prediction parameter, shape (n,)
        :param b(scalar): receipt_prediction parameter
        :returns:
            prediction(scalar): prediction
        """
        prediction = np.dot(X, w) + b
        return prediction

    # Prediction
    # Weights from the trained receipt_prediction
    weights = [-0.00258827, - 0.00103041, - 0.00394448, - 0.00516574, - 0.03378449, - 0.01515997]
    bias = 0.88279

    prediction = predict(X=pred_df, w=weights, b=bias) * 1e7 # 1e7 scales up the prediction to the original scale

    # Pair the predictions with the corresponding dates
    predictions = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    while current_date <= end_date:
        for i in prediction:
            predictions.append({'date': current_date.strftime('%Y-%m-%d'), 'receipt_count': int(i)})
            current_date += timedelta(days=1)

    return render_template('prediction.html', predictions=predictions)

if __name__=='__main__':
    app.run(debug=True)

