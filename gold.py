import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

data_path = 'C:/Users/data_17/Desktop/자료/2학년/.김진석 교수님/AI응용기술/Gold Price (2013-2023).csv'
df = pd.read_csv(data_path)

print(df.head())

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date')

df.set_index('Date', inplace=True)

# 쉼포를 제거하고 가격을 float 형으로 변환하여 가격 열 정비
df['Price'] = df['Price'].str.replace(',', '').astype(float)

plt.figure(figsize=(14, 5))
plt.plot(df['Price'])
plt.title('Gold Price (2013-2023)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1,1))

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step -1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 개선된 LSTM 모델 구축
model = Sequential()

# 50개 단위의 첫번째 LSTM 계층 반환 단계
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))

# 20%만큼 탈락층 구성하여 탈락
model.add(Dropout(0, 2))

# 50개 단위의 두번째 LSTM 계층 반환 단계
model.add(LSTM(50, return_sequences=True))

# 20%만큼 탈락층 구성하여 탈락
model.add(Dropout(0.2))

# 반환 단계가 아닌 50개 단위의 세번째 LSTM 계층
model.add(LSTM(50, return_sequences=False))

# 20%만큼 탈락층 구성하여 탈락
model.add(Dropout(0.2))

# 25개 단위의 조밀한 층
model.add(Dense(25))

# 1단위의 출력 레이어(예상가격)
model.add(Dense(1))

# Adan optimizer 와 평균 제곱 오차 손실 함수를 이용하여 모형 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 훈련 데이터로 모델 훈련
model.fit(X_train, y_train, batch_size=32, epochs=50)

# 학습 및 검증데이터 모두에 대한 예측 수행
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 확장된 예측을 원래 가격 수준으로 역변환
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Root Mean Squared Error (RMSE)를 계산하여 모델 성능 평가
train_rmse = np.sqrt(np.mean(((train_predict - scaler.inverse_transform(y_train.reshape(-1, 1))) ** 2)))
test_rmse = np.sqrt(np.mean(((test_predict - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2)))
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, :] = train_predict

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (time_step * 2) + 1 : len(scaled_data) - 1, :] = test_predict

plt.figure(figsize=(14, 5))
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(train_plot,label='Traub Predict')
plt.plot(test_plot, label='Test Predict')
plt.title('Gold Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()