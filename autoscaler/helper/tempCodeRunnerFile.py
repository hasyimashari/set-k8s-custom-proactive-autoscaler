
historical_data = [[216528], [217532], [218532], [219406], [220581], [221167], [223328], [225138], [225678], [225892]]
scaler = MinMaxScaler(feature_range=(-1,1))

model = ModelHandler()
prediction_result = model.predict(historical_data, scaler)
print(prediction_result)