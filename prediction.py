import joblib
def predict(data):
    lr = joblib.load("model.pkl")
    return lr.predict(data)