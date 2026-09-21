def test_model(model, X_test, y_test):
    predictions = (model.predict(X_test, verbose=0) > 0.5).astype("int32")
    print("Sample Predictions:", predictions[:5].flatten())
    print("True Labels       :", y_test[:5].values)
    return predictions