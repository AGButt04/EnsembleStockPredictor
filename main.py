import sys
sys.path.append('src')

from sklearn.metrics import mean_squared_error, r2_score
from src.dataLoader import load_apple_data
from src.featureEngineering import create_features, prepare_model_data
from src.models import train_linear_regression, train_random_forest, train_lstm, evaluate_model, save_model
from src.ensemble import simple_ensemble, evaluate_ensemble


def main():
    print("Starting Apple Stock Predictor...")

    # Load data
    data = load_apple_data()

    # Create features
    processed_data = create_features(data)
    X_train, X_test, y_train, y_test = prepare_model_data(processed_data)

    print(f"Training data shape: {X_train.shape}")

    # Train models
    print("Training Linear Regression...")
    linear_model = train_linear_regression(X_train, y_train)
    linear_results = evaluate_model(linear_model, X_test, y_test)
    print(f"Linear Regression R²: {linear_results['r2']:.4f}")

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test)
    print(f"Random Forest R²: {rf_results['r2']:.4f}")

    print("Training LSTM...")
    lstm_model, lstm_scaler, X_lstm_test, y_lstm_test = train_lstm(processed_data)
    lstm_predictions = lstm_model.predict(X_lstm_test)
    lstm_predictions_unscaled = lstm_scaler.inverse_transform(lstm_predictions)
    y_lstm_unscaled = lstm_scaler.inverse_transform(y_lstm_test)

    # Calculate LSTM metrics manually
    lstm_mse = mean_squared_error(y_lstm_unscaled, lstm_predictions_unscaled)
    lstm_r2 = r2_score(y_lstm_unscaled, lstm_predictions_unscaled)
    print(f"LSTM R²: {lstm_r2:.4f}")

    lstm_results = {
        'predictions': lstm_predictions_unscaled.flatten(),
        'r2': lstm_r2,
        'mse': lstm_mse
    }

    # Create ensemble (only Linear + RF since LSTM has different test set)
    print("Creating ensemble...")
    ensemble_pred = simple_ensemble([linear_results['predictions'], rf_results['predictions']])
    ensemble_results = evaluate_ensemble(y_test, ensemble_pred)
    print(f"Ensemble R²: {ensemble_results['r2']:.4f}")

    # Save models
    print("Saving models...")
    save_model(linear_model, 'models/linear_model.pkl')
    save_model(rf_model, 'models/random_forest_model.pkl')
    lstm_model.save('models/lstm_model.h5')

    print("Training complete!")


if __name__ == "__main__":
    main()