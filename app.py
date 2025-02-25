import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import io
import pickle
import tempfile

# Disable GPU usage by setting CUDA_VISIBLE_DEVICES to -1 before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

# Set page config as the first Streamlit command
st.set_page_config(page_title="AI-Driven Failure Prediction", layout="wide")

# Inform user that CPU is being used
st.sidebar.info("Training neural network on CPU only.")

st.title("AI-Driven Failure Prediction Model")
st.markdown("This application uses machine learning to predict component failures. Upload your dataset (CSV, Excel, or JSON), train a model or upload a pre-trained model, and use live predictions.")

if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = None

st.sidebar.header("Configuration")
dataset_file = st.sidebar.file_uploader("Upload your dataset (CSV, Excel, or JSON)", type=["csv", "xlsx", "xls", "json"], key="dataset_uploader")
model_file = st.sidebar.file_uploader("Upload a pre-trained model (H5 or Pickle)", type=["h5", "pkl"], key="model_uploader")

# Handle dataset upload
if dataset_file is not None:
    file_extension = dataset_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            st.session_state.df = pd.read_csv(dataset_file)
        elif file_extension in ['xlsx', 'xls']:
            st.session_state.df = pd.read_excel(dataset_file)
        elif file_extension == 'json':
            st.session_state.df = pd.read_json(dataset_file)
        else:
            st.sidebar.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
            st.session_state.df = None
        if st.session_state.df is not None:
            st.session_state.df.columns = [re.sub(r'[\[\]<>]', '_', col) for col in st.session_state.df.columns]
            st.sidebar.success("Dataset uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        st.session_state.df = None

# Handle model upload
if model_file is not None:
    try:
        if model_file.name.endswith('.h5'):
            st.session_state.model = load_model(model_file)
            st.session_state.training_complete = True
            st.session_state.model_choice = 'neural network'  # Assume neural network for .h5
            st.sidebar.success("Pre-trained Keras model uploaded successfully!")
        elif model_file.name.endswith('.pkl'):
            st.session_state.model = pickle.load(model_file)
            st.session_state.training_complete = True
            st.session_state.model_choice = 'other'  # Non-Keras model
            st.sidebar.success("Pre-trained scikit-learn/xgboost model uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.session_state.model = None
        st.session_state.training_complete = False
        st.session_state.model_choice = None

def classification_model(model_name, target_column, df, features, custom_params=None):
    if target_column not in df.columns:
        st.error(f"Column '{target_column}' does not exist in the DataFrame.")
        return None
    
    X = df[features]
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == 'logistic regression':
        model = LogisticRegression(**(custom_params if custom_params else {}))
    elif model_name == 'random forest':
        model = RandomForestClassifier(**(custom_params if custom_params else {}))
    elif model_name == 'support vector':
        model = SVC(**(custom_params if custom_params else {}))
    elif model_name == 'k-nearest neighbor':
        model = KNeighborsClassifier(**(custom_params if custom_params else {}))
    elif model_name == 'decision tree':
        model = DecisionTreeClassifier(**(custom_params if custom_params else {}))
    elif model_name == 'xgb':
        model = XGBClassifier(**(custom_params if custom_params else {}))
    elif model_name == 'neural network':
        output_size = len(np.unique(y))
        model = build_neural_network(len(features), output_size, True, custom_params)
        if output_size == 2:
            y_train_encoded = y_train.map({y_train.unique()[0]: 0, y_train.unique()[1]: 1}).values.reshape(-1, 1)
            y_test_encoded = y_test.map({y_test.unique()[0]: 0, y_test.unique()[1]: 1}).values.reshape(-1, 1)
        else:
            y_train_encoded = pd.get_dummies(y_train)
            y_test_encoded = pd.get_dummies(y_test)
        
        epochs = custom_params.get('epochs', 50)
        model.fit(x_train, y_train_encoded, epochs=epochs, batch_size=32, verbose=0, validation_split=0.2)
        st.write(f"Trained neural network with {len(custom_params['hidden_layers'])} hidden layers over {epochs} epochs on CPU.")
        # Evaluate
        pred_proba = model.predict(x_test)
        if output_size == 2:
            pred = (pred_proba > 0.5).astype(int)
        else:
            pred = np.argmax(pred_proba, axis=1)
        accuracy = accuracy_score(y_test_encoded if output_size == 2 else y_test, pred)
        precision = precision_score(y_test_encoded if output_size == 2 else y_test, pred, average='weighted')
        recall = recall_score(y_test_encoded if output_size == 2 else y_test, pred, average='weighted')
        st.write(f'Accuracy for neural network: {accuracy:.2%}')
        st.write(f'Precision for neural network: {precision:.2%}')
        st.write(f"Recall for neural network: {recall:.2%}")
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots()
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
        cmd.plot(ax=ax)
        st.pyplot(fig)
        return model
    else:
        st.error(f"Model '{model_name}' is not supported.")
        return None

    st.write(f"Training {model_name} with parameters: {custom_params if custom_params else 'Default'}")
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')
    st.write(f'Accuracy for {model_name}: {accuracy:.2%}')
    st.write(f'Precision for {model_name}: {precision:.2%}')
    st.write(f"Recall for {model_name}: {recall:.2%}")
    st.write(f"Confusion Matrix for {model_name}:")
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots()
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(ax=ax)
    st.pyplot(fig)
    return model

def try_different_classification(target_column, df, features, param_grid):
    if target_column not in df.columns:
        st.error(f"Column '{target_column}' does not exist in the DataFrame.")
        return None
    
    X = df[features]
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    default_params = {
        'logistic regression': {'C': [0.1, 1, 10]},
        'random forest': {'max_depth': list(range(1,10)), 'n_estimators': [1,10,100]},
        'support vector': {'kernel': ['linear', 'rbf']},
        'k-nearest neighbor': {'n_neighbors': [1,3,5,7,9,11,13,15]},
        'decision tree': {'criterion': ['gini', 'entropy'], 'max_depth': list(range(1,12))},
        'xgb': {'eta': [0.1,0.3,0.5,0.7,0.9], 'max_depth': list(range(1,10)), 'gamma': [0.5,0,1.5]}
    }
    
    models = {
        'logistic regression': LogisticRegression(),
        'random forest': RandomForestClassifier(),
        'support vector': SVC(),
        'k-nearest neighbor': KNeighborsClassifier(),
        'decision tree': DecisionTreeClassifier(),
        'xgb': XGBClassifier()
    }
    
    accuracies = {}
    for name, base_model in models.items():
        params = param_grid.get(name, default_params[name]) if param_grid is not None else default_params[name]
        clf = GridSearchCV(base_model, params, cv=3, n_jobs=-1, error_score='raise')
        st.write(f"Training {name} with parameter grid: {params}")
        with st.spinner(f"Training {name}..."):
            try:
                clf.fit(x_train, y_train)
                model = clf.best_estimator_
                pred = model.predict(x_test)
                accuracies[name] = accuracy_score(y_test, pred)
                st.write(f"Best params for {name}: {clf.best_params_}")
            except Exception as e:
                st.error(f"Failed to train {name}: {str(e)}")
                accuracies[name] = 0
    
    sorted_accuracies = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
    container = st.container()  # Removed border=True
    for name, accuracy in sorted_accuracies:
        container.write(f"{name}: {accuracy:.2%}")
    best_name = sorted_accuracies[0][0]
    best_model = GridSearchCV(models[best_name], param_grid.get(best_name, default_params[best_name]) if param_grid is not None else default_params[best_name], cv=3, n_jobs=-1, error_score='raise')
    best_model.fit(x_train, y_train)
    st.success(f"Best model: {best_name} with accuracy {sorted_accuracies[0][1]:.2%}")
    return best_model.best_estimator_

def regression_model(model_name, target_column, df, features, custom_params=None):
    if target_column not in df.columns:
        st.error(f"Column '{target_column}' does not exist in the DataFrame.")
        return None
    
    X = df[features]
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name == 'linear regression':
        model = LinearRegression(**(custom_params if custom_params else {}))
    elif model_name == 'random forest':
        model = RandomForestRegressor(**(custom_params if custom_params else {}))
    elif model_name == 'lasso':
        model = Lasso(**(custom_params if custom_params else {}))
    elif model_name == 'ridge':
        model = Ridge(**(custom_params if custom_params else {}))
    elif model_name == 'gradient descent':
        model = SGDRegressor(**(custom_params if custom_params else {}))
    elif model_name == 'k-nearest neighbor':
        model = KNeighborsRegressor(**(custom_params if custom_params else {}))
    elif model_name == 'decision tree':
        model = DecisionTreeRegressor(**(custom_params if custom_params else {}))
    elif model_name == 'xgb':
        model = XGBRegressor(**(custom_params if custom_params else {}))
    elif model_name == 'neural network':
        model = build_neural_network(len(features), 1, False, custom_params)
        epochs = custom_params.get('epochs', 50)
        model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_split=0.2)
        st.write(f"Trained neural network with {len(custom_params['hidden_layers'])} hidden layers over {epochs} epochs on CPU.")
        # Evaluate
        pred = model.predict(x_test).flatten()
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        st.write(f"Mean Squared Error for neural network: {mse:.4f}")
        st.write(f"R² Score for neural network: {r2:.4f}")
        residuals = y_test - pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(pred, residuals, color='blue')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title('Residuals vs Predicted')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        st.pyplot(fig)
        return model
    else:
        st.error(f"Model '{model_name}' is not supported.")
        return None
    
    st.write(f"Training {model_name} with parameters: {custom_params if custom_params else 'Default'}")
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    st.write(f"Mean Squared Error for {model_name}: {mse:.4f}")
    st.write(f"R² Score for {model_name}: {r2:.4f}")
    residuals = y_test - pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pred, residuals, color='blue')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Residuals vs Predicted')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    st.pyplot(fig)
    return model

def try_different_regression(target_column, df, features, param_grid):
    if target_column not in df.columns:
        st.error(f"Column '{target_column}' does not exist in the DataFrame.")
        return None
    
    X = df[features]
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    default_params = {
        'linear regression': {'fit_intercept': [True, False]},
        'random forest': {'max_depth': list(range(1,10)), 'n_estimators': [1,10,100]},
        'lasso': {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5]},
        'ridge': {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5]},
        'gradient descent': {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5], 'max_iter': [1,5,10,50,100,200]},
        'k-nearest neighbor': {'n_neighbors': [1,3,5,7,9,11,13,15]},
        'decision tree': {'criterion': ['absolute_error', 'squared_error', 'poisson'], 'max_depth': list(range(1,12))},
        'xgb': {'eta': [0.1,0.3,0.5,0.7,0.9], 'max_depth': list(range(1,10)), 'gamma': [0.5,0,1.5]}
    }
    
    models = {
        'linear regression': LinearRegression(),
        'random forest': RandomForestRegressor(),
        'lasso': Lasso(),
        'ridge': Ridge(),
        'gradient descent': SGDRegressor(),
        'k-nearest neighbor': KNeighborsRegressor(),
        'decision tree': DecisionTreeRegressor(),
        'xgb': XGBRegressor()
    }
    
    mse_scores = {}
    for name, base_model in models.items():
        params = param_grid.get(name, default_params[name]) if param_grid is not None else default_params[name]
        reg = GridSearchCV(base_model, params, cv=3, n_jobs=-1, error_score='raise')
        st.write(f"Training {name} with parameter grid: {params}")
        with st.spinner(f"Training {name}..."):
            try:
                reg.fit(x_train, y_train)
                model = reg.best_estimator_
                pred = model.predict(x_test)
                mse_scores[name] = mean_squared_error(y_test, pred)
                st.write(f"Best params for {name}: {reg.best_params_}")
            except Exception as e:
                st.error(f"Failed to train {name}: {str(e)}")
                mse_scores[name] = float('inf')
    
    sorted_mse = sorted(mse_scores.items(), key=lambda item: item[1])
    container = st.container()  # Removed border=True
    for name, mse in sorted_mse:
        container.write(f"{name}: MSE = {mse:.4f}")
    best_name = sorted_mse[0][0]
    best_model = GridSearchCV(models[best_name], param_grid.get(best_name, default_params[best_name]) if param_grid is not None else default_params[best_name], cv=3, n_jobs=-1, error_score='raise')
    best_model.fit(x_train, y_train)
    pred = best_model.predict(x_test)
    residuals = y_test - pred
    st.success(f"Best model: {best_name} with MSE {sorted_mse[0][1]:.4f}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pred, residuals, color='blue')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Residuals vs Predicted')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    st.pyplot(fig)
    return best_model.best_estimator_

def build_neural_network(input_size, output_size, is_classification, params):
    tf.keras.backend.clear_session()  # Clear previous session to avoid conflicts
    model = Sequential()
    
    # Input layer
    regularizer = None
    if params['hidden_layers'][0]['regularizer_type'] == 'L1':
        regularizer = regularizers.l1(params['hidden_layers'][0]['regularizer_value'])
    elif params['hidden_layers'][0]['regularizer_type'] == 'L2':
        regularizer = regularizers.l2(params['hidden_layers'][0]['regularizer_value'])
    
    model.add(Dense(units=params['hidden_layers'][0]['neurons'],
                    activation=params['hidden_layers'][0]['activation'],
                    input_shape=(input_size,),
                    kernel_regularizer=regularizer))
    
    # Hidden layers
    for layer in params['hidden_layers'][1:]:
        regularizer = None
        if layer['regularizer_type'] == 'L1':
            regularizer = regularizers.l1(layer['regularizer_value'])
        elif layer['regularizer_type'] == 'L2':
            regularizer = regularizers.l2(layer['regularizer_value'])
        
        model.add(Dense(units=layer['neurons'],
                        activation=layer['activation'],
                        kernel_regularizer=regularizer))
    
    # Output layer
    if is_classification:
        if output_size == 2:  # Binary classification
            model.add(Dense(1, activation='sigmoid'))
            loss_fn = 'binary_crossentropy'
        else:  # Multi-class classification
            model.add(Dense(output_size, activation='softmax'))
            loss_fn = 'categorical_crossentropy'
    else:  # Regression
        model.add(Dense(1, activation='linear'))
        loss_fn = 'mse'
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'] if is_classification else ['mse'])
    return model

# Main app logic
if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Feature and Target Selection")
    all_columns = df.columns.tolist()
    features = st.multiselect("Select Features", all_columns, default=all_columns)
    target = st.selectbox("Select Target Variable", all_columns, index=len(all_columns)-1)

    if features and target:
        if target in features:
            features.remove(target)
            st.warning(f"Removed '{target}' from features as it is the target variable.")

        is_categorical = df[target].dtype == 'object' or len(df[target].unique()) < 20
        if is_categorical:
            model_options = ['logistic regression', 'random forest', 'support vector', 'k-nearest neighbor', 'decision tree', 'xgb', 'neural network', 'Try different']
        else:
            model_options = ['linear regression', 'random forest', 'lasso', 'ridge', 'gradient descent', 'k-nearest neighbor', 'decision tree', 'xgb', 'neural network', 'Try different']

        # Only show training options if no model is uploaded
        if not model_file:
            st.subheader("Model Selection")
            model_choice = st.selectbox("Choose Model", model_options)
            st.session_state.model_choice = model_choice  # Store model choice for download logic

            st.subheader("Parameter Settings")
            use_defaults = st.checkbox("Use Default Values", value=True)
            custom_params = None
            param_grid = {}

            if not use_defaults and model_choice != 'Try different':
                if model_choice == 'neural network':
                    num_layers = st.slider("Number of Hidden Layers", 1, 10, 2, key="nn_layers")
                    epochs = st.slider("Number of Epochs", 1, 200, 50, key="nn_epochs")
                    hidden_layers = []
                    for i in range(num_layers):
                        with st.expander(f"Layer {i+1} Settings", expanded=False):
                            neurons = st.slider(f"Neurons in Layer {i+1}", 1, 128, 32, key=f"nn_neurons_{i}")
                            activation = st.selectbox(f"Activation Function for Layer {i+1}", 
                                                    ['relu', 'tanh', 'sigmoid'], 
                                                    key=f"nn_activation_{i}")
                            reg_type = st.selectbox(f"Regularizer Type for Layer {i+1}", 
                                                    ['None', 'L1', 'L2'], 
                                                    key=f"nn_reg_type_{i}")
                            reg_value = 0.0
                            if reg_type != 'None':
                                reg_value = st.slider(f"Regularizer Value for Layer {i+1}", 
                                                    0.0, 0.1, 0.01, step=0.001, 
                                                    key=f"nn_reg_value_{i}")
                            hidden_layers.append({
                                'neurons': neurons,
                                'activation': activation,
                                'regularizer_type': reg_type,
                                'regularizer_value': reg_value
                            })
                    custom_params = {'hidden_layers': hidden_layers, 'epochs': epochs}
                elif model_choice == 'logistic regression':
                    c = st.slider("C (Regularization strength)", 0.1, 10.0, 1.0, key="logistic_c")
                    custom_params = {'C': c}
                elif model_choice == 'random forest':
                    max_depth = st.slider("Max Depth", 1, 20, 5, key="rf_max_depth")
                    n_estimators = st.slider("Number of Estimators", 1, 200, 100, key="rf_n_estimators")
                    custom_params = {'max_depth': max_depth, 'n_estimators': n_estimators}
                elif model_choice == 'support vector':
                    kernel = st.selectbox("Kernel", ['linear', 'rbf'], key="svm_kernel")
                    custom_params = {'kernel': kernel}
                elif model_choice == 'k-nearest neighbor':
                    n_neighbors = st.slider("Number of Neighbors", 1, 15, 5, key="knn_n_neighbors")
                    weights = st.selectbox("Weights", ['uniform', 'distance'], key="knn_weights")
                    custom_params = {'n_neighbors': n_neighbors, 'weights': weights}
                elif model_choice == 'decision tree':
                    criterion = st.selectbox("Criterion", ['gini', 'entropy'] if is_categorical else ['squared_error', 'absolute_error', 'poisson'], key="dt_criterion")
                    max_depth = st.slider("Max Depth", 1, 20, 5, key="dt_max_depth")
                    custom_params = {'criterion': criterion, 'max_depth': max_depth}
                elif model_choice == 'xgb':
                    eta = st.slider("Learning Rate (eta)", 0.1, 1.0, 0.3, step=0.1, key="xgb_eta")
                    max_depth = st.slider("Max Depth", 1, 20, 6, key="xgb_max_depth")
                    gamma = st.slider("Gamma", 0.0, 2.0, 0.0, step=0.5, key="xgb_gamma")
                    custom_params = {'eta': eta, 'max_depth': max_depth, 'gamma': gamma}
                elif model_choice == 'lasso':
                    alpha = st.slider("Alpha", 0.1, 2.0, 1.0, step=0.1, key="lasso_alpha")
                    custom_params = {'alpha': alpha}
                elif model_choice == 'ridge':
                    alpha = st.slider("Alpha", 0.1, 2.0, 1.0, step=0.1, key="ridge_alpha")
                    custom_params = {'alpha': alpha}
                elif model_choice == 'gradient descent':
                    alpha = st.slider("Alpha", 0.1, 2.0, 0.1, step=0.1, key="sgd_alpha")
                    max_iter = st.slider("Max Iterations", 1, 200, 100, key="sgd_max_iter")
                    custom_params = {'alpha': alpha, 'max_iter': max_iter}
                elif model_choice == 'linear regression':
                    fit_intercept = st.checkbox("Fit Intercept", value=True, key="lr_fit_intercept")
                    custom_params = {'fit_intercept': fit_intercept}
            elif use_defaults and model_choice == 'neural network':
                custom_params = {
                    'hidden_layers': [
                        {'neurons': 32, 'activation': 'relu', 'regularizer_type': 'None', 'regularizer_value': 0.0},
                        {'neurons': 32, 'activation': 'relu', 'regularizer_type': 'None', 'regularizer_value': 0.0}
                    ],
                    'epochs': 50
                }

            if st.button("Train Model"):
                with st.spinner("Training the model..."):
                    if is_categorical:
                        if model_choice == 'Try different':
                            st.session_state.model = try_different_classification(target, df, features, param_grid)
                        else:
                            st.session_state.model = classification_model(model_choice, target, df, features, custom_params)
                    else:
                        if model_choice == 'Try different':
                            st.session_state.model = try_different_regression(target, df, features, param_grid)
                        else:
                            st.session_state.model = regression_model(model_choice, target, df, features, custom_params)
                    if st.session_state.model:
                        st.session_state.training_complete = True
                        st.success("Model training completed!")

            # Download button for trained model
            if st.session_state.training_complete and st.session_state.model:
                if st.session_state.model_choice == 'neural network':
                    # Save Keras model to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                        st.session_state.model.save(tmp.name)
                        with open(tmp.name, 'rb') as f:
                            buffer = f.read()
                    os.unlink(tmp.name)  # Clean up temporary file
                    st.download_button(
                        label="Download Trained Model",
                        data=buffer,
                        file_name="trained_model.h5",
                        mime="application/octet-stream"
                    )
                else:
                    # Save scikit-learn/xgboost model using pickle
                    buffer = io.BytesIO()
                    pickle.dump(st.session_state.model, buffer)
                    buffer.seek(0)
                    st.download_button(
                        label="Download Trained Model",
                        data=buffer,
                        file_name="trained_model.pkl",
                        mime="application/octet-stream"
                    )

        # Live Prediction section (available if model is trained or uploaded)
        if st.session_state.training_complete and st.session_state.model:
            st.subheader("Live Prediction")
            st.write("Enter values for a new component to predict:")
            live_input = {}
            for feature in features:
                is_feature_categorical = (df[feature].dtype == 'object' or 
                                        len(df[feature].unique()) < 20)
                if is_feature_categorical:
                    unique_values = df[feature].dropna().unique().tolist()
                    live_input[feature] = st.selectbox(f"{feature}", 
                                                      unique_values, 
                                                      key=f"live_{feature}")
                else:
                    live_input[feature] = st.number_input(f"{feature}", 
                                                         value=0.0, 
                                                         key=f"live_{feature}")
            
            if st.button("Predict"):
                live_data = pd.DataFrame([live_input])
                if hasattr(st.session_state.model, 'predict_proba'):  # scikit-learn/xgboost models
                    prediction = st.session_state.model.predict(live_data)[0]
                    if is_categorical:
                        st.write(f"**Prediction:** {prediction}")
                        prob = st.session_state.model.predict_proba(live_data)[0]
                        st.write(f"**Probabilities:** {dict(zip(st.session_state.model.classes_, prob))}")
                    else:
                        st.write(f"**Prediction:** {prediction:.4f}")
                else:  # Keras neural network
                    pred_proba = st.session_state.model.predict(live_data)
                    output_size = len(np.unique(df[target]))
                    if is_categorical:
                        if output_size == 2:
                            prediction = (pred_proba > 0.5).astype(int)[0][0]
                            classes = np.unique(df[target])
                            st.write(f"**Prediction:** {classes[prediction]}")
                            st.write(f"**Probability:** {pred_proba[0][0]:.4f}")
                        else:
                            prediction = np.argmax(pred_proba, axis=1)[0]
                            classes = np.unique(df[target])
                            st.write(f"**Prediction:** {classes[prediction]}")
                            st.write(f"**Probabilities:** {dict(zip(classes, pred_proba[0]))}")
                    else:
                        prediction = pred_proba[0][0]
                        st.write(f"**Prediction:** {prediction:.4f}")

else:
    st.info("Please upload a dataset (CSV, Excel, or JSON) to begin. Optionally, upload a pre-trained model to skip training.")
    if model_file and not dataset_file:
        st.warning("You have uploaded a model but no dataset. Please upload a dataset to proceed to live predictions.")

st.markdown("---")
st.write("Built with ❤️ by EJUST | Powered by Streamlit")
