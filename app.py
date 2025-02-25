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

st.set_page_config(page_title="AI-Driven Failure Prediction", layout="wide")

st.title("AI-Driven Failure Prediction Model")
st.markdown("This application uses machine learning to predict component failures. Upload your dataset (CSV, Excel, or JSON), select a model, customize parameters, and visualize the predictions. Live Prediction persists after interaction!")

if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV, Excel, or JSON)", type=["csv", "xlsx", "xls", "json"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.sidebar.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
            df = None
        if df is not None:
            df.columns = [re.sub(r'[\[\]<>]', '_', col) for col in df.columns]
            st.sidebar.success("Dataset uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        df = None
else:
    df = None

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
        params = param_grid.get(name, default_params[name])
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
    container = st.container(border=True)
    for name, accuracy in sorted_accuracies:
        container.write(f"{name}: {accuracy:.2%}")
    best_name = sorted_accuracies[0][0]
    best_model = GridSearchCV(models[best_name], param_grid[best_name], cv=3, n_jobs=-1, error_score='raise')
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
        params = param_grid.get(name, default_params[name])
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
    container = st.container(border=True)
    for name, mse in sorted_mse:
        container.write(f"{name}: MSE = {mse:.4f}")
    best_name = sorted_mse[0][0]
    best_model = GridSearchCV(models[best_name], param_grid[best_name], cv=3, n_jobs=-1, error_score='raise')
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

if df is not None:
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
            model_options = ['logistic regression', 'random forest', 'support vector', 'k-nearest neighbor', 'decision tree', 'xgb', 'Try different']
        else:
            model_options = ['linear regression', 'random forest', 'lasso', 'ridge', 'gradient descent', 'k-nearest neighbor', 'decision tree', 'xgb', 'Try different']

        st.subheader("Model Selection")
        model_choice = st.selectbox("Choose Model", model_options)

        st.subheader("Parameter Settings")
        use_defaults = st.checkbox("Use Default Values", value=True)
        custom_params = None
        param_grid = None

        if not use_defaults and model_choice != 'Try different':
            if model_choice == 'logistic regression':
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

        if model_choice == 'Try different':
            st.write("Using GridSearchCV with the following default parameter grids (uncheck 'Use Default Values' to customize ranges):")
            default_params = (
                {
                    'logistic regression': {'C': [0.1, 1, 10]},
                    'random forest': {'max_depth': list(range(1,10)), 'n_estimators': [1,10,100]},
                    'support vector': {'kernel': ['linear', 'rbf']},
                    'k-nearest neighbor': {'n_neighbors': [1,3,5,7,9,11,13,15]},
                    'decision tree': {'criterion': ['gini', 'entropy'], 'max_depth': list(range(1,12))},
                    'xgb': {'eta': [0.1,0.3,0.5,0.7,0.9], 'max_depth': list(range(1,10)), 'gamma': [0.5,0,1.5]}
                } if is_categorical else {
                    'linear regression': {'fit_intercept': [True, False]},
                    'random forest': {'max_depth': list(range(1,10)), 'n_estimators': [1,10,100]},
                    'lasso': {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5]},
                    'ridge': {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5]},
                    'gradient descent': {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5], 'max_iter': [1,5,10,50,100,200]},
                    'k-nearest neighbor': {'n_neighbors': [1,3,5,7,9,11,13,15]},
                    'decision tree': {'criterion': ['absolute_error', 'squared_error', 'poisson'], 'max_depth': list(range(1,12))},
                    'xgb': {'eta': [0.1,0.3,0.5,0.7,0.9], 'max_depth': list(range(1,10)), 'gamma': [0.5,0,1.5]}
                }
            )
            param_grid = default_params
            if not use_defaults:
                st.write("Customize parameter ranges (leave blank to use defaults):")
                for model_name in default_params.keys():
                    with st.expander(f"Parameters for {model_name}", expanded=False):
                        if model_name == 'logistic regression':
                            c_values = st.text_input("C values (comma-separated)", "0.1, 1, 10", key=f"logistic_c_{model_name}")
                            param_grid[model_name] = {'C': [float(x.strip()) for x in c_values.split(',')]}
                        elif model_name == 'random forest':
                            max_depth_min, max_depth_max = st.slider("Max Depth Range", 1, 20, (1, 10), key=f"rf_max_depth_{model_name}")
                            n_estimators_values = st.text_input("N Estimators (comma-separated)", "1, 10, 100", key=f"rf_n_estimators_{model_name}")
                            param_grid[model_name] = {
                                'max_depth': list(range(max_depth_min, max_depth_max + 1)),
                                'n_estimators': [int(x.strip()) for x in n_estimators_values.split(',')]
                            }
                        elif model_name == 'support vector':
                            kernels = st.multiselect("Kernels", ['linear', 'rbf'], default=['linear', 'rbf'], key=f"svm_kernel_{model_name}")
                            param_grid[model_name] = {'kernel': kernels}
                        elif model_name == 'k-nearest neighbor':
                            n_neighbors_min, n_neighbors_max = st.slider("N Neighbors Range", 1, 15, (1, 15), key=f"knn_n_neighbors_{model_name}")
                            param_grid[model_name] = {'n_neighbors': list(range(n_neighbors_min, n_neighbors_max + 1))}
                        elif model_name == 'decision tree':
                            criterion_options = ['gini', 'entropy'] if is_categorical else ['squared_error', 'absolute_error', 'poisson']
                            criteria = st.multiselect("Criterion", criterion_options, default=criterion_options, key=f"dt_criterion_{model_name}")
                            max_depth_min, max_depth_max = st.slider("Max Depth Range", 1, 20, (1, 12), key=f"dt_max_depth_{model_name}")
                            param_grid[model_name] = {
                                'criterion': criteria,
                                'max_depth': list(range(max_depth_min, max_depth_max + 1))
                            }
                        elif model_name == 'xgb':
                            eta_values = st.text_input("Eta values (comma-separated)", "0.1, 0.3, 0.5, 0.7, 0.9", key=f"xgb_eta_{model_name}")
                            max_depth_min, max_depth_max = st.slider("Max Depth Range", 1, 20, (1, 10), key=f"xgb_max_depth_{model_name}")
                            gamma_values = st.text_input("Gamma values (comma-separated)", "0.5, 0, 1.5", key=f"xgb_gamma_{model_name}")
                            param_grid[model_name] = {
                                'eta': [float(x.strip()) for x in eta_values.split(',')],
                                'max_depth': list(range(max_depth_min, max_depth_max + 1)),
                                'gamma': [float(x.strip()) for x in gamma_values.split(',')]
                            }
                        elif model_name == 'lasso':
                            alpha_values = st.text_input("Alpha values (comma-separated)", "0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5", key=f"lasso_alpha_{model_name}")
                            param_grid[model_name] = {'alpha': [float(x.strip()) for x in alpha_values.split(',')]}
                        elif model_name == 'ridge':
                            alpha_values = st.text_input("Alpha values (comma-separated)", "0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5", key=f"ridge_alpha_{model_name}")
                            param_grid[model_name] = {'alpha': [float(x.strip()) for x in alpha_values.split(',')]}
                        elif model_name == 'gradient descent':
                            alpha_values = st.text_input("Alpha values (comma-separated)", "0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5", key=f"sgd_alpha_{model_name}")
                            max_iter_values = st.text_input("Max Iter values (comma-separated)", "1, 5, 10, 50, 100, 200", key=f"sgd_max_iter_{model_name}")
                            param_grid[model_name] = {
                                'alpha': [float(x.strip()) for x in alpha_values.split(',')],
                                'max_iter': [int(x.strip()) for x in max_iter_values.split(',')]
                            }
                        elif model_name == 'linear regression':
                            fit_intercept = st.multiselect("Fit Intercept", [True, False], default=[True, False], key=f"lr_fit_intercept_{model_name}")
                            param_grid[model_name] = {'fit_intercept': fit_intercept}

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

        if st.session_state.training_complete and st.session_state.model:
            st.subheader("Live Prediction")
            st.write("Enter values for a new component to predict:")
            live_input = {}
            for feature in features:
                live_input[feature] = st.number_input(f"{feature}", value=0.0, key=f"live_{feature}")
            
            if st.button("Predict"):
                live_data = pd.DataFrame([live_input])
                prediction = st.session_state.model.predict(live_data)[0]
                if is_categorical:
                    st.write(f"**Prediction:** {prediction}")
                    prob = st.session_state.model.predict_proba(live_data)[0]
                    st.write(f"**Probabilities:** {dict(zip(st.session_state.model.classes_, prob))}")
                else:
                    st.write(f"**Prediction:** {prediction:.4f}")

else:
    st.info("Please upload a CSV, Excel, or JSON file to begin. The dataset should contain features and a target column.")

st.markdown("---")
st.write("Built with ❤️ by EJUST | Powered by Streamlit")