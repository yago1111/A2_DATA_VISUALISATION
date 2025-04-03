import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Set page configuration (optional, can set page title, layout, etc.)
st.set_page_config(page_title="ML Model Trainer", layout="wide")

# Define caching functions for data loading
@st.cache_data
def load_sample_data(name: str) -> pd.DataFrame:
    """Load a sample dataset from seaborn by name (cached)."""
    return sns.load_dataset(name)

@st.cache_data
def load_csv_data(file_bytes: bytes) -> pd.DataFrame:
    """Load dataset from uploaded CSV file content (cached)."""
    return pd.read_csv(BytesIO(file_bytes))

# Session state callback to reset selections when data source changes
def reset_selection():
    # Clear stored feature/target selections to avoid using old state with new data
    for key in ['target', 'selected_num', 'selected_cat']:
        if key in st.session_state:
            del st.session_state[key]

# Title and instructions
st.title("Machine Learning Model Trainer")
st.markdown("This app allows you to train ML models (Linear/Logistic Regression and Random Forest) on a chosen dataset. "
            "Select a dataset, choose features and target, configure the model, and click **Train Model** to see performance metrics and plots.")

# 1. Dataset selection
st.subheader("1. Choose Dataset")
data_source = st.radio("Select data source:", ("Sample dataset", "Upload CSV file"), index=0, key="data_source", on_change=reset_selection)
df = None  # initialize dataframe

if data_source == "Sample dataset":
    # Provide a selectbox for sample datasets (common seaborn datasets)
    sample_options = ["iris", "titanic", "penguins", "tips", "flights", "diamonds", "car_crashes", "planets"]
    dataset_name = st.selectbox("Select a sample dataset:", sample_options, key="dataset_name", on_change=reset_selection)
    if dataset_name:
        with st.spinner(f"Loading sample dataset `{dataset_name}`..."):
            df = load_sample_data(dataset_name)
elif data_source == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"], key="uploaded_file", on_change=reset_selection)
    if uploaded_file is not None:
        with st.spinner("Reading uploaded file..."):
            file_bytes = uploaded_file.getvalue()
            df = load_csv_data(file_bytes)

if df is not None:
    # Show basic info about the dataset
    st.write(f"**Dataset loaded.** Shape: {df.shape[0]} rows × {df.shape[1]} columns.")
    with st.expander("Preview dataset"):
        st.dataframe(df.head())

    # 2. Feature and target selection (inside a form to defer execution until submission)
    with st.form("selection_form"):
        st.subheader("2. Select Features and Target")
        # Target variable selection
        target = st.selectbox("Target variable (prediction goal):", options=list(df.columns), key="target")
        # Determine feature lists by data type (quantitative vs qualitative), excluding target
        cols_ex_target = [col for col in df.columns if col != target]
        # Classify columns as numeric or categorical based on dtype and unique values
        numeric_features = []
        categorical_features = []
        for col in cols_ex_target:
            if pd.api.types.is_numeric_dtype(df[col]):
                # If numeric but with few unique values, treat as categorical (e.g., codes like 0/1 or categories encoded as ints)
                if df[col].nunique() <= 15:
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)
        # Widgets for selecting features
        selected_num = st.multiselect("Select quantitative features:", options=numeric_features, default=numeric_features, key="selected_num")
        selected_cat = st.multiselect("Select qualitative features:", options=categorical_features, default=categorical_features, key="selected_cat")
        # Combine selected features
        features = selected_num + selected_cat

        st.subheader("3. Configure Model")
        # Determine whether task is regression or classification based on target variable
        y = df[target]
        # If target is numeric and has many unique values, assume regression; otherwise classification
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
            problem_type = "regression"
        else:
            problem_type = "classification"

        # Model selection based on problem type
        if problem_type == "regression":
            model_choice = st.selectbox("Select model type:", ("Linear Regression", "Random Forest"), key="model_type")
        else:
            model_choice = st.selectbox("Select model type:", ("Logistic Regression", "Random Forest"), key="model_type")

        # Common parameter: test size for train/test split
        test_size = st.slider("Test size (portion of data for testing):", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
        # Model-specific parameters
        fit_intercept = True
        if model_choice in ("Linear Regression", "Logistic Regression"):
            fit_intercept = st.checkbox("Fit intercept (use bias term in model)", value=True)
        n_estimators = None
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of trees in Random Forest:", min_value=50, max_value=500, value=100, step=50)

        # Submit button for form
        submitted = st.form_submit_button("Train Model")

    # Only proceed to model training and evaluation when form is submitted
    if submitted:
        # Basic validation: must have at least one feature selected
        if len(features) == 0:
            st.error("Please select at least one feature for training.")
            st.stop()

        # 4. Model training
        with st.spinner("Training model, please wait..."):
            # Prepare feature matrix X and target vector y
            X = df[features].copy()
            # One-hot encode categorical features
            if len(selected_cat) > 0:
                # Perform one-hot encoding for categorical features (drop first to avoid dummy trap for linear models)
                X = pd.get_dummies(X, columns=selected_cat, drop_first=True)
            y = df[target]
            # Split data into train and test sets
            stratify_param = y if problem_type == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, stratify=stratify_param)

            # Initialize the model based on user selection
            if model_choice == "Linear Regression":
                model = LinearRegression(fit_intercept=fit_intercept)
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(fit_intercept=fit_intercept, max_iter=1000)
            elif model_choice == "Random Forest":
                if problem_type == "regression":
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
                else:
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
            else:
                st.error("Unsupported model choice.")
                st.stop()

            # Train the model
            model.fit(X_train, y_train)

            # Prepare containers for results
            metrics_text = ""  # to accumulate text for metrics
            plots = {}         # to store plot figures

            # 5. Evaluation and metrics
            if problem_type == "classification":
                # Predict on test set
                y_pred = model.predict(X_test)
                # Compute accuracy
                accuracy = accuracy_score(y_test, y_pred)
                metrics_text += f"**Accuracy:** {accuracy*100:.2f}%\n\n"
                # Compute confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                classes = model.classes_  # unique class labels
                # Compute ROC curve data (only if model can estimate probabilities)
                y_score = None
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)
                roc_data = None
                if y_score is not None:
                    # Binarize the test labels for ROC (handles multi-class by one-vs-rest)
                    y_test_binarized = label_binarize(y_test, classes=classes)
                    n_classes = y_test_binarized.shape[1]
                    if n_classes == 2:
                        # Binary classification ROC
                        fpr, tpr, _ = roc_curve(y_test_binarized[:, 1], y_score[:, 1])
                        roc_auc_val = auc(fpr, tpr)
                        metrics_text += f"**AUC (ROC):** {roc_auc_val:.2f}\n\n"
                        roc_data = [(fpr, tpr, f"AUC = {roc_auc_val:.2f}")]
                    else:
                        # Multi-class ROC (one curve per class + micro-average)
                        fpr = {}; tpr = {}; roc_auc = {}
                        for i in range(n_classes):
                            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])
                        # Compute micro-average ROC curve and AUC
                        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
                        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                        metrics_text += f"**Micro-average AUC:** {roc_auc['micro']:.2f}\n\n"
                        # Prepare ROC data for plotting (each class + micro)
                        roc_data = []
                        for i in range(n_classes):
                            roc_data.append((fpr[i], tpr[i], f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})"))
                        roc_data.append((fpr["micro"], tpr["micro"], f"Micro-average (AUC = {roc_auc['micro']:.2f})"))

                # Determine feature importance (for models that support it)
                importances = None
                feature_names = X_train.columns
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    coef = model.coef_
                    # For logistic regression with multiple classes, take mean absolute coefficient
                    if coef.ndim > 1:
                        importances = np.mean(np.abs(coef), axis=0)
                    else:
                        importances = np.abs(coef.ravel())
                # 6. Plotting (Classification)
                # Confusion matrix heatmap
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=classes, yticklabels=classes, ax=ax_cm)
                ax_cm.set_xlabel("Predicted Label")
                ax_cm.set_ylabel("True Label")
                ax_cm.set_title("Confusion Matrix")
                fig_cm.tight_layout()
                plots["Confusion Matrix"] = fig_cm

                # ROC Curve plot (if available)
                if roc_data is not None:
                    fig_roc, ax_roc = plt.subplots()
                    for fpr, tpr, label in roc_data:
                        ax_roc.plot(fpr, tpr, label=label)
                    # Plot chance line
                    ax_roc.plot([0, 1], [0, 1], 'k--', label="Chance")
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.set_title("ROC Curve")
                    ax_roc.legend(loc="lower right")
                    fig_roc.tight_layout()
                    plots["ROC Curve"] = fig_roc

                # Feature importance bar chart
                if importances is not None:
                    # Create a horizontal bar chart of feature importances
                    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
                    fig_imp, ax_imp = plt.subplots()
                    ax_imp.barh(imp_series.index, imp_series.values)
                    ax_imp.set_xlabel("Importance")
                    ax_imp.set_title("Feature Importance")
                    fig_imp.tight_layout()
                    plots["Feature Importance"] = fig_imp

            else:
                # Regression
                # Predict on test set
                y_pred = model.predict(X_test)
                # Compute regression metrics
                r2_val = r2_score(y_test, y_pred)
                rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
                mae_val = mean_absolute_error(y_test, y_pred)
                metrics_text += f"**R-squared:** {r2_val:.2f}\n\n"
                metrics_text += f"**RMSE:** {rmse_val:.2f}\n\n"
                metrics_text += f"**MAE:** {mae_val:.2f}\n\n"
                # Residual distribution data
                residuals = y_test.values - y_pred
                # Feature importance (if available)
                importances = None
                feature_names = X_train.columns
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    coef = model.coef_
                    if coef.ndim > 1:
                        importances = np.mean(np.abs(coef), axis=0)
                    else:
                        importances = np.abs(coef.ravel())
                # 6. Plotting (Regression)
                # Residual distribution plot
                fig_res, ax_res = plt.subplots()
                sns.histplot(residuals, kde=True, ax=ax_res, color='teal')
                ax_res.axvline(0, color='red', linestyle='--')
                ax_res.set_xlabel("Residual (Actual − Predicted)")
                ax_res.set_title("Residuals Distribution")
                fig_res.tight_layout()
                plots["Residuals"] = fig_res

                # Feature importance bar chart
                if importances is not None:
                    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
                    fig_imp, ax_imp = plt.subplots()
                    ax_imp.barh(imp_series.index, imp_series.values, color='gray')
                    ax_imp.set_xlabel("Importance")
                    ax_imp.set_title("Feature Importance")
                    fig_imp.tight_layout()
                    plots["Feature Importance"] = fig_imp

        # 7. Display metrics and plots
        st.subheader("4. Model Performance")
        # Show metrics
        st.markdown(metrics_text)
        # Show plots in tabs for better organization
        if plots:
            tab_names = list(plots.keys())
            tabs = st.tabs(tab_names)
            for i, name in enumerate(tab_names):
                with tabs[i]:
                    st.pyplot(plots[name])
