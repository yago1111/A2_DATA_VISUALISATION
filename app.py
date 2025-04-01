import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    roc_auc_score,
)
import pickle

st.title("ML Model Trainer App")

# -------------------------------
# Dataset Loading Section
# -------------------------------
st.sidebar.header("Dataset Options")
dataset_source = st.sidebar.radio("Select Dataset Source", ["Seaborn Dataset", "Upload CSV"])

df = None
if dataset_source == "Seaborn Dataset":
    # Provide a list of available Seaborn datasets
    available_datasets = ["iris", "tips", "titanic", "diamonds"]
    dataset_name = st.sidebar.selectbox("Select a Seaborn dataset", available_datasets)

    @st.cache_data
    def load_seaborn_dataset(name):
        return sns.load_dataset(name)

    df = load_seaborn_dataset(dataset_name)
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

if df is not None:
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write("Shape:", df.shape)

    # -------------------------------
    # Feature and Target Selection
    # -------------------------------
    st.subheader("Feature Selection")
    all_columns = df.columns.tolist()
    target_column = st.selectbox("Select Target Variable", all_columns)
    feature_columns = st.multiselect("Select Feature Variables", [col for col in all_columns if col != target_column])
    if not feature_columns:
        st.error("Please select at least one feature variable.")
else:
    st.info("Awaiting dataset selection or file upload.")

# Proceed only if dataset is available and features have been selected.
if df is not None and feature_columns:
    # -------------------------------
    # Task and Model Selection
    # -------------------------------
    task = st.selectbox("Select Task Type", ["Regression", "Classification"])

    if task == "Regression":
        model_name = st.selectbox("Select Regression Model", ["Linear Regression", "Random Forest Regressor"])
    else:
        model_name = st.selectbox("Select Classification Model", ["Logistic Regression", "Random Forest Classifier"])

    # -------------------------------
    # Model Training Form
    # -------------------------------
    with st.form(key="model_training_form"):
        st.subheader("Model Parameter Configuration")
        test_size = st.slider("Test Size (proportion for test set)", 0.1, 0.5, value=0.3, step=0.05)
        
        # Model-specific parameters
        if model_name in ["Random Forest Regressor", "Random Forest Classifier"]:
            n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=500, value=100, step=10)
            max_depth = st.number_input("Max Depth (0 for None)", min_value=0, max_value=100, value=0, step=1)
        elif model_name == "Logistic Regression":
            C_param = st.number_input("Inverse Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01, format="%.2f")
        # For Linear Regression, no extra parameters are needed.

        submitted = st.form_submit_button("Fit Model")

    if submitted:
        # -------------------------------
        # Data Preparation
        # -------------------------------
        X = df[feature_columns]
        y = df[target_column]

        # Convert categorical features using one-hot encoding
        X = pd.get_dummies(X, drop_first=True)

        # For classification tasks, encode target if needed.
        if task == "Classification":
            if y.dtype == "object" or str(y.dtype).startswith("category"):
                y, _ = pd.factorize(y)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # -------------------------------
        # Model Training
        # -------------------------------
        if task == "Regression":
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest Regressor":
                model = RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=(None if max_depth == 0 else int(max_depth)),
                    random_state=42,
                )
        else:
            if model_name == "Logistic Regression":
                model = LogisticRegression(C=C_param, max_iter=1000, random_state=42)
            elif model_name == "Random Forest Classifier":
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=(None if max_depth == 0 else int(max_depth)),
                    random_state=42,
                )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success("Model training complete!")

        # Save model in session state for potential later use or download.
        st.session_state["model"] = model

        # -------------------------------
        # Model Evaluation & Visualizations
        # -------------------------------
        st.subheader("Model Performance Metrics")
        if task == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write("Mean Squared Error:", mse)
            st.write("R² Score:", r2)

            # Residual distribution plot
            residuals = y_test - y_pred
            fig, ax = plt.subplots()
            ax.hist(residuals, bins=20)
            ax.set_title("Residual Distribution")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            acc = accuracy_score(y_test, y_pred)
            st.write("Accuracy:", acc)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax_cm.figure.colorbar(im, ax=ax_cm)
            ax_cm.set(title="Confusion Matrix", xlabel="Predicted Label", ylabel="True Label")
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_cm.text(j, i, format(cm[i, j], "d"),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
            st.pyplot(fig_cm)

            # ROC Curve (only for binary classification)
            if len(np.unique(y_test)) == 2:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = roc_auc_score(y_test, y_prob)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                ax_roc.plot([0, 1], [0, 1], "k--")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("Receiver Operating Characteristic")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
            else:
                st.info("ROC Curve is available only for binary classification.")

        # -------------------------------
        # Feature Importance Visualization
        # -------------------------------
        st.subheader("Feature Importance")
        importance = None
        feature_names = X.columns
        # For linear models: use coefficients
        if hasattr(model, "coef_"):
            coef = model.coef_
            if task == "Regression" or (task == "Classification" and coef.ndim == 1):
                importance = coef
            else:
                # For multiclass classification, average the absolute coefficients
                importance = np.mean(np.abs(coef), axis=0)
        # For tree-based models: use feature_importances_
        elif hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

        if importance is not None:
            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
            imp_df = imp_df.sort_values(by="Importance", ascending=False)
            st.write(imp_df)
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(imp_df["Feature"], imp_df["Importance"])
            ax_imp.set_title("Feature Importance")
            ax_imp.set_xlabel("Feature")
            ax_imp.set_ylabel("Importance")
            plt.xticks(rotation=45)
            st.pyplot(fig_imp)
        else:
            st.info("The selected model does not provide feature importance.")

        # -------------------------------
        # Model Export/Download
        # -------------------------------
        st.subheader("Download Trained Model")
        model_bytes = pickle.dumps(model)
        st.download_button("Download Model", model_bytes, file_name="trained_model.pkl")


