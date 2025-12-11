import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from accuracy_matrix import regression_metrics
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def train_random_forest(
    X_train, y_train,
    X_test, y_test,
    output_dir="Output",
    n_estimators=100,
    random_state=42
):
    os.makedirs(output_dir, exist_ok=True)

    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=10)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute metrics
    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    # Save metrics
    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"])
    metrics_file = os.path.join(output_dir, "regression_metrics.csv")
    metrics_df.to_csv(metrics_file)
    print(f"✅ Metrics saved to {metrics_file}")

    # Feature importances
    feature_importances_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    feat_file = os.path.join(output_dir, "feature_importances.csv")
    feature_importances_df.to_csv(feat_file, index=False)
    print(f"✅ Feature importances saved to {feat_file}")
    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"])

    # Return all needed outputs
    return model, train_metrics, test_metrics, feature_importances_df, y_train_pred, y_test_pred, metrics_df



import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_random_forest_randomized_train_val(
    X_train, y_train,
    X_val, y_val,
    output_dir=".",
    param_distributions=None,
    n_iter=20,         # number of random samples
    cv=5,
    random_state=42
):
    """
    Train and tune Random Forest using RandomizedSearchCV on train+val set.
    """

    os.makedirs(output_dir, exist_ok=True)

    if param_distributions is None:
        param_distributions = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [None, 5, 7, 10, 15],
            "min_samples_split": [2, 5, 7, 10],
            "min_samples_leaf": [1, 2, 3, 5, 7],
            "max_features": ["sqrt"]  # "auto" deprecated
        }

    # Combine train + val for fitting RandomizedSearchCV
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)

    # Initialize Random Forest
    rf = RandomForestRegressor(random_state=random_state, n_jobs=10)

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",  # optimize RMSE
        n_jobs=10,
        #verbose=2,
        random_state=random_state
    )

    print("Running RandomizedSearchCV...")
    random_search.fit(X_combined, y_combined)
    print("Best parameters found:", random_search.best_params_)

    best_model = random_search.best_estimator_
    # --- Print actual tree depths ---
    tree_depths = [est.tree_.max_depth for est in best_model.estimators_]
    print("Actual depths of all trees:", tree_depths)
    print("Average depth:", np.mean(tree_depths))
    print("Max depth:", np.max(tree_depths))

    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    # Compute metrics
    def compute_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {"R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae}

    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_pred)

    # Feature importances
    feature_importances_df = pd.DataFrame({
        "feature": X_train.columns if hasattr(X_train, "columns") else np.arange(X_train.shape[1]),
        "importance": best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    # Save feature importances and best parameters
    feature_importances_df.to_csv(f"{output_dir}/feature_importances.csv", index=False)
    pd.DataFrame([random_search.best_params_]).to_csv(f"{output_dir}/best_params.csv", index=False)

    return best_model, train_metrics, val_metrics, feature_importances_df, y_train_pred, y_val_pred



# # K-Fold CV on train+validation
# # -----------------------------
# def cross_validate_rf(X_train, y_train, X_val, y_val,
#                       n_estimators=300, max_depth=None,
#                       min_samples_split=2, min_samples_leaf=1,
#                       max_features="sqrt", k_folds=1, random_state=42):
#     """
#     Perform k-fold CV on train+validation combined data.
#     """
#     X_combined = pd.concat([X_train, X_val], axis=0)
#     y_combined = pd.concat([y_train, y_val], axis=0)

#     kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
#     fold_metrics = []
#     y_oof = np.zeros(len(y_combined))
#     feature_importances = []

#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_combined)):
#         print(f"Fold {fold+1}/{k_folds}")
#         X_tr, X_val_fold = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
#         y_tr, y_val_fold = y_combined.iloc[train_idx], y_combined.iloc[val_idx]

#         model = RandomForestRegressor(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             max_features=max_features,
#             random_state=random_state,
#             n_jobs=10
#         )
#         model.fit(X_tr, y_tr)

#         y_val_pred = model.predict(X_val_fold)
#         y_tr_pred = model.predict(X_tr)
#         y_oof[val_idx] = y_val_pred

#         # Metrics for this fold
#         train_metrics_fold = regression_metrics(y_tr, y_tr_pred)
#         val_metrics_fold = regression_metrics(y_val_fold, y_val_pred)
#         fold_metrics.append({
#             "fold": fold+1,
#             **{f"train_{k}": v for k, v in train_metrics_fold.items()},
#             **{f"val_{k}": v for k, v in val_metrics_fold.items()}
#         })

#         feature_importances.append(model.feature_importances_)

#     # Average feature importances
#     feature_importances_df = pd.DataFrame(
#         np.mean(feature_importances, axis=0),
#         index=X_combined.columns,
#         columns=['Importance']
#     ).sort_values(by='Importance', ascending=False)

#     metrics_cv_df = pd.DataFrame(fold_metrics)

#     return y_oof, metrics_cv_df, feature_importances_df


# rf cross validation with progress bar
from tqdm import tqdm
import time

def cross_validate_rf(X_train, y_train, X_val, y_val,
                      n_estimators=300, max_depth=54,
                      min_samples_split=2, min_samples_leaf=1,
                      max_features="sqrt", k_folds=1, random_state=42):
    """
    Perform k-fold CV on train+validation combined data with a progress/time bar.
    """
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    fold_metrics = []
    y_oof = np.zeros(len(y_combined))
    feature_importances = []

    # tqdm progress bar for folds
    for fold, (train_idx, val_idx) in tqdm(
        enumerate(kf.split(X_combined), start=1),
        total=k_folds,
        desc="Cross-validation progress",
        ncols=100
    ):
        start_time = time.time()

        X_tr, X_val_fold = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
        y_tr, y_val_fold = y_combined.iloc[train_idx], y_combined.iloc[val_idx]

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=10
        )
        model.fit(X_tr, y_tr)

        y_val_pred = model.predict(X_val_fold)
        y_tr_pred = model.predict(X_tr)
        y_oof[val_idx] = y_val_pred

        # Metrics for this fold
        train_metrics_fold = regression_metrics(y_tr, y_tr_pred)
        val_metrics_fold = regression_metrics(y_val_fold, y_val_pred)
        fold_metrics.append({
            "fold": fold,
            **{f"train_{k}": v for k, v in train_metrics_fold.items()},
            **{f"val_{k}": v for k, v in val_metrics_fold.items()}
        })

        feature_importances.append(model.feature_importances_)

        # Print timing info for this fold
        elapsed = time.time() - start_time
        print(f"✅ Fold {fold}/{k_folds} completed in {elapsed:.1f}s")

    # Average feature importances
    feature_importances_df = pd.DataFrame(
        np.mean(feature_importances, axis=0),
        index=X_combined.columns,
        columns=['Importance']
    ).sort_values(by='Importance', ascending=False)

    metrics_cv_df = pd.DataFrame(fold_metrics)

    return y_oof, metrics_cv_df, feature_importances_df





# -----------------------------
# Train final RF on full train+val and evaluate on test
# -----------------------------

# from tqdm import tqdm
# def train_rf_final(X_train, y_train, X_val, y_val, X_test, y_test,
#                    n_estimators=350, max_depth=54,
#                    min_samples_split=7, min_samples_leaf=2,
#                    max_features="sqrt", output_dir="Output", random_state=42):
#     os.makedirs(output_dir, exist_ok=True)

#     # Combine train+val
#     X_trainval = pd.concat([X_train, X_val], axis=0)
#     y_trainval = pd.concat([y_train, y_val], axis=0)

#     # Train final model
#     model = RandomForestRegressor(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         max_features=max_features,
#         random_state=random_state,
#         n_jobs=10
#     )
#     model.fit(X_trainval, y_trainval)

#     # Predictions
#     y_trainval_pred = model.predict(X_trainval)
#     y_test_pred = model.predict(X_test)

#     # Metrics
#     trainval_metrics = regression_metrics(y_trainval, y_trainval_pred)
#     test_metrics = regression_metrics(y_test, y_test_pred)

#     # Save metrics
#     metrics_df = pd.DataFrame([trainval_metrics, test_metrics], index=["Train+Val", "Test"])
#     metrics_df.to_csv(os.path.join(output_dir, "regression_metrics.csv"))
#     print(f"✅ Metrics saved to {output_dir}/regression_metrics.csv")

#     # Feature importances
#     feature_importances_df = pd.DataFrame({
#         "Feature": X_trainval.columns,
#         "Importance": model.feature_importances_
#     }).sort_values(by='Importance', ascending=False)
#     feature_importances_df.to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)
#     print(f"✅ Feature importances saved to {output_dir}/feature_importances.csv")

#     # Save predictions
#     trainval_with_pred = X_trainval.copy()
#     trainval_with_pred['RF_Predicted'] = y_trainval_pred
#     trainval_with_pred.to_csv(os.path.join(output_dir, "trainval_with_predictions.csv"), index=False)

#     test_with_pred = X_test.copy()
#     test_with_pred['RF_Predicted'] = y_test_pred
#     test_with_pred.to_csv(os.path.join(output_dir, "test_with_predictions.csv"), index=False)

#     return model, trainval_metrics, test_metrics, feature_importances_df, y_trainval_pred, y_test_pred









# from tqdm import tqdm
# import os
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor

# def train_rf_final(X_train, y_train, X_test, y_test,
#                    n_estimators=350, max_depth=54,
#                    min_samples_split=7, min_samples_leaf=2,
#                    max_features="sqrt", output_dir="Output", random_state=42):
    
#     os.makedirs(output_dir, exist_ok=True)

#     # Train model
#     print("⏳ Training Random Forest...")
#     with tqdm(total=1) as pbar:
#         model = RandomForestRegressor(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             max_features=max_features,
#             random_state=random_state,
#             n_jobs=10
#         )
#         model.fit(X_train, y_train)
#         pbar.update(1)

#     # Predictions
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)

#     # Metrics
#     train_metrics = regression_metrics(y_train, y_train_pred)
#     test_metrics = regression_metrics(y_test, y_test_pred)

#     # Save metrics
#     metrics_df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"])
#     metrics_file = os.path.join(output_dir, "regression_metrics.csv")
#     metrics_df.to_csv(metrics_file)
#     print(f"✅ Metrics saved to {metrics_file}")

#     # Feature importances
#     feature_importances_df = pd.DataFrame({
#         "Feature": X_train.columns,
#         "Importance": model.feature_importances_
#     }).sort_values(by="Importance", ascending=False)
#     feat_file = os.path.join(output_dir, "feature_importances.csv")
#     feature_importances_df.to_csv(feat_file, index=False)
#     print(f"✅ Feature importances saved to {feat_file}")

#     # Save predictions
#     train_with_pred = X_train.copy()
#     train_with_pred['RF_Predicted'] = y_train_pred
#     train_with_pred.to_csv(os.path.join(output_dir, "train_with_predictions.csv"), index=False)

#     test_with_pred = X_test.copy()
#     test_with_pred['RF_Predicted'] = y_test_pred
#     test_with_pred.to_csv(os.path.join(output_dir, "test_with_predictions.csv"), index=False)

#     # Return all outputs including metrics_df
#     return model, train_metrics, test_metrics, feature_importances_df, y_train_pred, y_test_pred, metrics_df


# from tqdm import tqdm
# import os
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor

# def train_rf_final(X_train, y_train,
#                               X_test, y_test,
#                               X_test_2020, y_test_2020,
#                               n_estimators=350, max_depth=54,
#                               min_samples_split=7, min_samples_leaf=2,
#                               max_features="sqrt", output_dir="Output", random_state=42):

#     os.makedirs(output_dir, exist_ok=True)

#     print("⏳ Training Random Forest on Train only...")
#     with tqdm(total=1) as pbar:
#         model = RandomForestRegressor(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             max_features=max_features,
#             random_state=random_state,
#             n_jobs=10
#         )
#         model.fit(X_train, y_train)
#         pbar.update(1)

#     # -----------------------------
#     # Predictions
#     # -----------------------------
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
#     y_test_2020_pred = model.predict(X_test_2020)

#     # -----------------------------
#     # Metrics
#     # -----------------------------
#     train_metrics = regression_metrics(y_train, y_train_pred)
#     test_metrics = regression_metrics(y_test, y_test_pred)
#     test2020_metrics = regression_metrics(y_test_2020, y_test_2020_pred)

#     # Save metrics
#     metrics_df = pd.DataFrame(
#         [train_metrics, test_metrics, test2020_metrics],
#         index=["Train", "Test_All", "Test_2020"]
#     )
#     metrics_df.to_csv(os.path.join(output_dir, "regression_metrics.csv"))
#     print(f"✅ Metrics saved to {output_dir}/regression_metrics.csv")

#     # -----------------------------
#     # Feature Importances
#     # -----------------------------
#     feature_importances_df = pd.DataFrame({
#         "Feature": X_train.columns,
#         "Importance": model.feature_importances_
#     }).sort_values(by="Importance", ascending=False)
#     feature_importances_df.to_csv(
#         os.path.join(output_dir, "feature_importances.csv"),
#         index=False
#     )
#     print(f"✅ Feature importances saved to {output_dir}/feature_importances.csv")

#     # -----------------------------
#     # Save predictions
#     # -----------------------------
#     train_with_pred = X_train.copy()
#     train_with_pred["RF_Predicted"] = y_train_pred
#     train_with_pred.to_csv(
#         os.path.join(output_dir, "train_with_predictions.csv"),
#         index=False
#     )

#     test_with_pred = X_test.copy()
#     test_with_pred["RF_Predicted"] = y_test_pred
#     test_with_pred.to_csv(
#         os.path.join(output_dir, "test_with_predictions.csv"),
#         index=False
#     )

#     test2020_with_pred = X_test_2020.copy()
#     test2020_with_pred["RF_Predicted"] = y_test_2020_pred
#     test2020_with_pred.to_csv(
#         os.path.join(output_dir, "test_2020_with_predictions.csv"),
#         index=False
#     )

#     print("✅ Predictions saved for Train, Test, and Test_2020")

#     # -----------------------------
#     # Return everything
#     # -----------------------------
#     return (
#         model,
#         train_metrics,
#         test_metrics,
#         test2020_metrics,
#         feature_importances_df,
#         y_train_pred,
#         y_test_pred,
#         y_test_2020_pred,
#         metrics_df
#     )


from tqdm import tqdm
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_rf_final(X_train, y_train,
                   X_test, y_test,
                   X_test_2020, y_test_2020,
                   train_df=None,
                   test_df=None,
                   test2020_df=None,
                   n_estimators=350, max_depth=54,
                   min_samples_split=7, min_samples_leaf=2,
                   max_features="sqrt", output_dir="Output", random_state=42):
    """
    Train Random Forest, save metrics, feature importance, and predictions with original columns.
    
    Parameters:
    - train_df, test_df, test2020_df: full original DataFrames including columns not in X_train/X_test.
    """

    os.makedirs(output_dir, exist_ok=True)

    print("⏳ Training Random Forest...")
    with tqdm(total=1) as pbar:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=10
        )
        model.fit(X_train, y_train)
        pbar.update(1)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_2020_pred = model.predict(X_test_2020)

    # -----------------------------
    # Metrics
    # -----------------------------
    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)
    test2020_metrics = regression_metrics(y_test_2020, y_test_2020_pred)

    metrics_df = pd.DataFrame(
        [train_metrics, test_metrics, test2020_metrics],
        index=["Train", "Test_All", "Test_2020"]
    )
    metrics_df.to_csv(os.path.join(output_dir, "regression_metrics.csv"))
    print(f"✅ Metrics saved to {output_dir}/regression_metrics.csv")

    # -----------------------------
    # Feature Importances
    # -----------------------------
    feature_importances_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    feature_importances_df.to_csv(
        os.path.join(output_dir, "feature_importances.csv"),
        index=False
    )
    print(f"✅ Feature importances saved to {output_dir}/feature_importances.csv")

    # -----------------------------
    # Save predictions WITH ORIGINAL DATA
    # -----------------------------
    if train_df is not None:
        train_df["RF_Predicted"] = y_train_pred
        train_df.to_csv(os.path.join(output_dir, "train_with_predictions.csv"), index=False)

    if test_df is not None:
        test_df["RF_Predicted"] = y_test_pred
        test_df.to_csv(os.path.join(output_dir, "test_with_predictions.csv"), index=False)

    if test2020_df is not None:
        test2020_df["RF_Predicted"] = y_test_2020_pred
        test2020_df.to_csv(os.path.join(output_dir, "test_2020_with_predictions.csv"), index=False)

    print("✅ Predictions saved for Train, Test, and Test_2020 with original columns")

    # -----------------------------
    # Return everything
    # -----------------------------
    return (
        model,
        train_metrics,
        test_metrics,
        test2020_metrics,
        feature_importances_df,
        y_train_pred,
        y_test_pred,
        y_test_2020_pred,
        metrics_df
    )
