import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


DEFAULT_DATA_PATH = '/workspaces/PFAS-ML/data/science.ado6638_tables_s2_to_s5_and s9_to_s13.xlsx'


def run_pfas_pipeline(data_path=DEFAULT_DATA_PATH, sheet_name='Table S9', output_dir='outputs', test_size=0.2, seed=42, n_thresholds=101):
    """Train an XGBoost classifier on Table S9 and evaluate metrics across thresholds.

    Saves model and results into ``output_dir``.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Excel data (sheet={sheet_name}) from: {data_path} ...")
    df_obs = pd.read_excel(data_path, sheet_name=sheet_name, skiprows=3)

    # prepare target
    df_obs = df_obs.rename(columns=lambda c: c.strip())
    df_obs = df_obs.dropna(subset=['Observed PFAS detection']).reset_index(drop=True)
    y = df_obs['Observed PFAS detection'].astype(int).values

    # build features (small baseline set)
    features = []
    for col in ['Modeled PFAS 0.315 threshold', 'Modeled PFAS 0.5 threshold']:
        if col in df_obs.columns:
            df_obs[col] = pd.to_numeric(df_obs[col], errors='coerce').fillna(0)
            features.append(col)

    if 'Zip Code' in df_obs.columns:
        df_obs['Zip Code'] = pd.to_numeric(df_obs['Zip Code'], errors='coerce').fillna(0)
        features.append('Zip Code')

    if 'State Abbreviation' in df_obs.columns:
        state_dummies = pd.get_dummies(df_obs['State Abbreviation'].astype(str), prefix='ST')
        df_obs = pd.concat([df_obs, state_dummies], axis=1)
        features.extend(state_dummies.columns.tolist())

    X = df_obs[features].values
    print(f"Constructed features (count={len(features)}): {features}")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y))>1 else None)

    # model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    model_path = os.path.join(output_dir, 'xgb_model.json')
    model.save_model(model_path)
    print(f"Saved model to: {model_path}")

    y_probs = model.predict_proba(X_test)[:, 1]

    # thresholds
    thresholds = np.linspace(0, 1, n_thresholds)
    results = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = f1_score(y_test, y_pred)

        results.append({'threshold': t, 'precision': precision, 'recall': recall, 'f1_score': f1})

    res_df = pd.DataFrame(results)

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['threshold'], res_df['precision'], label='Precision', linewidth=2)
    plt.plot(res_df['threshold'], res_df['recall'], label='Recall', linewidth=2)
    plt.plot(res_df['threshold'], res_df['f1_score'], label='F1 Score', linestyle='--')
    plt.axvline(x=0.75, color='gray', linestyle=':', label='Paper Threshold (0.75)')

    plt.title('Impact of Classification Threshold on PFAS Prediction')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, 'threshold_analysis.png')
    plt.savefig(plot_path)
    print(f"Saved plot to: {plot_path}")

    csv_path = os.path.join(output_dir, 'threshold_results.csv')
    res_df.to_csv(csv_path, index=False)
    print(f"Saved threshold results to: {csv_path}")

    # reporting
    perf_75 = res_df.iloc[(res_df['threshold'] - 0.75).abs().argsort()[:1]]
    print("\n--- Performance at threshold 0.75 ---")
    print(perf_75.to_string(index=False))

    best_row = res_df.loc[res_df['f1_score'].idxmax()]
    print("\n--- Best F1 (threshold) ---")
    print(best_row)

    try:
        auc = roc_auc_score(y_test, y_probs)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PFAS ML pipeline: train and threshold analysis')
    parser.add_argument('--data-path', default=DEFAULT_DATA_PATH, help='Path to the input Excel file')
    parser.add_argument('--sheet-name', default='Table S9', help='Sheet name to read')
    parser.add_argument('--output-dir', default='outputs', help='Directory to save outputs')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split proportion')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--thresholds', type=int, default=101, help='Number of thresholds to sample')

    args = parser.parse_args()
    run_pfas_pipeline(data_path=args.data_path, sheet_name=args.sheet_name, output_dir=args.output_dir, test_size=args.test_size, seed=args.seed, n_thresholds=args.thresholds)