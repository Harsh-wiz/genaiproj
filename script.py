import os, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

# 1. Path Auto-Detection
train_path, test_path = None, None
for root, dirs, files in os.walk('/kaggle/input'):
    for file in files:
        if file == 'train.csv': train_path = os.path.join(root, file)
        elif file == 'test.csv': test_path = os.path.join(root, file)

train_df = pd.read_csv(train_path or 'train.csv')
test_df = pd.read_csv(test_path or 'test.csv')

signal_cols = [f'sig_{i}' for i in range(250)]

# 2. The Ultimate Feature Extraction (Instance Norm + Windows + FFT + RR)
def build_features(df):
    raw_sigs = df[signal_cols].fillna(0).values
    
    # Remove Patient Leakage (Instance Normalization)
    means = np.mean(raw_sigs, axis=1, keepdims=True)
    stds = np.std(raw_sigs, axis=1, keepdims=True) + 1e-8
    norm_sigs = (raw_sigs - means) / stds
    
    feats = pd.DataFrame(index=df.index)
    
    # Timing (Crucial for Class 1 and Class 3)
    feats['pre_rr'] = df['pre_rr'].fillna(0)
    feats['post_rr'] = df['post_rr'].fillna(0)
    feats['rr_ratio'] = df['rr_ratio'].fillna(0)
    feats['rr_diff'] = feats['post_rr'] - feats['pre_rr']
    feats['rr_sum'] = feats['pre_rr'] + feats['post_rr']
    
    # Spatial Features: Peak Positions
    feats['peak_position'] = np.argmax(norm_sigs, axis=1)
    feats['valley_position'] = np.argmin(norm_sigs, axis=1)
    
    # Windowed Features: Chopping signal into 5 sections to locate abnormalities
    for i in range(5):
        chunk = norm_sigs[:, i*50:(i+1)*50]
        feats[f'win_{i}_mean'] = np.mean(chunk, axis=1)
        feats[f'win_{i}_std'] = np.std(chunk, axis=1)
        feats[f'win_{i}_max'] = np.max(chunk, axis=1)
        feats[f'win_{i}_min'] = np.min(chunk, axis=1)
        
    # First & Second Derivatives (Wave Sharpness)
    diff1 = np.diff(norm_sigs, axis=1)
    diff2 = np.diff(diff1, axis=1)
    feats['diff1_max'] = np.max(diff1, axis=1)
    feats['diff1_min'] = np.min(diff1, axis=1)
    feats['diff2_var'] = np.var(diff2, axis=1)
    
    # FFT (Frequency spectrum for wide Ventricular beats)
    fft_vals = np.abs(np.fft.fft(norm_sigs, axis=1))
    for i in range(1, 20):  # Expanded to top 20 frequencies
        feats[f'fft_{i}'] = fft_vals[:, i]
        
    norm_df = pd.DataFrame(norm_sigs, columns=signal_cols, index=df.index)
    return pd.concat([feats, norm_df], axis=1)

print("Extracting Grandmaster-Level Features...")
X = build_features(train_df)
X_test = build_features(test_df)
y = train_df['label'].values

# Calculate Class Priors for final scaling
class_counts = np.bincount(y)
class_priors = class_counts / len(y)
print(f"Dataset Priors: {class_priors}")

# 3. Stratified 5-Fold Ensemble (CatBoost + LightGBM)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_test), 4))
oof_preds = np.zeros((len(X), 4))

print("\nTraining CatBoost + LightGBM Blend...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    # MODEL 1: CatBoost (Symmetric trees, highly resistant to overfitting)
    cat_model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.04,
        auto_class_weights='Balanced',
        random_seed=42 + fold,
        verbose=False
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    
    # MODEL 2: LightGBM (Depth-wise trees, highly accurate on tabular signals)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        class_weight='balanced',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Blend probabilities uniformly (50% CatBoost + 50% LightGBM)
    cat_val_preds = cat_model.predict_proba(X_val)
    lgb_val_preds = lgb_model.predict_proba(X_val)
    fold_val_preds = (cat_val_preds + lgb_val_preds) / 2.0
    oof_preds[val_idx] = fold_val_preds
    
    cat_test_preds = cat_model.predict_proba(X_test)
    lgb_test_preds = lgb_model.predict_proba(X_test)
    fold_test_preds = (cat_test_preds + lgb_test_preds) / 2.0
    
    test_preds += fold_test_preds / 5.0
    print(f"Fold {fold+1} Blended Successfully.")

# 4. Safe Power-Prior Scaling to Rescue Macro F1
# We divide by prior^0.45. This mathematically lifts Class 3 to visibility 
# without causing the chaotic overfitting of arbitrary threshold tuning.
beta = 0.45 
scaled_oof_preds = oof_preds / (class_priors ** beta)
scaled_test_preds = test_preds / (class_priors ** beta)

final_oof_f1 = f1_score(y, np.argmax(scaled_oof_preds, axis=1), average='macro')
print(f"\n---> ULTIMATE OOF MACRO F1: {final_oof_f1:.4f}")

# 5. Generate Final Submission
final_predictions = np.argmax(scaled_test_preds, axis=1)
submission = pd.DataFrame({'id': test_df['id'], 'label': final_predictions})
submission.to_csv('submission.csv', index=False)
print("Saved final_submission.csv! Commit and submit this to the leaderboard.")
