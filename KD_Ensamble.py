import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

def convert_gfr_to_stages(gfr_values):
    """
    convert GFR values to CKD stages based on the medical guidelines that follows:
    Stage 1: GFR ≥ 90 (Normal/High)
    Stage 2: GFR 60-89 (Mild decrease)
    Stage 3: GFR 30-59 (Moderate decrease)
    Stage 4: GFR 15-29 (Severe decrease)
    Stage 5: GFR < 15 (Kidney failure)
    """
    stages = pd.cut(gfr_values, 
                    bins=[0, 15, 30, 60, 90, float('inf')],
                    labels=['Stage 5', 'Stage 4', 'Stage 3', 'Stage 2', 'Stage 1'],
                    ordered=True)
    return stages

def load_and_preprocess_classification(data):
    # drop leakage columns, there are no missing values in the dataset so no imputation needed

    drop_cols = ['GFR', 'CKD_Status'] # these are features that causes data leakage (cheating)
    
    dx = data.drop(columns=[col for col in drop_cols if col in data.columns])
    
    dy = convert_gfr_to_stages(data['GFR'])
    
    print(f"\nClass distribution:")
    print(dy.value_counts().sort_index())
    
    # encode categoricals
    dx = pd.get_dummies(dx, drop_first=True)
    
    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dx)
    
    return scaler, X_scaled, dy, dx.columns

def train_random_forest(X_train, y_train):
    # train Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    # train Gradient Boosting classifier
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        random_state=42,
        subsample=0.8
    )
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    # train Support Vector Machine classifier
    model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        random_state=42,
        class_weight='balanced',
        probability=True  # Enable probability estimates
    )
    model.fit(X_train, y_train)
    return model

def plot_confusion_matrix_detailed(y_test, y_pred, title="Confusion Matrix", filename='confusion_matrix.png'):
    # detailed confusion matrix with counts and percentages

    classes = ['Stage 5', 'Stage 4', 'Stage 3', 'Stage 2', 'Stage 1']

    # calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                ax=axes[0], cbar_kws={'label': 'Count'}
    )
    axes[0].set_title(f'{title} - Counts')
    axes[0].set_ylabel('True Stage')
    axes[0].set_xlabel('Predicted Stage')
    
    # plot 2: Percentages (normalized by true label)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'}
    )
    axes[1].set_title(f'{title} - Percentages')
    axes[1].set_ylabel('True Stage')
    axes[1].set_xlabel('Predicted Stage')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved as '{filename}'")
    plt.show()
    
    return cm

def plot_model_comparison(results_dict):
    # create comparison visualization of all models
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # extract data
    model_names = list(results_dict.keys())
    accuracies = [results_dict[name]['accuracy'] for name in model_names]
    
    # plot 1: Accuracy Comparison
    ax = axes[0]
    bars = ax.bar(model_names, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # plot 2: Per-Class F1 Scores
    ax = axes[1]
    stages = ['Stage 5', 'Stage 4', 'Stage 3', 'Stage 2', 'Stage 1']
    x = np.arange(len(stages))
    width = 0.25
    
    for i, (name, color) in enumerate(zip(model_names, ['#2ecc71', '#3498db', '#e74c3c'])):
        f1_scores = results_dict[name]['f1_scores']
        ax.bar(x + i*width, f1_scores, width, label=name, color=color, alpha=0.8)
    
    ax.set_xlabel('CKD Stage', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Per-Stage F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Model comparison saved as 'model_comparison.png'")
    plt.show()

def extract_metrics_from_report(report_dict):
    # extract per-class metrics from classification report
    stages = ['Stage 5', 'Stage 4', 'Stage 3', 'Stage 2', 'Stage 1']
    precisions = []
    recalls = []
    f1_scores = []
    
    for stage in stages:
        if stage in report_dict:
            precisions.append(report_dict[stage]['precision'])
            recalls.append(report_dict[stage]['recall'])
            f1_scores.append(report_dict[stage]['f1-score'])
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
    
    return precisions, recalls, f1_scores

def comprehensive_multimodel_analysis(data):
    # complete multi-model classification analysis
    print("\n" + "="*70)
    print("MULTI-MODEL CLASSIFICATION ANALYSIS - GFR STAGE PREDICTION")
    print("="*70)
    
    # prepare data
    scaler, X, y, feature_names = load_and_preprocess_classification(data)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # define models
    models = {
        'Random Forest': train_random_forest,
        'Gradient Boosting': train_gradient_boosting,
        'SVM': train_svm
    }
    
    results = {}
    trained_models = {}
    
    # train and evaluate each model
    for model_name, train_func in models.items():
        print(f"\n{'='*70}")
        print(f"Training {model_name}...")
        print(f"{'='*70}")
        
        # train
        model = train_func(X_train, y_train)
        trained_models[model_name] = model
        
        # predict
        y_pred = model.predict(X_test)
        
        # calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # extract metrics
        precisions, recalls, f1_scores = extract_metrics_from_report(report)
        
        # store results
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'precisions': precisions,
            'recalls': recalls,
            'f1_scores': f1_scores
        }
        
        # plot confusion matrix
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plot_confusion_matrix_detailed(y_test, y_pred, 
                                      title=f"{model_name} - CKD Stage Classification",
                                      filename=filename)
    
    # create comparison visualization
    print(f"\n{'='*70}")
    print("GENERATING MODEL COMPARISON...")
    print(f"{'='*70}")
    plot_model_comparison(results)
    
    # determine best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print(f"\n{'='*70}")
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"{'='*70}")
    
    return trained_models, results, scaler, feature_names, y_test

def predict_gfr_from_raw_input(
    model,
    scaler,
    feature_columns,
    raw_input_dict
):
    # raw_input_dict: dictionary with ORIGINAL feature names
    # create DataFrame with correct columns
    input_df = pd.DataFrame([raw_input_dict])

    # add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # ensure correct column order
    input_df = input_df[feature_columns]

    # scale
    input_scaled = scaler.transform(input_df)

    # predict
    return model.predict(input_scaled)[0]

def ensemble_prediction(models, scaler, feature_columns, raw_input_dict):
    # make predictions using all models and return ensemble result
    predictions = {}
    
    # create DataFrame with correct columns
    input_df = pd.DataFrame([raw_input_dict])
    
    # add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # ensure correct column order
    input_df = input_df[feature_columns]
    
    # scale
    input_scaled = scaler.transform(input_df)
    
    # get predictions from each model
    for model_name, model in models.items():
        pred = model.predict(input_scaled)[0]
        predictions[model_name] = pred
    
    # voting ensemble (most common prediction)
    from collections import Counter
    vote_counts = Counter(predictions.values())
    ensemble_pred = vote_counts.most_common(1)[0][0]
    
    return predictions, ensemble_pred

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # load your data
    data = pd.read_csv('kidney_dataset.csv')
    
    print("="*70)
    print("KIDNEY FUNCTION PREDICTION - MULTI-MODEL ANALYSIS")
    print("="*70)
    
    # run multi-model analysis
    trained_models, results, scaler, feature_names, y_test = comprehensive_multimodel_analysis(data)
    
    # test with sample input
    print("\n\n" + "="*70)
    print("TESTING ENSEMBLE PREDICTION")
    print("="*70)
    
    raw_input = {
        'Creatinine': 6.989712459845687,
        'BUN': 118.99988144292793,
        'Urine_Output': 1082.7779183529092,
        'Diabetes': 0,
        'Hypertension': 0,
        'Age': 39.663323034173146,
        'Protein_in_Urine': 331.22303947029536,
        'Water_Intake': 1.7487539428038266,
        'Medication_ARB': 0,
        'Medication_ACE_Inhibitor': 0
    }
    
    predictions, ensemble_pred = ensemble_prediction(trained_models, scaler, feature_names, raw_input)
    
    print("\nIndividual Model Predictions:")
    for model_name, pred in predictions.items():
        print(f"  {model_name:20s}: {pred}")
    
    print(f"\n🎯 Ensemble Prediction (Voting): {ensemble_pred}")
    print("="*70)
