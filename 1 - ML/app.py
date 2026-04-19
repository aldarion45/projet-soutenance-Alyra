from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise RuntimeError("xgboost est requis: pip install xgboost") from exc

BASE_DIR = Path("github/projet-soutenance-Alyra/1 - ML")
ARTIFACTS_DIR = BASE_DIR / "artifacts_intensite_v6"
DATA_CANDIDATES = [
    BASE_DIR / "gym_members_exercise_tracking_v2_realiste.csv",
    BASE_DIR / "gym_members_exercise_tracking_v2.csv",
    BASE_DIR / "gym_members_exercise_tracking.csv",
]
ARTIFACT_FILES = {
    "rf": ARTIFACTS_DIR / "rf_v6.joblib",
    "xgb": ARTIFACTS_DIR / "xgb_v6.joblib",
    "lr": ARTIFACTS_DIR / "lr_v6.joblib",
    "xgb_le": ARTIFACTS_DIR / "label_encoder_xgb_v6.joblib",
    "rf_xgb_columns": ARTIFACTS_DIR / "rf_xgb_columns_v6.csv",
}

WORKOUT_CHOICES = ["Cardio", "HIIT", "Strength", "Yoga"]
FEATURES_V6 = [
    "Age",
    "Height (m)",
    "Weight (kg)",
    "Fat_Percentage",
    "Avg_BPM",
    "Resting_BPM",
    "Session_Duration (hours)",
    "Workout_Type",
    "hr_ratio_reserve",
    "bpm_reserve",
    "effort_load",
    "fat_mass_kg",
    "lean_mass_kg",
    "BMI_calc",
    "lean_mass_per_height",
    "resting_to_lean_ratio",
]

# Scénario fixe utilisé pour l'analyse de sensibilité (seul le paramètre choisi varie)
FIXED_SCENARIO = {
    "Age": 35,
    "Height (m)": 1.75,
    "Weight (kg)": 75.0,
    "Fat_Percentage": 22.0,
    "Avg_BPM": 155,
    "Resting_BPM": 62,
    "Session_Duration (hours)": 1.0,
    "Workout_Type": "Cardio",
}

PARAMETER_GRID = {
    "Age": np.arange(18, 61, 1),
    "Height (m)": np.round(np.arange(1.40, 2.11, 0.02), 2),
    "Weight (kg)": np.round(np.arange(40.0, 130.1, 1.0), 1),
    "Fat_Percentage": np.round(np.arange(5.0, 50.1, 1.0), 1),
    "Avg_BPM": np.arange(90, 201, 2),
    "Resting_BPM": np.arange(40, 101, 1),
    "Session_Duration (hours)": np.round(np.arange(0.5, 2.01, 0.02), 2),
}
KARVONEN_PARAMS = ["Age", "Avg_BPM", "Resting_BPM"]


def resolve_data_path() -> str:
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Aucun CSV trouve.")


def to_intensity(score: pd.Series) -> pd.Series:
    q1, q2 = score.quantile([0.33, 0.66])

    def _map(x: float) -> str:
        if x <= q1:
            return "faible"
        if x <= q2:
            return "moyen"
        return "eleve"

    return score.apply(_map)


def prepare_dataset_v6(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[
        (df["Age"].between(16, 80))
        & (df["Height (m)"].between(1.35, 2.15))
        & (df["Weight (kg)"].between(40, 170))
        & (df["Fat_Percentage"].between(5, 55))
        & (df["Session_Duration (hours)"].between(0.3, 3.0))
        & (df["Resting_BPM"].between(35, 110))
        & (df["Avg_BPM"].between(90, 210))
    ].copy()

    hr_max_theorique = 220 - df["Age"]
    denom = (hr_max_theorique - df["Resting_BPM"]).replace(0, np.nan)
    df["hr_ratio_reserve"] = (df["Avg_BPM"] - df["Resting_BPM"]) / denom
    df["bpm_reserve"] = df["Avg_BPM"] - df["Resting_BPM"]
    df["effort_load"] = df["hr_ratio_reserve"] * df["Session_Duration (hours)"] * (df["Weight (kg)"] / 70.0)
    df["fat_mass_kg"] = df["Weight (kg)"] * (df["Fat_Percentage"] / 100.0)
    df["lean_mass_kg"] = df["Weight (kg)"] - df["fat_mass_kg"]
    df["BMI_calc"] = df["Weight (kg)"] / (df["Height (m)"] ** 2)
    df["lean_mass_per_height"] = df["lean_mass_kg"] / df["Height (m)"]
    df["resting_to_lean_ratio"] = df["Resting_BPM"] / df["lean_mass_kg"].replace(0, np.nan)

    intensity_score = (
        0.54 * df["hr_ratio_reserve"]
        + 0.16 * (df["Session_Duration (hours)"] / 2.0)
        + 0.10 * ((df["Avg_BPM"] - 100.0) / 100.0)
        + 0.08 * ((df["fat_mass_kg"] / df["Weight (kg)"]) - 0.22)
        + 0.05 * ((df["BMI_calc"] - 24.0) / 10.0)
        + 0.04 * ((df["resting_to_lean_ratio"] * 10.0) - 0.8)
        + 0.03 * ((df["lean_mass_per_height"] - 35.0) / 10.0)
    )
    df["intensite"] = to_intensity(intensity_score)

    model_df = df[FEATURES_V6 + ["intensite"]].replace([np.inf, -np.inf], np.nan).dropna()
    return model_df


def build_models_v6(model_df: pd.DataFrame):
    X_raw = model_df.drop(columns=["intensite"]).copy()
    y = model_df["intensite"].copy()

    X_ohe = pd.get_dummies(X_raw, columns=["Workout_Type"], drop_first=True)

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_ohe, y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(le.classes_),
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    xgb.fit(X_ohe, y_enc)

    num_cols = [
        "Age",
        "Height (m)",
        "Weight (kg)",
        "Fat_Percentage",
        "Avg_BPM",
        "Resting_BPM",
        "Session_Duration (hours)",
        "hr_ratio_reserve",
        "bpm_reserve",
        "effort_load",
        "fat_mass_kg",
        "lean_mass_kg",
        "BMI_calc",
        "lean_mass_per_height",
        "resting_to_lean_ratio",
    ]
    cat_cols = ["Workout_Type"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    lr = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(solver="lbfgs", max_iter=3000, class_weight="balanced")),
        ]
    )
    lr.fit(X_raw, y)

    return {
        "rf": rf,
        "xgb": xgb,
        "lr": lr,
        "xgb_le": le,
        "rf_xgb_columns": X_ohe.columns.tolist(),
        "classes": sorted(y.unique()),
    }


def load_artifact_models_v6() -> Dict:
    missing = [str(path) for path in ARTIFACT_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Artefacts V6 manquants: " + ", ".join(missing))

    rf = joblib.load(ARTIFACT_FILES["rf"])
    xgb = joblib.load(ARTIFACT_FILES["xgb"])
    lr = joblib.load(ARTIFACT_FILES["lr"])
    xgb_le = joblib.load(ARTIFACT_FILES["xgb_le"])
    rf_xgb_columns = (
        pd.read_csv(ARTIFACT_FILES["rf_xgb_columns"], header=None)
        .iloc[:, 0]
        .dropna()
        .astype(str)
        .tolist()
    )
    rf_xgb_columns = [c for c in rf_xgb_columns if c not in {"0", "Unnamed: 0"}]

    return {
        "rf": rf,
        "xgb": xgb,
        "lr": lr,
        "xgb_le": xgb_le,
        "rf_xgb_columns": rf_xgb_columns,
        "classes": list(xgb_le.classes_),
    }


def build_input_row(params: dict) -> pd.DataFrame:
    age = int(params["Age"])
    height_m = float(params["Height (m)"])
    weight = float(params["Weight (kg)"])
    fat_percentage = float(params["Fat_Percentage"])
    avg_bpm = int(params["Avg_BPM"])
    resting_bpm = int(params["Resting_BPM"])
    session_duration = float(params["Session_Duration (hours)"])
    workout_type = str(params["Workout_Type"])

    hr_max_theorique = 220 - age
    denom = hr_max_theorique - resting_bpm
    hr_ratio_reserve = (avg_bpm - resting_bpm) / denom if denom != 0 else np.nan
    bpm_reserve = avg_bpm - resting_bpm
    effort_load = hr_ratio_reserve * session_duration * (weight / 70.0) if np.isfinite(hr_ratio_reserve) else np.nan
    fat_mass_kg = weight * (fat_percentage / 100.0)
    lean_mass_kg = weight - fat_mass_kg
    bmi_calc = weight / (height_m ** 2) if height_m > 0 else np.nan
    lean_mass_per_height = lean_mass_kg / height_m if height_m > 0 else np.nan
    resting_to_lean_ratio = resting_bpm / lean_mass_kg if lean_mass_kg != 0 else np.nan

    return pd.DataFrame([
        {
            "Age": age,
            "Height (m)": height_m,
            "Weight (kg)": weight,
            "Fat_Percentage": fat_percentage,
            "Avg_BPM": avg_bpm,
            "Resting_BPM": resting_bpm,
            "Session_Duration (hours)": session_duration,
            "Workout_Type": workout_type,
            "hr_ratio_reserve": hr_ratio_reserve,
            "bpm_reserve": bpm_reserve,
            "effort_load": effort_load,
            "fat_mass_kg": fat_mass_kg,
            "lean_mass_kg": lean_mass_kg,
            "BMI_calc": bmi_calc,
            "lean_mass_per_height": lean_mass_per_height,
            "resting_to_lean_ratio": resting_to_lean_ratio,
        }
    ])


def class_to_score_label(label: str) -> float:
    norm = str(label).strip().lower()
    if "faible" in norm:
        return 0.165
    if "moy" in norm:
        return 0.495
    return 0.835


def expected_intensity_from_probs(classes: list[str], probs: np.ndarray) -> float:
    return float(sum(class_to_score_label(c) * p for c, p in zip(classes, probs)))


def karvonen_score(age: int, avg_bpm: int, resting_bpm: int) -> float:
    hr_max = 220 - age
    denom = hr_max - resting_bpm
    if denom <= 0:
        return np.nan
    score = (avg_bpm - resting_bpm) / denom
    return float(np.clip(score, 0.0, 1.2))


def predict_point(params: dict) -> dict:
    row = build_input_row(params)
    if row.isna().any(axis=None):
        raise gr.Error("Paramètres invalides: vérifie les valeurs fixes.")

    row_ohe = pd.get_dummies(row, columns=["Workout_Type"], drop_first=True)
    row_ohe = row_ohe.reindex(columns=MODELS["rf_xgb_columns"], fill_value=0)

    rf_probs = MODELS["rf"].predict_proba(row_ohe)[0]
    xgb_probs = MODELS["xgb"].predict_proba(row_ohe)[0]
    lr_probs = MODELS["lr"].predict_proba(row)[0]

    classes = MODELS["classes"]
    rf_score = expected_intensity_from_probs(classes, rf_probs)
    xgb_score = expected_intensity_from_probs(classes, xgb_probs)
    lr_score = expected_intensity_from_probs(classes, lr_probs)

    kar_score = karvonen_score(int(params["Age"]), int(params["Avg_BPM"]), int(params["Resting_BPM"]))

    return {
        "classes": classes,
        "rf_probs": rf_probs,
        "xgb_probs": xgb_probs,
        "lr_probs": lr_probs,
        "rf_score": rf_score,
        "xgb_score": xgb_score,
        "lr_score": lr_score,
        "kar_score": kar_score,
    }


def compare_at_fixed_point(
    age: int,
    height_m: float,
    weight: float,
    fat_percentage: float,
    avg_bpm: int,
    resting_bpm: int,
    session_duration: float,
    workout_type: str,
) -> pd.DataFrame:
    fixed = {
        "Age": age,
        "Height (m)": height_m,
        "Weight (kg)": weight,
        "Fat_Percentage": fat_percentage,
        "Avg_BPM": avg_bpm,
        "Resting_BPM": resting_bpm,
        "Session_Duration (hours)": session_duration,
        "Workout_Type": workout_type,
    }
    payload = predict_point(fixed)
    classes = payload["classes"]

    def probs_txt(probs):
        return " | ".join([f"{c}:{p*100:.1f}%" for c, p in zip(classes, probs)])

    rows = [
        ["RandomForest V6", payload["rf_score"], payload["rf_score"] - payload["kar_score"], probs_txt(payload["rf_probs"])],
        ["XGBoost V6", payload["xgb_score"], payload["xgb_score"] - payload["kar_score"], probs_txt(payload["xgb_probs"])],
        ["LogisticRegression V6", payload["lr_score"], payload["lr_score"] - payload["kar_score"], probs_txt(payload["lr_probs"])],
        ["Karvonen", payload["kar_score"], 0.0, "-"],
    ]
    return pd.DataFrame(rows, columns=["Modele", "Score", "Ecart_vs_Karvonen", "Probabilites"])


def plot_sensitivity(
    varying_param: str,
    age: int,
    height_m: float,
    weight: float,
    fat_percentage: float,
    avg_bpm: int,
    resting_bpm: int,
    session_duration: float,
    workout_type: str,
):
    if varying_param not in PARAMETER_GRID:
        raise gr.Error("Paramètre évolutif non supporté.")

    fixed = {
        "Age": age,
        "Height (m)": height_m,
        "Weight (kg)": weight,
        "Fat_Percentage": fat_percentage,
        "Avg_BPM": avg_bpm,
        "Resting_BPM": resting_bpm,
        "Session_Duration (hours)": session_duration,
        "Workout_Type": workout_type,
    }
    xs = PARAMETER_GRID[varying_param]
    rf_scores, xgb_scores, lr_scores, kar_scores = [], [], [], []

    for x in xs:
        params = fixed.copy()
        params[varying_param] = float(x) if isinstance(x, np.floating) else x

        payload = predict_point(params)
        rf_scores.append(payload["rf_score"])
        xgb_scores.append(payload["xgb_score"])
        lr_scores.append(payload["lr_score"])
        kar_scores.append(payload["kar_score"])

    rf_scores = np.array(rf_scores)
    xgb_scores = np.array(xgb_scores)
    lr_scores = np.array(lr_scores)
    kar_scores = np.array(kar_scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(xs, rf_scores, label="RF V6", linewidth=2)
    axes[0].plot(xs, xgb_scores, label="XGB V6", linewidth=2)
    axes[0].plot(xs, lr_scores, label="LR V6", linewidth=2)
    axes[0].plot(xs, kar_scores, label="Karvonen", linewidth=2, linestyle="--", color="black")
    axes[0].set_title(f"Scores d'intensité vs {varying_param}")
    axes[0].set_xlabel(varying_param)
    axes[0].set_ylabel("Score intensité")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(xs, rf_scores - kar_scores, label="RF - Karvonen", linewidth=2)
    axes[1].plot(xs, xgb_scores - kar_scores, label="XGB - Karvonen", linewidth=2)
    axes[1].plot(xs, lr_scores - kar_scores, label="LR - Karvonen", linewidth=2)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"Écart au score Karvonen vs {varying_param}")
    axes[1].set_xlabel(varying_param)
    axes[1].set_ylabel("Écart de score")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fixed_txt = ", ".join([f"{k}={v}" for k, v in fixed.items() if k != varying_param])
    fig.suptitle(f"Paramètres fixes: {fixed_txt}", fontsize=9)
    fig.tight_layout()
    return fig


DATA_PATH = resolve_data_path()
MODEL_DF = prepare_dataset_v6(DATA_PATH)
try:
    MODELS = load_artifact_models_v6()
    print(f"[INFO] Modèles V6 chargés depuis: {ARTIFACTS_DIR}")
except Exception as exc:
    print(f"[INFO] Artefacts V6 indisponibles ({exc}). Réentraînement local des modèles V6.")
    MODELS = build_models_v6(MODEL_DF)


with gr.Blocks(title="Comparaison V6 vs Karvonen") as demo:
    gr.Markdown(
        """
# Comparaison modèles V6 vs formule de Karvonen
- Tu choisis tes paramètres fixes.
- Tu choisis **un seul** paramètre évolutif.
- L'app trace les courbes des scores et des écarts à Karvonen.
"""
    )

    with gr.Row():
        age = gr.Slider(18, 60, value=35, step=1, label="Age (fixe)")
        height_m = gr.Slider(1.40, 2.10, value=1.75, step=0.01, label="Height (m) (fixe)")
        weight = gr.Slider(40.0, 130.0, value=75.0, step=0.1, label="Weight (kg) (fixe)")
        fat_percentage = gr.Slider(5.0, 50.0, value=22.0, step=0.1, label="Fat_Percentage (fixe)")

    with gr.Row():
        avg_bpm = gr.Slider(90, 200, value=155, step=1, label="Avg_BPM (fixe)")
        resting_bpm = gr.Slider(40, 100, value=62, step=1, label="Resting_BPM (fixe)")
        session_duration = gr.Slider(0.5, 2.0, value=1.0, step=0.01, label="Session_Duration (hours) (fixe)")
        workout_type = gr.Dropdown(WORKOUT_CHOICES, value="Cardio", label="Workout_Type (fixe)")

    with gr.Tab("Point fixe"):
        btn_point = gr.Button("Comparer au point fixe choisi", variant="primary")
        out_point = gr.Dataframe(label="Comparaison au point fixe", interactive=False)
        btn_point.click(
            fn=compare_at_fixed_point,
            inputs=[age, height_m, weight, fat_percentage, avg_bpm, resting_bpm, session_duration, workout_type],
            outputs=out_point,
        )

    with gr.Tab("Sensibilité (1 paramètre évolutif)"):
        varying_param = gr.Dropdown(
            choices=KARVONEN_PARAMS,
            value="Avg_BPM",
            label="Paramètre évolutif (Karvonen)",
        )
        btn_plot = gr.Button("Tracer les courbes", variant="secondary")
        out_plot = gr.Plot(label="Courbes de comparaison")
        btn_plot.click(
            fn=plot_sensitivity,
            inputs=[varying_param, age, height_m, weight, fat_percentage, avg_bpm, resting_bpm, session_duration, workout_type],
            outputs=out_plot,
        )


if __name__ == "__main__":
    demo.launch()
