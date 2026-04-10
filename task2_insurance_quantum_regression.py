from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


DEFAULT_DATASET = Path(
    r"C:\Users\namfa\Downloads\abfa2rbci2UF6CTj_cal_insurance_fire_census_weather (2).csv"
)
DEFAULT_OUTPUT = Path("results_task2")
QUANTUM_QUBITS = 4
TRAIN_TARGET_YEARS = [2019, 2020]
FINAL_TARGET_YEAR = 2021


@dataclass
class ResourceEstimate:
    qubits: int
    depth: int
    size: int
    one_qubit_gates: int
    two_qubit_gates: int
    observables: int
    train_samples: int
    test_samples: int
    simulator: str
    shots_if_run_on_qasm_backend: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 2 starter: hybrid quantum regression for insurance premium prediction."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--risk-features",
        type=Path,
        default=None,
        help="Optional CSV with columns zip,target_year,risk_score to merge as an extra feature.",
    )
    return parser.parse_args()


def load_insurance_rows(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, low_memory=False)
    df = df.rename(columns={"ZIP": "zip", "Year": "year"})
    df["zip"] = pd.to_numeric(df["zip"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["zip"].notna() & df["year"].notna()].copy()
    df["zip"] = df["zip"].astype(int)
    df["year"] = df["year"].astype(int)
    df = df[df["year"].between(2018, 2021)]

    numeric_cols = [
        "Avg Fire Risk Score",
        "Avg PPC",
        "CAT Cov A Fire -  Incurred Losses",
        "CAT Cov A Fire -  Number of Claims",
        "CAT Cov A Smoke -  Incurred Losses",
        "CAT Cov A Smoke -  Number of Claims",
        "CAT Cov C Fire -  Incurred Losses",
        "CAT Cov C Fire -  Number of Claims",
        "CAT Cov C Smoke -  Incurred Losses",
        "CAT Cov C Smoke -  Number of Claims",
        "Cov A Amount Weighted Avg",
        "Cov C Amount Weighted Avg",
        "Earned Exposure",
        "Earned Premium",
        "Non-CAT Cov A Fire -  Incurred Losses",
        "Non-CAT Cov A Fire -  Number of Claims",
        "Non-CAT Cov A Smoke -  Incurred Losses",
        "Non-CAT Cov A Smoke -  Number of Claims",
        "Non-CAT Cov C Fire -  Incurred Losses",
        "Non-CAT Cov C Fire -  Number of Claims",
        "Non-CAT Cov C Smoke -  Incurred Losses",
        "Non-CAT Cov C Smoke -  Number of Claims",
        "Number of High Fire Risk Exposure",
        "Number of Low Fire Risk Exposure",
        "Number of Moderate Fire Risk Exposure",
        "Number of Negligible Fire Risk Exposure",
        "Number of Very High Fire Risk Exposure",
        "avg_tmax_c",
        "avg_tmin_c",
        "tot_prcp_mm",
        "total_population",
        "median_income",
        "total_housing_units",
        "average_household_size",
        "educational_attainment_bachelor_or_higher",
        "poverty_status",
        "housing_occupancy_number",
        "housing_value",
        "year_structure_built",
        "housing_vacancy_number",
        "median_monthly_housing_costs",
        "owner_occupied_housing_units",
        "renter_occupied_housing_units",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    category_cols = ["Category_CO", "Category_DO", "Category_DT", "Category_HO", "Category_MH", "Category_RT"]
    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    return df


def aggregate_zip_year(df: pd.DataFrame) -> pd.DataFrame:
    loss_cols = [
        "CAT Cov A Fire -  Incurred Losses",
        "CAT Cov A Smoke -  Incurred Losses",
        "CAT Cov C Fire -  Incurred Losses",
        "CAT Cov C Smoke -  Incurred Losses",
        "Non-CAT Cov A Fire -  Incurred Losses",
        "Non-CAT Cov A Smoke -  Incurred Losses",
        "Non-CAT Cov C Fire -  Incurred Losses",
        "Non-CAT Cov C Smoke -  Incurred Losses",
    ]
    claim_cols = [
        "CAT Cov A Fire -  Number of Claims",
        "CAT Cov A Smoke -  Number of Claims",
        "CAT Cov C Fire -  Number of Claims",
        "CAT Cov C Smoke -  Number of Claims",
        "Non-CAT Cov A Fire -  Number of Claims",
        "Non-CAT Cov A Smoke -  Number of Claims",
        "Non-CAT Cov C Fire -  Number of Claims",
        "Non-CAT Cov C Smoke -  Number of Claims",
    ]
    fire_exposure_cols = [
        "Number of High Fire Risk Exposure",
        "Number of Low Fire Risk Exposure",
        "Number of Moderate Fire Risk Exposure",
        "Number of Negligible Fire Risk Exposure",
        "Number of Very High Fire Risk Exposure",
    ]
    category_cols = ["Category_CO", "Category_DO", "Category_DT", "Category_HO", "Category_MH", "Category_RT"]
    static_cols = [
        "avg_tmax_c",
        "avg_tmin_c",
        "tot_prcp_mm",
        "total_population",
        "median_income",
        "total_housing_units",
        "average_household_size",
        "educational_attainment_bachelor_or_higher",
        "poverty_status",
        "housing_occupancy_number",
        "housing_value",
        "year_structure_built",
        "housing_vacancy_number",
        "median_monthly_housing_costs",
        "owner_occupied_housing_units",
        "renter_occupied_housing_units",
        "Avg Fire Risk Score",
        "Avg PPC",
        "Cov A Amount Weighted Avg",
        "Cov C Amount Weighted Avg",
    ]

    for col in loss_cols + claim_cols + fire_exposure_cols + static_cols + category_cols:
        if col not in df.columns:
            df[col] = np.nan if col not in category_cols else 0

    grouped = (
        df.groupby(["zip", "year"], as_index=False)
        .agg(
            earned_premium=("Earned Premium", "sum"),
            earned_exposure=("Earned Exposure", "sum"),
            avg_fire_risk_score=("Avg Fire Risk Score", "mean"),
            avg_ppc=("Avg PPC", "mean"),
            cov_a_amount_weighted_avg=("Cov A Amount Weighted Avg", "mean"),
            cov_c_amount_weighted_avg=("Cov C Amount Weighted Avg", "mean"),
            avg_tmax=("avg_tmax_c", "mean"),
            avg_tmin=("avg_tmin_c", "mean"),
            total_prcp=("tot_prcp_mm", "mean"),
            total_population=("total_population", "mean"),
            median_income=("median_income", "mean"),
            total_housing_units=("total_housing_units", "mean"),
            average_household_size=("average_household_size", "mean"),
            educational_attainment_bachelor_or_higher=(
                "educational_attainment_bachelor_or_higher",
                "mean",
            ),
            poverty_status=("poverty_status", "mean"),
            housing_occupancy_number=("housing_occupancy_number", "mean"),
            housing_value=("housing_value", "mean"),
            year_structure_built=("year_structure_built", "mean"),
            housing_vacancy_number=("housing_vacancy_number", "mean"),
            median_monthly_housing_costs=("median_monthly_housing_costs", "mean"),
            owner_occupied_housing_units=("owner_occupied_housing_units", "mean"),
            renter_occupied_housing_units=("renter_occupied_housing_units", "mean"),
        )
    )

    grouped["total_losses"] = df.groupby(["zip", "year"])[loss_cols].sum(min_count=1).sum(axis=1).values
    grouped["total_claims"] = df.groupby(["zip", "year"])[claim_cols].sum(min_count=1).sum(axis=1).values
    grouped["total_fire_risk_exposure"] = (
        df.groupby(["zip", "year"])[fire_exposure_cols].sum(min_count=1).sum(axis=1).values
    )

    fire_exposure_breakdown = df.groupby(["zip", "year"], as_index=False)[fire_exposure_cols].sum(min_count=1)
    category_presence = df.groupby(["zip", "year"])[category_cols].max().reset_index()
    grouped = grouped.merge(fire_exposure_breakdown, on=["zip", "year"], how="left")
    grouped = grouped.merge(category_presence, on=["zip", "year"], how="left")

    grouped["premium_per_exposure"] = grouped["earned_premium"] / grouped["earned_exposure"].replace(0, np.nan)
    grouped["loss_ratio"] = grouped["total_losses"] / grouped["earned_premium"].replace(0, np.nan)
    grouped["claims_per_exposure"] = grouped["total_claims"] / grouped["earned_exposure"].replace(0, np.nan)
    grouped["high_risk_exposure_share"] = (
        grouped["Number of High Fire Risk Exposure"].fillna(0)
        + grouped["Number of Very High Fire Risk Exposure"].fillna(0)
    ) / grouped["total_fire_risk_exposure"].replace(0, np.nan)
    grouped["heat_stress"] = grouped["avg_tmax"] - grouped["avg_tmin"]
    grouped["hot_dry_idx"] = grouped["avg_tmax"] - 0.1 * grouped["total_prcp"]
    return grouped


def build_supervised_table(zip_year_df: pd.DataFrame, risk_df: pd.DataFrame | None = None) -> pd.DataFrame:
    lookup = zip_year_df.set_index(["zip", "year"])
    zip_codes = sorted(zip_year_df["zip"].unique().tolist())

    rows: list[dict[str, float | int]] = []
    lag_base_cols = [
        "earned_premium",
        "earned_exposure",
        "premium_per_exposure",
        "total_losses",
        "total_claims",
        "loss_ratio",
        "claims_per_exposure",
        "avg_fire_risk_score",
        "avg_ppc",
        "high_risk_exposure_share",
        "avg_tmax",
        "avg_tmin",
        "total_prcp",
        "heat_stress",
        "hot_dry_idx",
        "median_income",
        "housing_value",
        "total_population",
    ]
    category_cols = ["Category_CO", "Category_DO", "Category_DT", "Category_HO", "Category_MH", "Category_RT"]

    risk_lookup = None
    if risk_df is not None:
        risk_lookup = risk_df.set_index(["zip", "target_year"])

    for zip_code in zip_codes:
        for target_year in [2020, 2021]:
            prior_years = [year for year in range(2018, target_year) if (zip_code, year) in lookup.index]
            if not prior_years:
                continue

            row: dict[str, float | int] = {"zip": zip_code, "target_year": target_year}
            latest_year = max(prior_years)

            for col in lag_base_cols:
                row[f"{col}_prev1"] = float(lookup.loc[(zip_code, latest_year), col]) if (zip_code, latest_year) in lookup.index else np.nan
                hist = [float(lookup.loc[(zip_code, year), col]) for year in prior_years if pd.notna(lookup.loc[(zip_code, year), col])]
                row[f"{col}_hist_mean"] = float(np.mean(hist)) if hist else np.nan
                row[f"{col}_hist_trend"] = float(hist[-1] - hist[0]) if len(hist) >= 2 else 0.0

            for col in category_cols:
                hist = [float(lookup.loc[(zip_code, year), col]) for year in prior_years if pd.notna(lookup.loc[(zip_code, year), col])]
                row[f"{col}_mean"] = float(np.mean(hist)) if hist else 0.0

            if risk_lookup is not None and (zip_code, target_year) in risk_lookup.index:
                row["external_risk_score"] = float(risk_lookup.loc[(zip_code, target_year), "risk_score"])
            else:
                row["external_risk_score"] = np.nan

            if (zip_code, target_year) not in lookup.index:
                continue
            target_premium = float(lookup.loc[(zip_code, target_year), "earned_premium"])
            row["target_earned_premium"] = target_premium
            row["target_log_earned_premium"] = float(np.log1p(max(target_premium, 0.0)))
            rows.append(row)

    return pd.DataFrame(rows)


def quantum_feature_circuit(angles: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(QUANTUM_QUBITS)
    for i, theta in enumerate(angles):
        qc.ry(float(theta), i)
        qc.rz(float(theta * theta / np.pi), i)
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 0)):
        qc.cz(left, right)
    for i, theta in enumerate(angles):
        neighbor = angles[(i + 1) % QUANTUM_QUBITS]
        qc.rx(float(0.5 * (theta + neighbor)), i)
    for left, right in ((0, 2), (1, 3)):
        qc.cz(left, right)
    return qc


def build_observables() -> list[SparsePauliOp]:
    labels = ["ZIII", "IZII", "IIZI", "IIIZ", "ZZII", "IZZI", "IIZZ", "ZIIZ"]
    return [SparsePauliOp(label) for label in labels]


def statevector_expectations(angles: np.ndarray, observables: list[SparsePauliOp]) -> np.ndarray:
    state = Statevector.from_instruction(quantum_feature_circuit(angles))
    return np.array(
        [float(np.real(state.expectation_value(observable))) for observable in observables],
        dtype=float,
    )


def build_quantum_features(encoded_matrix: np.ndarray) -> np.ndarray:
    observables = build_observables()
    cache: dict[tuple[float, ...], np.ndarray] = {}
    feature_rows = []
    for row in encoded_matrix:
        key = tuple(np.round(row, 8))
        if key not in cache:
            cache[key] = statevector_expectations(row, observables)
        feature_rows.append(cache[key])
    return np.vstack(feature_rows)


def estimate_resources(reference_angles: np.ndarray, train_size: int, test_size: int) -> ResourceEstimate:
    simulator = AerSimulator(method="statevector")
    circuit = quantum_feature_circuit(reference_angles)
    transpiled = transpile(circuit, simulator, optimization_level=1)
    counts = transpiled.count_ops()
    one_qubit = int(sum(count for gate, count in counts.items() if gate not in {"cz", "cx", "ecr"}))
    two_qubit = int(sum(counts.get(gate, 0) for gate in ("cz", "cx", "ecr")))
    return ResourceEstimate(
        qubits=transpiled.num_qubits,
        depth=transpiled.depth(),
        size=transpiled.size(),
        one_qubit_gates=one_qubit,
        two_qubit_gates=two_qubit,
        observables=len(build_observables()),
        train_samples=train_size,
        test_samples=test_size,
        simulator="AerSimulator(statevector)",
        shots_if_run_on_qasm_backend=2048,
    )


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae = float(mean_absolute_error(y_true_log, y_pred_log))
    r2 = float(r2_score(y_true_log, y_pred_log))
    return {"rmse_log": rmse, "mae_log": mae, "r2_log": r2}


def plot_predictions(results_df: pd.DataFrame, output_dir: Path) -> None:
    top = results_df.sort_values("predicted_earned_premium", ascending=False).head(15).copy()
    top["zip"] = top["zip"].astype(str)
    plt.figure(figsize=(10, 5))
    plt.bar(top["zip"], top["predicted_earned_premium"], color="#4078c0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Predicted 2021 earned premium")
    plt.title("Top predicted 2021 earned premium by ZIP")
    plt.tight_layout()
    plt.savefig(output_dir / "top_predicted_premiums_2021.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    risk_df = None
    if args.risk_features is not None and args.risk_features.exists():
        risk_df = pd.read_csv(args.risk_features)
        risk_df = risk_df.rename(columns={"ZIP": "zip", "Year": "target_year"})
        risk_df["zip"] = pd.to_numeric(risk_df["zip"], errors="coerce").astype("Int64")
        risk_df["target_year"] = pd.to_numeric(risk_df["target_year"], errors="coerce").astype("Int64")
        risk_df = risk_df[risk_df["zip"].notna() & risk_df["target_year"].notna()].copy()
        risk_df["zip"] = risk_df["zip"].astype(int)
        risk_df["target_year"] = risk_df["target_year"].astype(int)

    insurance_rows = load_insurance_rows(args.dataset)
    zip_year_df = aggregate_zip_year(insurance_rows)
    supervised_df = build_supervised_table(zip_year_df, risk_df=risk_df)

    feature_columns = [
        col
        for col in supervised_df.columns
        if col not in {"zip", "target_year", "target_earned_premium", "target_log_earned_premium"}
    ]
    feature_columns = [
        col for col in feature_columns if not supervised_df[col].isna().all()
    ]

    train_df = supervised_df[supervised_df["target_year"].isin(TRAIN_TARGET_YEARS)].copy()
    test_df = supervised_df[supervised_df["target_year"] == FINAL_TARGET_YEAR].copy()

    prep = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=QUANTUM_QUBITS, random_state=42)),
            ("angle_scale", MinMaxScaler(feature_range=(-np.pi, np.pi))),
        ]
    )

    x_train_angles = prep.fit_transform(train_df[feature_columns])
    x_test_angles = prep.transform(test_df[feature_columns])
    x_train_quantum = build_quantum_features(x_train_angles)
    x_test_quantum = build_quantum_features(x_test_angles)
    x_train_hybrid = np.hstack([x_train_angles, x_train_quantum])
    x_test_hybrid = np.hstack([x_test_angles, x_test_quantum])

    y_train = train_df["target_log_earned_premium"].to_numpy()
    y_test = test_df["target_log_earned_premium"].to_numpy()

    models = {
        "ridge_pca": Ridge(alpha=1.0),
        "ridge_hybrid": Ridge(alpha=1.0),
        "random_forest_raw": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "gradient_boosting_raw": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("gbr", GradientBoostingRegressor(random_state=42)),
            ]
        ),
    }

    models["ridge_pca"].fit(x_train_angles, y_train)
    models["ridge_hybrid"].fit(x_train_hybrid, y_train)
    models["random_forest_raw"].fit(train_df[feature_columns], y_train)
    models["gradient_boosting_raw"].fit(train_df[feature_columns], y_train)

    pred_pca = models["ridge_pca"].predict(x_test_angles)
    pred_hybrid = models["ridge_hybrid"].predict(x_test_hybrid)
    pred_rf = models["random_forest_raw"].predict(test_df[feature_columns])
    pred_gbr = models["gradient_boosting_raw"].predict(test_df[feature_columns])

    metrics = {
        "ridge_pca": regression_metrics(y_test, pred_pca),
        "ridge_hybrid": regression_metrics(y_test, pred_hybrid),
        "random_forest_raw": regression_metrics(y_test, pred_rf),
        "gradient_boosting_raw": regression_metrics(y_test, pred_gbr),
    }

    prediction_map = {
        "ridge_pca": pred_pca,
        "ridge_hybrid": pred_hybrid,
        "random_forest_raw": pred_rf,
        "gradient_boosting_raw": pred_gbr,
    }
    selected_model = min(metrics, key=lambda name: metrics[name]["rmse_log"])
    final_pred_log = prediction_map[selected_model]
    results_df = test_df[["zip", "target_year", "target_earned_premium"]].copy()
    results_df["predicted_log_earned_premium"] = final_pred_log
    results_df["predicted_earned_premium"] = np.expm1(final_pred_log).clip(min=0.0)
    results_df["absolute_error"] = (
        results_df["predicted_earned_premium"] - results_df["target_earned_premium"]
    ).abs()
    results_df = results_df.sort_values("predicted_earned_premium", ascending=False)

    summary = {
        "dataset_path": str(args.dataset.resolve()),
        "train_target_years": TRAIN_TARGET_YEARS,
        "test_target_year": FINAL_TARGET_YEAR,
        "zip_year_aggregates": int(len(zip_year_df)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "models": metrics,
        "selected_model": selected_model,
        "resource_estimate": asdict(
            estimate_resources(
                x_train_angles[0],
                train_size=len(train_df),
                test_size=len(test_df),
            )
        ),
        "hybrid_feature_dimensions": {
            "classical_pca_features": int(x_train_angles.shape[1]),
            "quantum_observables": int(x_train_quantum.shape[1]),
            "combined_features": int(x_train_hybrid.shape[1]),
        },
        "pca_explained_variance_ratio": prep.named_steps["pca"].explained_variance_ratio_.round(6).tolist(),
        "external_risk_feature_used": risk_df is not None,
    }

    zip_year_df.to_csv(args.output_dir / "task2_zip_year_aggregates.csv", index=False)
    supervised_df.to_csv(args.output_dir / "task2_supervised_dataset.csv", index=False)
    results_df.to_csv(args.output_dir / "task2_predictions_2021.csv", index=False)
    (args.output_dir / "task2_results_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_predictions(results_df, args.output_dir)

    print(json.dumps(summary, indent=2))
    print("\nTop 10 predicted premiums:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
