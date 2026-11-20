from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import pandas as pd


@dataclass
class GateResult:
    gate_name: str
    passed: bool
    observed: float | int
    threshold: float | int
    details: str


class QualityGateError(RuntimeError):
    """Lançada quando um quality gate crítico reprova e bloqueia o pipeline."""


def evaluate_data_quality(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    qg_cfg = config["quality_gates"]

    date_col = data_cfg["date_column"]
    required_cols = data_cfg["required_columns"]
    bounds_cfg = data_cfg.get("physical_bounds", {})

    gates: list[GateResult] = []
    notes: list[str] = []

    # 1. Colunas obrigatórias
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {missing_cols}")

    # 2. Quantidade mínima de registros
    n_rows = len(df)
    min_records = qg_cfg.get("min_records", 1)
    gates.append(
        GateResult(
            gate_name="min_records",
            passed=n_rows >= min_records,
            observed=n_rows,
            threshold=min_records,
            details=f"Total de registros: {n_rows} (mínimo exigido: {min_records}).",
        )
    )

    # 3. Duplicidade e ordenação temporal
    if date_col in df.columns:
        parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
        invalid_dates = int(parsed_dates.isna().sum())
        dup_dates = int(parsed_dates.dropna().duplicated().sum())
        is_sorted = bool(parsed_dates.dropna().is_monotonic_increasing)

        if invalid_dates > 0:
            notes.append(f"{invalid_dates} datas inválidas encontradas em '{date_col}'.")
        if dup_dates > 0:
            notes.append(f"{dup_dates} datas duplicadas encontradas em '{date_col}'.")
        if not is_sorted:
            notes.append("Série temporal não está estritamente ordenada por data.")

    # 4. Proporção máxima de dados faltantes (maior taxa entre colunas obrigatórias)
    max_missing_ratio = float(qg_cfg.get("max_missing_ratio", 0.05))
    missing_ratios = {
        col: float(df[col].isna().mean()) if len(df) > 0 else 1.0
        for col in required_cols
    }
    worst_missing_col = max(missing_ratios, key=missing_ratios.get)
    worst_missing_val = round(missing_ratios[worst_missing_col], 4)

    gates.append(
        GateResult(
            gate_name="max_missing_ratio",
            passed=worst_missing_val <= max_missing_ratio,
            observed=worst_missing_val,
            threshold=max_missing_ratio,
            details=f"Maior taxa de ausência em '{worst_missing_col}': {worst_missing_val:.2%}.",
        )
    )

    # 5. Limites físicos (valores fora da faixa esperada)
    max_oob_ratio = float(qg_cfg.get("max_out_of_bounds_ratio", 0.01))
    oob_ratios: dict[str, float] = {}

    for col, limits in bounds_cfg.items():
        if col not in df.columns or len(df) == 0:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        oob_mask = pd.Series(False, index=df.index)
        if "min" in limits and limits["min"] is not None:
            oob_mask = oob_mask | (series < float(limits["min"]))
        if "max" in limits and limits["max"] is not None:
            oob_mask = oob_mask | (series > float(limits["max"]))
        oob_ratios[col] = float(oob_mask.mean())

    if oob_ratios:
        worst_oob_col = max(oob_ratios, key=oob_ratios.get)
        worst_oob_val = round(oob_ratios[worst_oob_col], 4)
        gates.append(
            GateResult(
                gate_name="max_out_of_bounds_ratio",
                passed=worst_oob_val <= max_oob_ratio,
                observed=worst_oob_val,
                threshold=max_oob_ratio,
                details=f"Maior taxa fora dos limites físicos em '{worst_oob_col}': {worst_oob_val:.2%}.",
            )
        )

    overall_passed = all(g.passed for g in gates)
    report = {
        "cycle_id": config["experiment"]["cycle_id"],
        "run_id": config["experiment"]["run_id"],
        "overall_status": "APROVADO" if overall_passed else "REPROVADO",
        "row_count": n_rows,
        "missing_ratio_by_column": {k: round(v, 4) for k, v in missing_ratios.items()},
        "out_of_bounds_ratio_by_column": {k: round(v, 4) for k, v in oob_ratios.items()},
        "gates": [asdict(g) for g in gates],
        "notes": notes,
    }

    if not overall_passed and bool(qg_cfg.get("fail_on_gate_breach", True)):
        failed_names = [g.gate_name for g in gates if not g.passed]
        raise QualityGateError(
            f"Qualidade reprovada nos gates: {failed_names}. Relatório: {report}"
        )

    return report