#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
from statistics import mean
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

def find_file(folder: str, candidates: List[str]) -> Optional[str]:
    lc = {f.lower(): f for f in os.listdir(folder)}
    for cand in candidates:
        for k, v in lc.items():
            if k == cand.lower():
                return os.path.join(folder, v)
    for cand in candidates:
        for k, v in lc.items():
            if cand.lower() in k:
                return os.path.join(folder, v)
    return None


def snake_like(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def safe_mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def fmt(v: Optional[float]) -> Optional[float]:
    return None if v is None else float(v)


def add_row(rows: List[Dict[str, Any]], section: str, item: str, value: Optional[float]):
    rows.append({
        "Section": section,
        "Item": item,
        "Value": "" if value is None else value
    })


def parse_lm_eval(lm_path: str) -> List[Dict[str, Any]]:
    with open(lm_path, "r", encoding="utf-8") as f:
        lm = json.load(f)

    results: Dict[str, Any] = lm.get("results", {})
    groups: Dict[str, Any] = lm.get("groups", {})
    group_subtasks: Dict[str, List[str]] = lm.get("group_subtasks", {})

    rows: List[Dict[str, Any]] = []

    hrm8k_map = {
        "hrm8k_gsm8k": "GSM8k",
        "hrm8k_ksm": "KSM",
        "hrm8k_math": "Math",
        "hrm8k_mmmlu": "MMLU",
        "hrm8k_omni_math": "OMNIMATH",
    }
    hrm_values = []
    for key, label in hrm8k_map.items():
        val = None
        if key in results:
            for met in ("exact_match,none", "exact_match"):
                if met in results[key]:
                    val = fmt(results[key][met])
                    break
        add_row(rows, "HRM8k", label, val)
        if val is not None:
            hrm_values.append(val)
    add_row(rows, "HRM8k", "Average", safe_mean(hrm_values))

    kmmlu_groups = [
        ("kmmlu_applied_science", "Applied_science"),
        ("kmmlu_humss", "humss"),
        ("kmmlu_other", "other"),
        ("kmmlu_stem", "stem"),
    ]
    kmmlu_vals = []
    for gkey, glabel in kmmlu_groups:
        val = None
        if gkey in groups:
            val = fmt(groups[gkey].get("acc,none"))
        add_row(rows, "KMMLU", glabel, val)
        if val is not None:
            kmmlu_vals.append(val)

    kmmlu_avg = None
    if "kmmlu" in groups and groups["kmmlu"].get("acc,none") is not None:
        kmmlu_avg = fmt(groups["kmmlu"]["acc,none"])
    else:
        kmmlu_avg = safe_mean(kmmlu_vals)
    add_row(rows, "KMMLU", "Average", kmmlu_avg)

    haerae_tasks = group_subtasks.get("haerae", [])
    haerae_map = {
        "haerae_correct_definition_matching": "correct_definition_matching",
        "haerae_csat_geo": "csat_geo",
        "haerae_csat_law": "csat_law",
        "haerae_csat_socio": "csat_socio",
        "haerae_date_understanding": "date_understanding",
        "haerae_general_knowledge": "general_knowledge",
        "haerae_history": "history",
        "haerae_loan_word": "loan_word",
        "haerae_rare_word": "rare_word",
        "haerae_reading_comprehension": "reading_comprehension",
        "haerae_standard_nomenclature": "standard_nomenclature",
    }
    hae_vals = []
    for t in haerae_tasks:
        label = haerae_map.get(t, t.replace("haerae_", "").replace("_", " "))
        val = None
        if t in results:
            for met in ("acc,none", "acc"):
                if met in results[t]:
                    val = fmt(results[t][met])
                    break
        add_row(rows, "HAERAE", label, val)
        if val is not None:
            hae_vals.append(val)

    haerae_avg = None
    if "haerae" in groups and groups["haerae"].get("acc,none") is not None:
        haerae_avg = fmt(groups["haerae"]["acc,none"])
    else:
        haerae_avg = safe_mean(hae_vals)
    add_row(rows, "HAERAE", "Average", haerae_avg)

    return rows


FCB_SINGLE_KEYS = [
    "exact_1",
    "random_4",
    "close_4",
    "random_8",
    "close_8",
]

DIALOG_KEYS = [
    "tool_call",
    "answer_completion",
    "slot_question",
    "relevance_det",
]


def read_tsv_as_dicts(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    if pd is not None:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        headers = list(df.columns)
        records = df.to_dict(orient="records")
        return headers, records
    else:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            headers = reader.fieldnames or []
            records = [row for row in reader]
        return headers, records


def standardize_headers(headers: List[str]) -> Dict[str, str]:
    return {h: snake_like(h) for h in headers}


def extract_means(records: List[Dict[str, Any]],
                  std_hdr_map: Dict[str, str],
                  wanted_patterns: List[str]) -> Dict[str, float]:

    def matches(pattern: str, std_name: str) -> bool:
        if std_name == pattern:
            return True
        if pattern in std_name:
            return True
        if pattern == "answer_completion" and ("answer_complet" in std_name or "answer_completion" in std_name):
            return True
        if pattern == "relevance_det" and ("relevance" in std_name and ("det" in std_name or "detection" in std_name)):
            return True
        return False

    inv = {v: k for k, v in std_hdr_map.items()}
    out: Dict[str, float] = {}

    for pat in wanted_patterns:
        cand_std_names = [std for std in std_hdr_map.values() if matches(pat, std)]
        values: List[float] = []
        for std_name in cand_std_names:
            orig = inv.get(std_name)
            if not orig:
                continue
            for row in records:
                if orig not in row:
                    continue
                raw = row[orig]
                if raw is None or str(raw).strip() == "":
                    continue
                try:
                    v = float(raw)
                    values.append(v)
                except Exception:
                    low = str(raw).strip().lower()
                    if low in ("true", "t", "yes", "y"):
                        values.append(1.0)
                    elif low in ("false", "f", "no", "n"):
                        values.append(0.0)
                    else:
                        pass
        if values:
            out[pat] = sum(values) / len(values)

    return out


def read_tsv(path: str) -> "pd.DataFrame":
    return pd.read_csv(path, sep="\t")


def pass_ratio_by(df: "pd.DataFrame", status_col: str, type_col: str) -> Dict[str, float]:
    d = df.copy()
    d["_pass"] = d[status_col].astype(str).str.lower().eq("pass")
    return d.groupby(d[type_col].astype(str).str.lower())["_pass"].mean().to_dict(), float(d["_pass"].mean())


def parse_functionchat(folder: str, single_tsv: Optional[str], dialog_tsv: Optional[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # ---- SingleCall ----
    if single_tsv and os.path.exists(single_tsv):
        df_single = read_tsv(single_tsv)
        # cols: '#serial_num', 'is_pass', 'tools_type', ...
        ratio_map, micro = pass_ratio_by(df_single, "is_pass", "tools_type")
        single_order = [
            ("exact",     "exact-1"),
            ("4_random",  "random-4"),
            ("4_close",   "close-4"),
            ("8_random",  "random-8"),
            ("8_close",   "close-8"),
        ]
        for key, label in single_order:
            add_row(rows, "FunctionChat-Bench/SingleCall", label, fmt(ratio_map.get(key)))
        add_row(rows, "FunctionChat-Bench/SingleCall", "micro Avg", fmt(micro))

    # ---- Dialog ----
    if dialog_tsv and os.path.exists(dialog_tsv):
        df_dialog = read_tsv(dialog_tsv)
        # cols: '#serial_num', 'is_pass', 'type_of_output', ...
        ratio_map, _ = pass_ratio_by(df_dialog, "is_pass", "type_of_output")
        dialog_order = [
            ("call",        "Tool Call"),
            ("completion",  "Answer Complet"),
            ("slot",        "Slot Question"),
            ("relevance",   "Relevance Dete"),
        ]
        for key, label in dialog_order:
            add_row(rows, "FunctionChat-Bench/Dialog", label, fmt(ratio_map.get(key)))

    return rows

def parse_logickor(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    rows: List[Dict[str, Any]] = []

    cats = [k for k in j.keys() if k.lower() != "overall"]

    single_vals = []
    for c in cats:
        v = j[c].get("single_turn")
        add_row(rows, "LogicKor/Single turn", c, fmt(v))
        if v is not None:
            single_vals.append(v)
    add_row(rows, "LogicKor/Single turn", "Singleturn Avg",
            safe_mean(single_vals))

    multi_vals = []
    for c in cats:
        v = j[c].get("multi_turn")
        add_row(rows, "LogicKor/Multiturn", c, fmt(v))
        if v is not None:
            multi_vals.append(v)
    add_row(rows, "LogicKor/Multiturn", "Multiturn Avg",
            safe_mean(multi_vals))

    ov = j.get("Overall", {})
    add_row(rows, "LogicKor/Overall", "Single turn", fmt(ov.get("single_turn")))
    add_row(rows, "LogicKor/Overall", "Multiturn", fmt(ov.get("multi_turn")))
    add_row(rows, "LogicKor/Overall", "Avg", fmt(ov.get("overall")))

    return rows

def main():
    ap = argparse.ArgumentParser(description="Aggregate 4 eval files into one CSV like the provided sheet image.")
    ap.add_argument("folder", type=str, help="Folder containing lm_eval.json, logickor.json, eval_singlecall.tsv, eval_dialogue.tsv")
    ap.add_argument("--output", type=str, default="summary.csv", help="Output CSV filename (default: summary.csv)")
    args = ap.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")

    lm_eval_path = find_file(folder, ["lm_evals.json"])
    logickor_path = find_file(folder, ["logickor.json"])
    single_tsv = find_file(folder, ["FunctionChat-Singlecall.inhouse.all.eval_report.tsv"])
    dialog_tsv = find_file(folder, ["FunctionChat-Dialog.inhouse.eval_report.tsv"])

    if not lm_eval_path:
        raise SystemExit("lm_evals.json not found in the folder.")
    if not logickor_path:
        raise SystemExit("logickor.json not found in the folder.")
    if not single_tsv:
        raise SystemExit("FunctionChat-Singlecall.inhouse.all.eval_report.tsv not found in the folder.")
    if not dialog_tsv:
        raise SystemExit("FunctionChat-Dialog.inhouse.eval_report.tsv not found in the folder.")

    rows: List[Dict[str, Any]] = []
    rows += parse_lm_eval(lm_eval_path)
    rows += parse_functionchat(folder, single_tsv, dialog_tsv)
    rows += parse_logickor(logickor_path)

    out_path = os.path.join(folder, args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Section", "Item", "Value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    main()