"""
train_model.py
Script de treino de um classificador supervisionado (XGBoost) para waveforms
de EPG, usando arquivos brutos .DXX + anotações .ANA (Stylet+/Stylet+a) como
ground truth.

USO:
    python train_model.py --data-dir /caminho/para/dataset --out modelo_epg.joblib

Estrutura esperada de --data-dir: um arquivo .ANA por gravação/inseto, com o
MESMO nome-base dos arquivos .DXX correspondentes. Ex:
    PsilideoT1-3-ch6.D01, ..., PsilideoT1-3-ch6.D09, PsilideoT1-3-ch6.ANA
    PsilideoT1-4-ch2.D01, ..., PsilideoT1-4-ch2.D06, PsilideoT1-4-ch2.ANA

O split de treino/teste é feito por GRAVAÇÃO (recording_id), nunca por janela
isolada, para evitar vazamento de dados entre janelas altamente
autocorrelacionadas do mesmo inseto.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from epg_ana import DEFAULT_CODE_MAP, build_labeled_dataset
from epg_core import extract_channel_key, group_and_stitch, natural_segment_number, parse_epg_bytes
from epg_features import extract_features_advanced, feature_columns


def discover_recordings(data_dir: str):
    """
    Agrupa arquivos .DXX por canal/gravação e localiza o .ANA correspondente
    (mesmo nome-base, extensão .ana, case-insensitive). A busca é tolerante a
    pequenas variações no nome-base (ex: underscore/espaço extra antes da
    extensão), comparando os nomes normalizados (sem esses caracteres).
    """
    dxx_paths = sorted(glob.glob(os.path.join(data_dir, "*.D[0-9][0-9]")))
    dxx_paths += sorted(glob.glob(os.path.join(data_dir, "*.d[0-9][0-9]")))

    by_channel = defaultdict(list)
    for p in dxx_paths:
        key = extract_channel_key(os.path.basename(p))
        by_channel[key].append(p)

    def normalize(name: str) -> str:
        return re.sub(r"[\s_\-]+", "", name).lower()

    ana_paths = glob.glob(os.path.join(data_dir, "*.[Aa][Nn][Aa]"))
    ana_by_norm = {normalize(os.path.splitext(os.path.basename(p))[0]): p for p in ana_paths}

    recordings = []
    for key, paths in by_channel.items():
        norm_key = normalize(key)
        ana_path = ana_by_norm.get(norm_key)
        if ana_path is None:
            print(f"[aviso] Sem arquivo .ANA para '{key}' — gravação ignorada no treino.")
            continue
        recordings.append(
            {
                "recording_id": key,
                "dxx_paths": sorted(paths, key=lambda pp: natural_segment_number(os.path.basename(pp))),
                "ana_path": ana_path,
            }
        )
    return recordings


def build_dataset(recordings, win_seconds: float, overlap: float, code_map=None):
    all_labeled = []
    for rec in recordings:
        parsed = [parse_epg_bytes(open(p, "rb").read(), os.path.basename(p)) for p in rec["dxx_paths"]]
        channel = group_and_stitch(parsed)[0]

        feat_df = extract_features_advanced(
            channel.values, channel.sample_rate, win_seconds=win_seconds, overlap=overlap
        )
        with open(rec["ana_path"], "rb") as f:
            ana_bytes = f.read()

        labeled = build_labeled_dataset(
            feat_df, ana_bytes, channel.total_seconds,
            code_map=code_map, recording_id=rec["recording_id"],
        )
        n_before = len(labeled)
        labeled = labeled.dropna(subset=["waveform"])
        print(
            f"  {rec['recording_id']}: {len(labeled)}/{n_before} janelas rotuladas "
            f"({channel.total_seconds/3600:.1f}h)"
        )
        all_labeled.append(labeled)

    return pd.concat(all_labeled, ignore_index=True) if all_labeled else pd.DataFrame()


def train(dataset: pd.DataFrame, test_size: float = 0.25, random_state: int = 42, n_blocks: int = 16):
    feat_cols = [c for c in feature_columns(dataset) if c not in ("recording_id",)]
    dataset = dataset.sort_values(["recording_id", "t_center_s"]).reset_index(drop=True)
    X = dataset[feat_cols].to_numpy()
    y_raw = dataset["waveform"].to_numpy()
    groups = dataset["recording_id"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    n_recordings = dataset["recording_id"].nunique()

    if n_recordings >= 2:
        # split por gravação -- teste com inseto(s) nunca vistos no treino
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        split_desc = f"por gravação ({n_recordings} gravações disponíveis)"
    else:
        # apenas 1 gravação anotada: GroupShuffleSplit não é possível.
        # Fallback: blocos temporais alternados distribuídos por toda a
        # gravação (em vez de um único corte 75/25, que pode isolar longos
        # trechos monótonos de uma única classe no final da gravação).
        print(
            "[aviso] Apenas 1 gravação anotada — GroupShuffleSplit não é "
            "possível. Usando split por blocos temporais alternados como "
            "aproximação. Isso NÃO valida generalização para novos insetos; "
            "colete mais gravações anotadas assim que possível."
        )
        n = len(dataset)
        block_id = np.arange(n) * n_blocks // n
        test_blocks = set(range(1, n_blocks, 2))
        test_mask = np.isin(block_id, list(test_blocks))
        train_idx, test_idx = np.where(~test_mask)[0], np.where(test_mask)[0]
        split_desc = f"blocos temporais alternados ({n_blocks} blocos, 1 gravação apenas)"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # pesos de classe para lidar com o desbalanceamento (Np/C dominantes,
    # E1/pd raros) -- ver Willett et al. 2016
    class_counts = pd.Series(y_train).value_counts()
    total = len(y_train)
    class_weight = {cls: total / (len(class_counts) * cnt) for cls, cnt in class_counts.items()}
    sample_weight = np.array([class_weight[label] for label in y_train])

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    present = sorted(set(y_test) | set(y_pred))
    target_names = le.inverse_transform(present)
    print("\n" + "=" * 60)
    print(f"RELATÓRIO DE CLASSIFICAÇÃO (split: {split_desc})")
    print("=" * 60)
    print(classification_report(y_test, y_pred, labels=present, target_names=target_names, zero_division=0))
    print("Matriz de confusão (linhas=real, colunas=previsto):")
    cm = confusion_matrix(y_test, y_pred, labels=present)
    print(pd.DataFrame(cm, index=target_names, columns=target_names).to_string())
    print(f"\nF1-macro: {f1_score(y_test, y_pred, average='macro', labels=present):.3f}")

    if n_recordings >= 2:
        n_train_rec = len(set(groups[train_idx]))
        n_test_rec = len(set(groups[test_idx]))
        print(f"\nGravações no treino: {n_train_rec} | Gravações no teste: {n_test_rec}")
    else:
        print(
            "\n⚠️  Resultado baseado em 1 único inseto -- NÃO representa "
            "generalização para novos indivíduos. Trate como prova de conceito."
        )

    return model, le, feat_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Pasta com os arquivos .DXX e .ANA")
    parser.add_argument("--out", default="modelo_epg.joblib", help="Caminho de saída do modelo treinado")
    parser.add_argument("--win-seconds", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    print(f"Buscando gravações (.DXX + .ANA) em: {args.data_dir}")
    recordings = discover_recordings(args.data_dir)
    print(f"{len(recordings)} gravação(ões) com anotação encontrada(s).\n")
    if not recordings:
        print("Nenhuma gravação anotada encontrada. Verifique se os .ANA têm o mesmo nome-base dos .DXX.")
        return

    print("Extraindo features e rotulando janelas...")
    dataset = build_dataset(recordings, args.win_seconds, args.overlap, code_map=DEFAULT_CODE_MAP)
    print(f"\nDataset final: {len(dataset)} janelas rotuladas de {dataset['recording_id'].nunique()} gravações.")
    print("Distribuição de classes:")
    print(dataset["waveform"].value_counts().to_string())

    model, le, feat_cols = train(dataset, test_size=args.test_size)

    joblib.dump(
        {
            "model": model,
            "label_encoder": le,
            "feature_columns": feat_cols,
            "win_seconds": args.win_seconds,
            "overlap": args.overlap,
        },
        args.out,
    )
    print(f"\nModelo salvo em: {args.out}")


if __name__ == "__main__":
    main()
