# utils.py — EDA helpers (lisibles & paramétrables)

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def seaborn_missing_rate(df, top_n=None, figsize=(10,4), rotation=60):
    rates = df.isna().mean().sort_values(ascending=False)
    if top_n is not None:
        rates = rates.head(int(top_n))
    plt.figure(figsize=figsize)
    ax = rates.plot(kind="bar", rot=rotation)
    ax.set_title("Taux de valeurs manquantes par colonne")
    ax.set_ylabel("Taux de valeurs manquantes")
    ax.set_xlabel("Colonnes")
    plt.tight_layout()
    return ax

def plot_missing_bars_h(df, top_n=35, figsize=(10, 6)):
    rates = df.isna().mean().sort_values(ascending=True).tail(int(top_n))
    plt.figure(figsize=figsize)
    ax = rates.plot(kind="barh")
    ax.set_xlabel("Taux de valeurs manquantes")
    ax.set_ylabel("Colonnes")
    ax.set_title(f"Top {top_n} colonnes par taux de NA")
    plt.tight_layout()
    return ax

def msno_matrix_sample(df, n=3000, seed=42, figsize=(12, 4), sort='descending'):
    try:
        import missingno as msno
    except ImportError:
        raise ImportError("missingno n'est pas installé. pip install missingno")
    samp = df.sample(min(int(n), len(df)), random_state=seed)
    msno.matrix(samp, figsize=figsize, sparkline=False)
    msno.bar(df, figsize=figsize, sort=sort)

def _corr_heatmap_base(df, cols, method="spearman", target="TARGET",
                       top_by_abs_target=None, max_cols=30, mask_upper=True,
                       figsize=(9,8), annot=False, fmt=".2f", cmap="coolwarm",
                       linewidths=.5, rotate=45):
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    cols_wo_target = [c for c in cols if c != target]

    if top_by_abs_target and (target in df.columns):
        cor_target = df[cols_wo_target + [target]].corr(method=method)[target].abs().sort_values(ascending=False)
        selected = [c for c in cor_target.index if c != target][:int(top_by_abs_target)]
    else:
        selected = cols_wo_target[:int(max_cols)]

    corr = df[selected].corr(method=method)

    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    sns.heatmap(
        corr, mask=mask, vmin=-1, vmax=1, center=0, square=True,
        cmap=cmap, cbar_kws={"shrink": .8}, linewidths=linewidths, annot=annot, fmt=fmt
    )
    plt.title(f"Corrélation {method} (variables: {len(selected)})")
    plt.xticks(rotation=rotate, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return corr

def pearson_heatmap(df, num_cols=None, **kwargs):
    if num_cols is None:
        num_cols = df.select_dtypes(include="number").columns.tolist()
    return _corr_heatmap_base(df, num_cols, method="pearson", **kwargs)

def spearman_heatmap(df, num_cols=None, **kwargs):
    if num_cols is None:
        num_cols = df.select_dtypes(include="number").columns.tolist()
    return _corr_heatmap_base(df, num_cols, method="spearman", **kwargs)

def categorical_column_counts(df, top_n=15, figsize=(9,4)):
    cat_cols = df.select_dtypes(exclude="number").columns
    for col in cat_cols:
        vc = df[col].value_counts().head(int(top_n))
        plt.figure(figsize=figsize)
        sns.barplot(x=vc.values, y=vc.index)
        plt.title(f"{col} — top {int(top_n)} catégories")
        plt.xlabel("Effectif")
        plt.ylabel("")
        plt.tight_layout()
    return list(cat_cols)

def plot_top_categories_h(df, col, top_n=15, figsize=(10, 4)):
    vc = df[col].value_counts().head(int(top_n)).sort_values(ascending=True)
    plt.figure(figsize=figsize)
    sns.barplot(x=vc.values, y=vc.index)
    plt.title(f"{col} — top {top_n} catégories")
    plt.xlabel("Effectif")
    plt.ylabel("")
    plt.tight_layout()
    return vc

def num_vs_target_anova_diagnostic(df, num_cols, target="TARGET"):
    import scipy.stats as st
    res = []
    y = df[target].values
    for c in num_cols:
        if c == target:
            continue
        x = df[c].values
        try:
            q = pd.qcut(pd.Series(x).rank(method="first"), q=min(10, pd.Series(x).nunique()), duplicates="drop")
            groups = [df.loc[q == lvl, target] for lvl in q.unique()]
            _, p = st.f_oneway(*groups)
        except Exception:
            p = np.nan
        try:
            corr = np.corrcoef(pd.to_numeric(df[c], errors="coerce").fillna(0), y)[0,1] if np.std(y) else 0.0
        except Exception:
            corr = np.nan
        res.append({"feature": c, "p_value": p, "corr_target": corr})
    out = pd.DataFrame(res).sort_values("p_value")
    out["kept"] = ~out["p_value"].isna()
    out["reason"] = np.where(out["p_value"].isna(), "test failed", "")
    return out

def cat_vs_target_chi2_summary(df, cat_cols, target="TARGET"):
    import scipy.stats as st
    res = []
    for c in cat_cols:
        try:
            tab = pd.crosstab(df[c], df[target])
            chi2, p, dof, _ = st.chi2_contingency(tab)
        except Exception:
            p = np.nan
        res.append({"feature": c, "p_value": p})
    return pd.DataFrame(res).sort_values("p_value")

def filter_numeric_for_tests(df, cols, min_non_na=50, min_unique=2, target="TARGET"):
    out = []
    for c in cols:
        if c == target:
            continue
        s = df[c]
        if s.notna().sum() >= min_non_na and s.nunique(dropna=True) >= min_unique:
            out.append(c)
    return out

class suppress_degenerate_warnings:
    def __enter__(self):
        warnings.filterwarnings(
            "ignore",
            message="DegenerateDataWarning",
            module="scipy.stats._stats_py"
        )
    def __exit__(self, exc_type, exc, tb):
        warnings.resetwarnings()

def inspect_images_info(df, n=50):
    from PIL import Image
    import random
    sample = df.sample(n)

    sizes = []
    modes = []
    for fp in sample['filepath']:
        img = Image.open(fp)
        sizes.append(img.size)
        modes.append(img.mode)

    print("Résolutions rencontrées :", pd.Series(sizes).value_counts())
    print("Modes couleur rencontrés :", pd.Series(modes).value_counts())

def show_images_grid(filepaths, ncols=4, figsize=(12,8)):
    from PIL import Image
    import matplotlib.pyplot as plt

    n = len(filepaths)
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=figsize)
    for i, fp in enumerate(filepaths):
        img = Image.open(fp)
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(img, cmap='gray' if img.mode != "RGB" else None)
        plt.axis('off')
        plt.title(fp.split('/')[-1])
    plt.tight_layout()
    plt.show()

   