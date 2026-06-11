"""
econometrics.py
================
Moteur de calcul économétrique pour l'analyse de séries financières
hebdomadaires (prix ou rendements d'indices boursiers).

Aucune dépendance à Streamlit : toutes les fonctions sont « pures » et
renvoient des dictionnaires / DataFrames. Elles sont donc réutilisables
dans un notebook Jupyter, un script ou directement dans un mémoire.

Tests implémentés
-----------------
1. ADF                 -> stationnarité (racine unitaire)
2. ARCH-LM             -> présence d'effets ARCH (hétéroscédasticité conditionnelle)
3. GARCH univarié      -> modélisation de la volatilité conditionnelle
4. Causalité de Granger-> liens de précédence temporelle entre deux séries
5. Forbes-Rigobon      -> test de contagion ajusté de l'hétéroscédasticité
"""
from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model


# ======================================================================
# 1. Préparation des données
# ======================================================================
def to_log_returns(prices: pd.Series, pct: bool = True) -> pd.Series:
    """Rendements logarithmiques r_t = ln(P_t / P_{t-1}).

    pct=True -> rendements exprimés en pourcentage (×100), recommandé
    pour la stabilité numérique de l'estimation GARCH.
    """
    r = np.log(prices.astype(float) / prices.astype(float).shift(1))
    if pct:
        r = r * 100.0
    return r.rename(prices.name)


def to_simple_returns(prices: pd.Series, pct: bool = True) -> pd.Series:
    """Rendements arithmétiques r_t = P_t / P_{t-1} - 1."""
    r = prices.astype(float).pct_change()
    if pct:
        r = r * 100.0
    return r.rename(prices.name)


def descriptive_stats(series: pd.Series) -> dict:
    """Statistiques descriptives usuelles + test de normalité de Jarque-Bera."""
    s = pd.Series(series).dropna().astype(float)
    jb = stats.jarque_bera(s)
    return {
        "Observations": int(s.shape[0]),
        "Moyenne": float(s.mean()),
        "Médiane": float(s.median()),
        "Écart-type": float(s.std(ddof=1)),
        "Minimum": float(s.min()),
        "Maximum": float(s.max()),
        "Skewness (asymétrie)": float(stats.skew(s)),
        "Kurtosis (normale = 3)": float(stats.kurtosis(s, fisher=False)),
        "Jarque-Bera (stat.)": float(jb.statistic),
        "Jarque-Bera (p-value)": float(jb.pvalue),
        "Normale (5 %)": bool(jb.pvalue >= 0.05),
    }


# ======================================================================
# 2. Test ADF — stationnarité
# ======================================================================
def adf_test(
    series: pd.Series,
    regression: str = "c",
    autolag: str = "AIC",
    max_lag: int | None = None,
) -> dict:
    """Test de Dickey-Fuller Augmenté.

    H0 : la série possède une racine unitaire (NON stationnaire).
    H1 : la série est stationnaire.
    -> p-value < 5 %  =>  on rejette H0  =>  série STATIONNAIRE.

    regression : 'c' (constante), 'ct' (constante + tendance), 'n' (aucune).
    """
    s = pd.Series(series).dropna().astype(float)
    res = adfuller(s, maxlag=max_lag, regression=regression, autolag=autolag)
    adf_stat, pvalue, usedlag, nobs, crit, _icbest = res
    return {
        "statistique_adf": float(adf_stat),
        "p_value": float(pvalue),
        "retards_utilises": int(usedlag),
        "n_obs": int(nobs),
        "valeurs_critiques": {k: float(v) for k, v in crit.items()},
        "stationnaire_5pct": bool(pvalue < 0.05),
        "regression": regression,
    }


# ======================================================================
# 3. Test ARCH-LM — effets ARCH
# ======================================================================
def arch_lm_test(series: pd.Series, nlags: int = 12, demean: bool = True) -> dict:
    """Test du multiplicateur de Lagrange d'Engle (ARCH-LM).

    H0 : absence d'effets ARCH (pas d'hétéroscédasticité conditionnelle).
    H1 : présence d'effets ARCH.
    -> p-value < 5 %  =>  effets ARCH présents  =>  un GARCH est justifié.

    Le test est appliqué sur la série centrée (résidus d'un modèle à moyenne
    constante), conformément à la pratique courante.
    """
    s = pd.Series(series).dropna().astype(float)
    if demean:
        s = s - s.mean()
    try:
        out = het_arch(s, nlags=nlags)
    except TypeError:  # compatibilité anciennes versions de statsmodels
        out = het_arch(s, maxlag=nlags)
    lm_stat, lm_p, f_stat, f_p = out[:4]
    return {
        "lm_stat": float(lm_stat),
        "lm_p_value": float(lm_p),
        "f_stat": float(f_stat),
        "f_p_value": float(f_p),
        "nlags": int(nlags),
        "effets_arch_5pct": bool(lm_p < 0.05),
    }


# ======================================================================
# 4. GARCH univarié
# ======================================================================
def garch_fit(
    series: pd.Series,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    mean: str = "Constant",
    dist: str = "normal",
    vol: str = "GARCH",
) -> dict:
    """Estimation d'un modèle GARCH univarié via le package `arch`.

    p : ordre ARCH (chocs passés au carré)
    q : ordre GARCH (variances conditionnelles passées)
    o : ordre asymétrique (>0 => modèle GJR-GARCH / effet de levier)
    mean : 'Constant', 'Zero', 'AR'
    dist : 'normal', 't' (Student), 'skewt', 'ged'
    vol  : 'GARCH', 'EGARCH', 'APARCH', 'FIGARCH'

    Persistance = somme alpha + beta (+ o/2 si terme asymétrique).
    Proche de 1 => chocs de volatilité très persistants.
    """
    s = pd.Series(series).dropna().astype(float)
    am = arch_model(s, mean=mean, vol=vol, p=p, o=o, q=q, dist=dist, rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off")

    params = res.params
    alpha = float(sum(v for k, v in params.items() if k.startswith("alpha")))
    beta = float(sum(v for k, v in params.items() if k.startswith("beta")))
    gamma = float(sum(v for k, v in params.items() if k.startswith("gamma")))
    persistence = alpha + beta + gamma / 2.0

    # Diagnostic sur les résidus standardisés : reste-t-il des effets ARCH ?
    resid_std = pd.Series(res.std_resid).dropna()
    arch_resid = arch_lm_test(resid_std, nlags=min(12, max(2, len(resid_std) // 5)),
                              demean=True)

    return {
        "params": {k: float(v) for k, v in params.items()},
        "pvalues": {k: float(v) for k, v in res.pvalues.items()},
        "tstats": {k: float(v) for k, v in res.tvalues.items()},
        "std_errors": {k: float(v) for k, v in res.std_err.items()},
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "alpha": alpha,
        "beta": beta,
        "persistence": float(persistence),
        "conditional_volatility": pd.Series(res.conditional_volatility, name="vol_cond"),
        "std_resid": resid_std,
        "arch_residuals": arch_resid,
        "summary": str(res.summary()),
        "spec": f"{vol}({p},{o},{q}) — moyenne {mean} — loi {dist}",
        "_res": res,
    }


# ======================================================================
# 5. Causalité de Granger
# ======================================================================
def granger_causality(y: pd.Series, x: pd.Series, maxlag: int = 4) -> pd.DataFrame:
    """Teste si `x` cause `y` au sens de Granger (x précède y).

    H0 : x ne cause pas y (les retards de x n'améliorent pas la prévision de y).
    -> p-value < 5 %  =>  x CAUSE y au sens de Granger.

    ATTENTION : les séries doivent être STATIONNAIRES (utiliser les
    rendements, pas les prix). On teste tous les retards de 1 à `maxlag`.
    """
    df = pd.concat([y, x], axis=1).dropna()
    data = df.values
    sink = io.StringIO()
    with redirect_stdout(sink), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = grangercausalitytests(data, maxlag=maxlag)

    rows = []
    for lag in range(1, maxlag + 1):
        test = res[lag][0]
        f_val, f_p = test["ssr_ftest"][0], test["ssr_ftest"][1]
        chi2, chi2_p = test["ssr_chi2test"][0], test["ssr_chi2test"][1]
        rows.append(
            {
                "Retard": lag,
                "F": float(f_val),
                "p-value (F)": float(f_p),
                "Chi²": float(chi2),
                "p-value (Chi²)": float(chi2_p),
                "Significatif (5 %)": bool(f_p < 0.05),
            }
        )
    return pd.DataFrame(rows)


def granger_bidirectional(a: pd.Series, b: pd.Series, maxlag: int = 4) -> dict:
    """Teste les deux directions de causalité entre a et b."""
    return {
        "a_vers_b": granger_causality(y=b, x=a, maxlag=maxlag),  # a -> b
        "b_vers_a": granger_causality(y=a, x=b, maxlag=maxlag),  # b -> a
        "nom_a": a.name,
        "nom_b": b.name,
    }


# ======================================================================
# 6. Test de contagion de Forbes-Rigobon (2002)
# ======================================================================
def forbes_rigobon_test(
    source: pd.Series,
    recipient: pd.Series,
    crisis_mask: pd.Series,
    one_sided: bool = True,
) -> dict:
    """Test de contagion de Forbes & Rigobon (2002),
    « No Contagion, Only Interdependence ».

    Idée : pendant une crise, la corrélation entre marchés augmente
    mécaniquement à cause de la hausse de la volatilité du marché-source
    (biais d'hétéroscédasticité). Forbes-Rigobon CORRIGE ce biais pour
    distinguer la véritable contagion d'une simple interdépendance.

    Paramètres
    ----------
    source       : rendements du marché où naît la crise (sert à l'ajustement
                   de volatilité).
    recipient    : rendements de l'autre marché (potentiellement « contaminé »).
    crisis_mask  : série booléenne, True = période de crise, False = période stable.

    Corrélation ajustée :
        ρ* = ρ_crise / sqrt( 1 + δ · (1 − ρ_crise²) )
        avec δ = Var_crise(source) / Var_stable(source) − 1

    Test (transformation z de Fisher) :
        H0 : ρ* = ρ_stable   (pas de contagion, simple interdépendance)
        H1 : ρ* > ρ_stable   (contagion)
    """
    df = pd.concat([source, recipient], axis=1).dropna()
    df.columns = ["source", "recipient"]
    mask = crisis_mask.reindex(df.index).fillna(False).astype(bool)

    stable = df[~mask]
    crisis = df[mask]
    n_stable, n_crisis = len(stable), len(crisis)
    if n_stable < 5 or n_crisis < 5:
        raise ValueError(
            f"Trop peu d'observations (stable={n_stable}, crise={n_crisis}). "
            "Chaque sous-période doit compter au moins 5 points."
        )

    rho_stable = float(np.corrcoef(stable["source"], stable["recipient"])[0, 1])
    rho_crisis = float(np.corrcoef(crisis["source"], crisis["recipient"])[0, 1])

    var_stable = float(stable["source"].var(ddof=1))
    var_crisis = float(crisis["source"].var(ddof=1))
    delta = var_crisis / var_stable - 1.0

    denom = np.sqrt(1.0 + delta * (1.0 - rho_crisis**2))
    rho_adj = rho_crisis / denom
    rho_adj = float(np.clip(rho_adj, -0.999999, 0.999999))

    # Test de Fisher : comparaison corrélation ajustée (crise) vs stable
    se = np.sqrt(1.0 / (n_crisis - 3) + 1.0 / (n_stable - 3))
    z_adj = (np.arctanh(rho_adj) - np.arctanh(rho_stable)) / se
    z_unadj = (np.arctanh(rho_crisis) - np.arctanh(rho_stable)) / se

    if one_sided:
        p_adj = float(1.0 - stats.norm.cdf(z_adj))
        p_unadj = float(1.0 - stats.norm.cdf(z_unadj))
    else:
        p_adj = float(2.0 * (1.0 - stats.norm.cdf(abs(z_adj))))
        p_unadj = float(2.0 * (1.0 - stats.norm.cdf(abs(z_unadj))))

    return {
        "n_stable": n_stable,
        "n_crisis": n_crisis,
        "rho_stable": rho_stable,
        "rho_crisis_brut": rho_crisis,
        "rho_crisis_ajuste": rho_adj,
        "delta_volatilite": float(delta),
        "var_stable_source": var_stable,
        "var_crisis_source": var_crisis,
        "z_ajuste": float(z_adj),
        "p_value_ajuste": p_adj,
        "z_brut": float(z_unadj),
        "p_value_brut": p_unadj,
        "contagion_5pct": bool(p_adj < 0.05),
        "one_sided": one_sided,
        "nom_source": source.name,
        "nom_recipient": recipient.name,
    }


# ======================================================================
# 7. DCC-GARCH bivarié (Engle, 2002)
# ======================================================================
def dcc_garch_bivariate(
    a_series: pd.Series,
    b_series: pd.Series,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    dist: str = "t",
    vol: str = "GARCH",
) -> dict:
    """Modèle DCC-GARCH bivarié d'Engle (2002), estimé en deux étapes.

    Étape 1 : un GARCH univarié sur chaque série -> résidus standardisés zₜ.
    Étape 2 : dynamique de la matrice de corrélation conditionnelle
        Qₜ = (1 − a − b)·Q̄ + a·(zₜ₋₁ zₜ₋₁') + b·Qₜ₋₁
        Rₜ = diag(Qₜ)^(−½) · Qₜ · diag(Qₜ)^(−½)
    où a, b ≥ 0 et a + b < 1 (stationnarité de la corrélation).

    La corrélation conditionnelle ρₜ = Rₜ[0,1] est renvoyée semaine par semaine.
    """
    from scipy.optimize import minimize

    # --- alignement des deux séries sur les mêmes dates -----------------
    df = pd.concat([a_series, b_series], axis=1).dropna()
    nom_a = a_series.name if a_series.name is not None else "A"
    nom_b = b_series.name if b_series.name is not None else "B"
    df.columns = [nom_a, nom_b]

    # --- étape 1 : GARCH univarié sur chaque série ----------------------
    g_a = garch_fit(df[nom_a], p=p, q=q, o=o, dist=dist, vol=vol)
    g_b = garch_fit(df[nom_b], p=p, q=q, o=o, dist=dist, vol=vol)

    Z = pd.concat([g_a["std_resid"].rename(nom_a),
                   g_b["std_resid"].rename(nom_b)], axis=1).dropna()
    z = Z.values
    T = z.shape[0]
    if T < 30:
        raise ValueError("Trop peu d'observations communes pour estimer le DCC (min. 30).")

    Qbar = np.cov(z, rowvar=False)  # matrice 2×2 inconditionnelle

    def filtre_rho(a, b):
        """Filtre la corrélation conditionnelle ρₜ pour des (a, b) donnés."""
        q11 = Qbar[0, 0]; q22 = Qbar[1, 1]; q12 = Qbar[0, 1]
        rho = np.empty(T)
        rho[0] = q12 / np.sqrt(q11 * q22)
        for t in range(1, T):
            za, zb = z[t - 1, 0], z[t - 1, 1]
            q11 = Qbar[0, 0] * (1 - a - b) + a * za * za + b * q11
            q22 = Qbar[1, 1] * (1 - a - b) + a * zb * zb + b * q22
            q12 = Qbar[0, 1] * (1 - a - b) + a * za * zb + b * q12
            rho[t] = q12 / np.sqrt(q11 * q22)
        return rho

    def neg_loglik(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 0.9999:
            return 1e12
        rho = np.clip(filtre_rho(a, b), -0.9999, 0.9999)
        det = 1.0 - rho ** 2
        z1, z2 = z[:, 0], z[:, 1]
        quad = (z1 ** 2 - 2.0 * rho * z1 * z2 + z2 ** 2) / det
        ll = -0.5 * np.sum(np.log(det) + quad)
        return -ll

    res = minimize(neg_loglik, x0=[0.02, 0.95], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    a_hat, b_hat = float(res.x[0]), float(res.x[1])
    loglik = float(-res.fun)

    # --- écarts-types numériques (Hessienne par différences finies) -----
    se_a = se_b = p_a = p_b = None
    try:
        def hessienne(f, x, eps=1e-4):
            n = len(x); H = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
                    xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
                    xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
                    xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
                    H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps ** 2)
            return H
        H = hessienne(neg_loglik, np.array([a_hat, b_hat]))
        cov = np.linalg.inv(H)
        se_a, se_b = float(np.sqrt(abs(cov[0, 0]))), float(np.sqrt(abs(cov[1, 1])))
        p_a = float(2 * (1 - stats.norm.cdf(abs(a_hat / se_a))))
        p_b = float(2 * (1 - stats.norm.cdf(abs(b_hat / se_b))))
    except Exception:
        pass

    rho_t = pd.Series(np.clip(filtre_rho(a_hat, b_hat), -0.9999, 0.9999),
                      index=Z.index, name="rho_conditionnelle")
    rho_uncond = float(np.corrcoef(z[:, 0], z[:, 1])[0, 1])

    # volatilités conditionnelles des marges, alignées sur ρₜ (pour le portefeuille)
    sigma_a = g_a["conditional_volatility"].reindex(rho_t.index)
    sigma_b = g_b["conditional_volatility"].reindex(rho_t.index)

    return {
        "a": a_hat, "b": b_hat,
        "a_se": se_a, "b_se": se_b, "a_p": p_a, "b_p": p_b,
        "persistance": a_hat + b_hat,
        "loglik": loglik,
        "convergence": bool(res.success),
        "rho_t": rho_t,
        "sigma_a": sigma_a, "sigma_b": sigma_b,
        "rho_moyenne": float(rho_t.mean()),
        "rho_min": float(rho_t.min()),
        "rho_max": float(rho_t.max()),
        "rho_ecart_type": float(rho_t.std(ddof=1)),
        "rho_inconditionnelle": rho_uncond,
        "garch_a": g_a, "garch_b": g_b,
        "nom_a": nom_a, "nom_b": nom_b,
        "n_obs": T,
        "spec": f"DCC({p},{q}) — GARCH {vol} marges, loi {dist}",
    }


def dcc_garch_multivariate(
    series_list,
    noms=None,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    dist: str = "t",
    vol: str = "GARCH",
) -> dict:
    """Modèle DCC-GARCH multivarié d'Engle (2002) pour N séries (N ≥ 2).

    Généralisation matricielle de la version bivariée :
        Qₜ = (1 − a − b)·Q̄ + a·(zₜ₋₁ zₜ₋₁') + b·Qₜ₋₁
        Rₜ = diag(Qₜ)^(−½) · Qₜ · diag(Qₜ)^(−½)
    où zₜ est le vecteur des résidus standardisés des N marges GARCH, et
    Rₜ est la matrice N×N de corrélation conditionnelle. Les paramètres
    scalaires (a, b) sont communs à toutes les paires (DCC « scalaire »).

    On renvoie la matrice de corrélation moyenne, l'indice d'intégration
    (corrélation moyenne entre tous les couples, semaine par semaine) et
    la corrélation conditionnelle de chaque couple.
    """
    from scipy.optimize import minimize

    # --- alignement des séries sur les mêmes dates ----------------------
    if noms is None:
        noms = [s.name if s.name is not None else f"S{i+1}"
                for i, s in enumerate(series_list)]
    noms = list(noms)
    df = pd.concat([s.rename(n) for s, n in zip(series_list, noms)], axis=1).dropna()
    N = df.shape[1]
    if N < 2:
        raise ValueError("Au moins deux séries sont nécessaires pour le DCC.")

    # --- étape 1 : un GARCH univarié par série --------------------------
    garchs, zcols, sigmas = {}, [], {}
    for c in noms:
        g = garch_fit(df[c], p=p, q=q, o=o, dist=dist, vol=vol)
        garchs[c] = g
        zcols.append(g["std_resid"].rename(c))
        sigmas[c] = g["conditional_volatility"]
    Z = pd.concat(zcols, axis=1).dropna()
    z = Z.values
    T = z.shape[0]
    if T < 30:
        raise ValueError("Trop peu d'observations communes pour estimer le DCC (min. 30).")

    Qbar = np.cov(z, rowvar=False)            # matrice N×N inconditionnelle

    def neg_loglik(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 0.9999:
            return 1e12
        Q = Qbar.copy()
        ll = 0.0
        for t in range(T):
            if t > 0:
                zz = np.outer(z[t - 1], z[t - 1])
                Q = Qbar * (1 - a - b) + a * zz + b * Q
            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0 or not np.isfinite(logdet):
                return 1e12
            try:
                sol = np.linalg.solve(R, z[t])
            except np.linalg.LinAlgError:
                return 1e12
            ll += -0.5 * (logdet + float(z[t] @ sol))
        return -ll if np.isfinite(ll) else 1e12

    res = minimize(neg_loglik, x0=[0.02, 0.95], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    a_hat, b_hat = float(res.x[0]), float(res.x[1])
    loglik = float(-res.fun)

    # --- écarts-types numériques (Hessienne par différences finies) -----
    se_a = se_b = p_a = p_b = None
    try:
        def hessienne(f, x, eps=1e-4):
            n = len(x); H = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
                    xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
                    xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
                    xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
                    H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps ** 2)
            return H
        H = hessienne(neg_loglik, np.array([a_hat, b_hat]))
        cov = np.linalg.inv(H)
        se_a, se_b = float(np.sqrt(abs(cov[0, 0]))), float(np.sqrt(abs(cov[1, 1])))
        p_a = float(2 * (1 - stats.norm.cdf(abs(a_hat / se_a))))
        p_b = float(2 * (1 - stats.norm.cdf(abs(b_hat / se_b))))
    except Exception:
        pass

    # --- reconstruction des matrices Rₜ avec (â, b̂) --------------------
    R_all = np.empty((T, N, N))
    Q = Qbar.copy()
    for t in range(T):
        if t > 0:
            zz = np.outer(z[t - 1], z[t - 1])
            Q = Qbar * (1 - a_hat - b_hat) + a_hat * zz + b_hat * Q
        d = np.sqrt(np.diag(Q))
        R_all[t] = np.clip(Q / np.outer(d, d), -0.9999, 0.9999)

    iu = np.triu_indices(N, k=1)              # couples (i < j)
    paires = [(noms[i], noms[j]) for i, j in zip(*iu)]
    rho_pairs = {
        (noms[i], noms[j]): pd.Series(R_all[:, i, j], index=Z.index,
                                      name=f"{noms[i]}-{noms[j]}")
        for i, j in zip(*iu)
    }
    rho_pairs_df = pd.DataFrame(
        {f"{i} | {j}": s.values for (i, j), s in rho_pairs.items()}, index=Z.index
    )
    corr_moyenne_t = pd.Series(R_all[:, iu[0], iu[1]].mean(axis=1),
                               index=Z.index, name="corr_moyenne")
    R_moyenne = pd.DataFrame(R_all.mean(axis=0), index=noms, columns=noms)
    R_finale = pd.DataFrame(R_all[-1], index=noms, columns=noms)
    R_uncond = pd.DataFrame(Qbar / np.outer(np.sqrt(np.diag(Qbar)),
                                            np.sqrt(np.diag(Qbar))),
                            index=noms, columns=noms)
    sigma_df = pd.DataFrame(sigmas).reindex(Z.index)

    return {
        "a": a_hat, "b": b_hat,
        "a_se": se_a, "b_se": se_b, "a_p": p_a, "b_p": p_b,
        "persistance": a_hat + b_hat,
        "loglik": loglik,
        "convergence": bool(res.success),
        "noms": noms,
        "paires": paires,
        "rho_pairs": rho_pairs,
        "rho_pairs_df": rho_pairs_df,
        "corr_moyenne_t": corr_moyenne_t,
        "corr_moyenne": float(corr_moyenne_t.mean()),
        "R_moyenne": R_moyenne,
        "R_finale": R_finale,
        "R_inconditionnelle": R_uncond,
        "sigma": sigma_df,
        "garchs": garchs,
        "n_obs": T,
        "spec": f"DCC({p},{q}) multivarié — {N} séries, GARCH {vol} marges, loi {dist}",
    }


# ======================================================================
# 8. Spillover de volatilité — Diebold & Yilmaz (2012)
# ======================================================================
def garch_vol_matrix(
    returns_df: pd.DataFrame, p: int = 1, q: int = 1, o: int = 0,
    dist: str = "t", vol: str = "GARCH",
) -> pd.DataFrame:
    """Matrice des volatilités conditionnelles σₜ (un GARCH par colonne)."""
    out = {}
    for c in returns_df.columns:
        g = garch_fit(returns_df[c].dropna(), p=p, q=q, o=o, dist=dist, vol=vol)
        out[c] = g["conditional_volatility"]
    return pd.DataFrame(out).dropna()


def diebold_yilmaz(vol_df: pd.DataFrame, lags: int = 2, horizon: int = 10) -> dict:
    """Indice de spillover de Diebold & Yilmaz (2012), version généralisée
    (décomposition de variance d'erreur de prévision invariante à l'ordre, KPPS).

    Renvoie la table de connectivité (en %), les spillovers directionnels
    « vers les autres » / « depuis les autres », le spillover net, et l'indice
    de spillover total.
    """
    from statsmodels.tsa.api import VAR

    cols = list(vol_df.columns)
    k = len(cols)
    res = VAR(vol_df).fit(lags)
    Sigma = np.asarray(res.sigma_u)
    A = res.ma_rep(maxn=horizon)  # (horizon+1, k, k), A[0] = I
    sig_diag = np.diag(Sigma)

    theta = np.zeros((k, k))
    for i in range(k):
        denom = sum(A[h][i, :] @ Sigma @ A[h][i, :].T for h in range(horizon))
        for j in range(k):
            num = sum((A[h][i, :] @ Sigma[:, j]) ** 2 for h in range(horizon))
            theta[i, j] = (num / sig_diag[j]) / denom

    theta_tilde = theta / theta.sum(axis=1, keepdims=True)  # lignes normalisées à 1

    from_others = (theta_tilde.sum(axis=1) - np.diag(theta_tilde)) * 100.0  # reçu par i
    to_others = (theta_tilde.sum(axis=0) - np.diag(theta_tilde)) * 100.0     # émis par j
    net = to_others - from_others
    total = float((theta_tilde.sum() - np.trace(theta_tilde)) / k * 100.0)

    table = pd.DataFrame(theta_tilde * 100.0, index=cols, columns=cols)
    table["FROM (reçu)"] = from_others
    ligne_to = pd.Series(to_others, index=cols)
    ligne_to["FROM (reçu)"] = np.nan
    table.loc["TO (émis)"] = ligne_to

    return {
        "table": table,
        "to_others": pd.Series(to_others, index=cols),
        "from_others": pd.Series(from_others, index=cols),
        "net": pd.Series(net, index=cols),
        "total": total,
        "lags": lags, "horizon": horizon, "n_obs": int(vol_df.shape[0]),
    }


def dy_rolling_total(
    vol_df: pd.DataFrame, lags: int = 2, horizon: int = 10,
    window: int = 104, step: int = 1,
) -> pd.Series:
    """Indice de spillover total de Diebold-Yilmaz estimé en fenêtre glissante."""
    idx, vals = [], []
    n = len(vol_df)
    for end in range(window, n + 1, step):
        sub = vol_df.iloc[end - window:end]
        try:
            vals.append(diebold_yilmaz(sub, lags=lags, horizon=horizon)["total"])
            idx.append(vol_df.index[end - 1])
        except Exception:
            continue
    return pd.Series(vals, index=idx, name="spillover_total")


# ======================================================================
# 9. Portefeuille à variance minimale dynamique (covariance DCC)
# ======================================================================
def dcc_min_variance_portfolio(
    dcc_result: dict, r_a: pd.Series, r_b: pd.Series,
    allow_short: bool = True, periods_per_year: int = 52,
) -> dict:
    """Portefeuille à variance minimale à partir de la covariance conditionnelle
    DCC (Hₜ = Dₜ Rₜ Dₜ). Les poids varient dans le temps.

    Pour deux actifs :  w₁ₜ = (σ₂² − covₜ) / (σ₁² + σ₂² − 2·covₜ),  w₂ₜ = 1 − w₁ₜ.
    Compare la stratégie min-variance à un portefeuille 50/50 et aux actifs seuls,
    et fournit le ratio de couverture dynamique + l'efficacité de couverture.
    """
    rho = dcc_result["rho_t"]
    sa = dcc_result["sigma_a"]; sb = dcc_result["sigma_b"]
    nom_a = dcc_result["nom_a"]; nom_b = dcc_result["nom_b"]

    d = pd.concat([rho.rename("rho"), sa.rename("sa"), sb.rename("sb")], axis=1).dropna()
    cov = d["rho"] * d["sa"] * d["sb"]
    va = d["sa"] ** 2; vb = d["sb"] ** 2

    w1 = (vb - cov) / (va + vb - 2.0 * cov)
    if not allow_short:
        w1 = w1.clip(0.0, 1.0)
    w2 = 1.0 - w1

    port_var = w1 ** 2 * va + w2 ** 2 * vb + 2.0 * w1 * w2 * cov
    port_vol = np.sqrt(port_var)                       # volatilité conditionnelle (hebdo, %)
    vol_5050 = np.sqrt(0.25 * va + 0.25 * vb + 0.5 * cov)
    hedge_ratio = cov / vb                              # couvrir A par B : β = cov/Var(B)

    # rendements réalisés sans anticipation : poids en t-1 appliqués aux rendements en t
    R = pd.concat([r_a.rename("a"), r_b.rename("b")], axis=1).reindex(d.index)
    rp = (w1.shift(1) * R["a"] + w2.shift(1) * R["b"]).dropna()
    r_5050 = (0.5 * R["a"] + 0.5 * R["b"]).reindex(rp.index)
    ann = np.sqrt(periods_per_year)

    def vol_ann(x):
        return float(x.std(ddof=1) * ann)

    var_minvar = rp.var(ddof=1)
    reduction_vs_a = float(1.0 - var_minvar / R["a"].reindex(rp.index).var(ddof=1))
    reduction_vs_b = float(1.0 - var_minvar / R["b"].reindex(rp.index).var(ddof=1))
    reduction_vs_5050 = float(1.0 - var_minvar / r_5050.var(ddof=1))

    series = pd.DataFrame({
        f"Poids {nom_a}": w1, f"Poids {nom_b}": w2,
        "Vol. min-variance": port_vol, "Vol. 50/50": vol_5050,
        f"Vol. {nom_a}": d["sa"], f"Vol. {nom_b}": d["sb"],
        "Ratio de couverture": hedge_ratio,
    })

    resume = pd.DataFrame({
        "Volatilité annualisée (%)": [
            vol_ann(rp), vol_ann(r_5050),
            vol_ann(R["a"].reindex(rp.index)), vol_ann(R["b"].reindex(rp.index)),
        ],
    }, index=["Min-variance (DCC)", "Naïf 50/50", f"100 % {nom_a}", f"100 % {nom_b}"])

    return {
        "series": series,
        "resume": resume,
        "poids_moyen_a": float(w1.mean()), "poids_moyen_b": float(w2.mean()),
        "reduction_vs_a": reduction_vs_a, "reduction_vs_b": reduction_vs_b,
        "reduction_vs_5050": reduction_vs_5050,
        "hedge_ratio_moyen": float(hedge_ratio.mean()),
        "nom_a": nom_a, "nom_b": nom_b,
        "allow_short": allow_short,
        "rendements_portefeuille": rp,
    }
