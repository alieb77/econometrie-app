"""
generer_exemple.py
==================
Génère un fichier Excel de démonstration : prix hebdomadaires de 4 indices
boursiers fictifs sur ~10 ans.

Le modèle (à facteur commun) est construit pour illustrer chaque test :
  * volatilité groupée (GARCH) -> ADF, ARCH-LM, GARCH ont du sens ;
  * un épisode de CRISE où la volatilité du facteur commun augmente ;
  * MASI  : son exposition au facteur AUGMENTE en crise -> vraie CONTAGION ;
  * CAC40 : exposition quasi constante -> simple INTERDÉPENDANCE
            (la corrélation monte seulement à cause de la volatilité).

Usage :  python generer_exemple.py
Sortie :  exemple_donnees.xlsx
"""
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)

N = 520  # ~10 ans de données hebdomadaires
dates = pd.date_range("2015-01-04", periods=N, freq="W-SUN")

# Fenêtre de crise
crisis_start, crisis_end = 300, 345
is_crisis = np.zeros(N, dtype=bool)
is_crisis[crisis_start:crisis_end] = True


def garch_series(n, omega=0.04, alpha=0.16, beta=0.80, mu=0.0, df=6):
    """Série GARCH(1,1) à innovations Student-t d'écart-type inconditionnel ~1
    (omega = 1 - alpha - beta). Sert de brique de base 'volatilité groupée'.
    Les innovations Student-t (df=6) donnent des queues épaisses réalistes.
    Les chocs ne sont PAS amplifiés -> processus stable, sans explosion."""
    z = rng.standard_t(df, n) / np.sqrt(df / (df - 2))  # variance unitaire
    sigma2 = np.empty(n)
    e = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha * e[t - 1] ** 2 + beta * sigma2[t - 1]
        e[t] = np.sqrt(sigma2[t]) * z[t]
    return mu + e


# --- Facteur commun (régional/mondial), volatilité accrue en crise --------
factor = garch_series(N)                 # écart-type ~1, groupé
regime = np.where(is_crisis, 2.4, 1.0)   # la vol du facteur ~2,4x en crise
factor = factor * regime


def market(beta_normal, beta_crisis, mu, idio_scale):
    """Rendements (%) = mu + beta_t * facteur + bruit idiosyncratique (GARCH).
    beta_crisis > beta_normal => l'exposition augmente en crise (CONTAGION)."""
    beta = np.where(is_crisis, beta_crisis, beta_normal)
    idio = garch_series(N) * idio_scale
    return mu + beta * factor + idio


# Facteur retardé d'une semaine : crée un effet de précédence (Granger)
factor_lag = np.concatenate([[0.0], factor[:-1]])

# Marché-source : essentiellement le facteur (sa vol explose en crise)
r_source = 0.10 + 2.00 * factor + garch_series(N) * 0.40
# MASI : exposition qui BONDIT en crise -> contagion ; + réaction RETARDÉE
# au facteur/source (marché émergent qui réagit avec un décalage)
# => SOURCE_US devrait causer MASI au sens de Granger.
r_masi = market(beta_normal=0.55, beta_crisis=1.45, mu=0.05, idio_scale=1.15) \
    + 0.50 * factor_lag
# CAC40 : forte interdépendance mais exposition ~constante -> pas de contagion
r_cac = market(beta_normal=1.50, beta_crisis=1.55, mu=0.05, idio_scale=1.5)
# SP500 : suit étroitement le facteur
r_sp = market(beta_normal=1.80, beta_crisis=1.90, mu=0.04, idio_scale=0.8)


def to_prices(returns, p0):
    """Indice de prix à partir des rendements (%) : P_t = P_0 · exp(Σ r/100)."""
    return p0 * np.exp(np.cumsum(returns / 100.0))


df = pd.DataFrame(
    {
        "Date": dates,
        "MASI": to_prices(r_masi, 10000),
        "CAC40": to_prices(r_cac, 5000),
        "SP500": to_prices(r_sp, 2000),
        "SOURCE_US": to_prices(r_source, 3000),
    }
)

out = os.path.join(SCRIPT_DIR, "exemple_donnees.xlsx")
df.to_excel(out, index=False, sheet_name="Prix hebdomadaires")
print(f"Fichier genere : {out}")
print(f"  {N} observations hebdomadaires du {dates[0].date()} au {dates[-1].date()}")
print(f"  Indices : MASI, CAC40, SP500, SOURCE_US")
print(f"  Episode de crise : {dates[crisis_start].date()} -> {dates[crisis_end-1].date()}")
