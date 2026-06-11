# 📈 Application d'économétrie financière des indices boursiers

Application web (Streamlit) qui, à partir d'un **fichier Excel/CSV de séries
hebdomadaires** de prix ou de rendements d'indices boursiers, réalise :

| # | Test | Question |
|---|------|----------|
| 1 | **ADF** (Augmented Dickey-Fuller) | La série est-elle stationnaire ? |
| 2 | **ARCH-LM** (Engle) | Y a-t-il des effets ARCH (volatilité groupée) ? |
| 3 | **GARCH univarié** (GARCH / EGARCH / GJR) | Modélisation de la volatilité conditionnelle |
| 4 | **Causalité de Granger** | Un indice en précède-t-il un autre ? |
| 5 | **Forbes-Rigobon** | Y a-t-il contagion entre deux marchés ? |

---

## 🚀 Lancement rapide

### Option A — double-clic (Windows)
Double-cliquez sur **`lancer_app.bat`**. L'application s'ouvre dans votre navigateur.

### Option B — ligne de commande
```powershell
cd C:\Users\hp\econometrie-app
python -m streamlit run app.py
```

> Première utilisation : si une dépendance manque, le lanceur l'installe
> automatiquement (ou faites `python -m pip install -r requirements.txt`).

---

## 📄 Format du fichier attendu

- **1ʳᵉ colonne** : les **dates** (hebdomadaires).
- **Colonnes suivantes** : un indice par colonne (**au moins 2** pour Granger et
  Forbes-Rigobon).
- Les valeurs peuvent être des **prix** (l'app calcule les rendements) ou
  directement des **rendements**.

| Date       | MASI   | CAC40 | SP500 |
|------------|--------|-------|-------|
| 2015-01-04 | 10000  | 5000  | 2000  |
| 2015-01-11 | 10045  | 5012  | 2007  |
| …          | …      | …     | …     |

### Fichier de démonstration
```powershell
python generer_exemple.py
```
Génère `exemple_donnees.xlsx` (indices fictifs *MASI, CAC40, SP500, SOURCE_US*)
avec un épisode de **contagion** (2020-2021) qui illustre tous les tests :
- *MASI* : véritable **contagion** depuis SOURCE_US ;
- *CAC40* : simple **interdépendance** (pas de contagion après ajustement).

---

## 🧪 Utilisation

1. Chargez votre fichier dans le panneau **📂 Données** (à gauche).
2. Sélectionnez la **colonne de dates** et indiquez **prix** ou **rendements**.
3. Parcourez les onglets :
   - **🗂️ Données & descriptif** — aperçu, statistiques, graphiques ;
   - **1️⃣ ADF** — stationnarité (prix vs rendements) ;
   - **2️⃣ ARCH-LM** — effets ARCH ;
   - **3️⃣ GARCH** — volatilité conditionnelle, persistance, diagnostics ;
   - **4️⃣ Granger** — causalité bidirectionnelle ;
   - **5️⃣ Forbes-Rigobon** — contagion (définir la fenêtre de crise) ;
   - **ℹ️ Méthodologie** — formules et interprétations.

Chaque test affiche une **conclusion en français** prête à citer dans un mémoire.

---

## 📁 Contenu du dossier

| Fichier | Rôle |
|---------|------|
| `app.py` | Interface web Streamlit |
| `econometrics.py` | Moteur de calcul (fonctions réutilisables, sans Streamlit) |
| `generer_exemple.py` | Génère le fichier Excel de démonstration |
| `lancer_app.bat` | Lanceur Windows (double-clic) |
| `requirements.txt` | Dépendances Python |

Le module `econometrics.py` est **autonome** : vous pouvez importer ses fonctions
(`adf_test`, `arch_lm_test`, `garch_fit`, `granger_causality`,
`forbes_rigobon_test`) directement dans un notebook ou un script.

---

## 📚 Références

- Dickey & Fuller (1979) — *Distribution of the Estimators for Autoregressive Time Series with a Unit Root*.
- Engle (1982) — *Autoregressive Conditional Heteroscedasticity*.
- Bollerslev (1986) — *Generalized Autoregressive Conditional Heteroskedasticity*.
- Granger (1969) — *Investigating Causal Relations by Econometric Models and Cross-spectral Methods*.
- Forbes & Rigobon (2002) — *No Contagion, Only Interdependence: Measuring Stock Market Comovements*, **Journal of Finance**.

*Construit avec `statsmodels`, `arch`, `scipy`, `pandas` et `streamlit`.*
