# Déployer l'application (gratuit, accessible à tous)

On utilise **Streamlit Community Cloud** : hébergement gratuit, l'app reçoit une
URL publique (`https://...streamlit.app`) que tu peux envoyer à n'importe qui.
Aucune carte bancaire, aucun serveur à gérer.

Il faut 2 comptes gratuits : **GitHub** (pour héberger le code) et
**Streamlit Cloud** (pour faire tourner l'app). Compte ~15 minutes la première fois.

---

## Étape 1 — Créer un compte GitHub

1. Va sur https://github.com → **Sign up**.
2. Choisis un identifiant, ton e-mail, un mot de passe. Confirme l'e-mail.

## Étape 2 — Installer Git sur ton PC

1. Télécharge : https://git-scm.com/download/win (le téléchargement démarre seul).
2. Installe en cliquant **Next** partout (les options par défaut conviennent).
3. Vérifie : ouvre **PowerShell** et tape `git --version`. Tu dois voir un numéro.

## Étape 3 — Mettre le code en ligne (une seule fois)

Ouvre **PowerShell** et colle ces commandes **une par une**
(remplace `TON-IDENTIFIANT` par ton pseudo GitHub) :

```powershell
cd "C:\Users\hp\econometrie-app"
git init
git add .
git commit -m "Application econometrie"
git branch -M main
```

Crée ensuite un dépôt vide sur GitHub :
1. https://github.com/new
2. **Repository name** : `econometrie-app`
3. Laisse **Public** coché (obligatoire pour la version gratuite de Streamlit).
4. Ne coche **rien** d'autre (pas de README). Clique **Create repository**.

Relie ton dossier au dépôt et envoie le code :

```powershell
git remote add origin https://github.com/TON-IDENTIFIANT/econometrie-app.git
git push -u origin main
```

Une fenêtre te demandera de te connecter à GitHub → autorise. Le code est en ligne.

## Étape 4 — Déployer sur Streamlit Cloud

1. Va sur https://share.streamlit.io → **Sign in with GitHub** → autorise.
2. Clique **Create app** (ou **New app**) → **Deploy a public app from GitHub**.
3. Remplis :
   - **Repository** : `TON-IDENTIFIANT/econometrie-app`
   - **Branch** : `main`
   - **Main file path** : `app.py`
4. Clique **Deploy**. La première construction prend 3–5 min (il installe `arch`,
   `statsmodels`, etc. — c'est normal que ce soit un peu long).

Quand c'est fini, tu obtiens une **URL publique**. Copie-la, envoie-la à qui tu
veux : la personne ouvre le lien dans son navigateur, dépose son fichier Excel,
et utilise l'app. Rien à installer de leur côté.

---

## Mettre à jour l'app plus tard

Dès que tu modifies `app.py` ou `econometrics.py`, renvoie les changements :

```powershell
cd "C:\Users\hp\econometrie-app"
git add .
git commit -m "mise a jour"
git push
```

Streamlit Cloud redéploie **tout seul** en ~1 min. L'URL ne change pas.

---

## Notes utiles

- **Confidentialité** : le code est public sur GitHub, mais **pas les fichiers
  Excel** que les visiteurs déposent — ceux-ci restent privés à leur session.
- **Mise en veille** : si personne n'utilise l'app pendant plusieurs jours, elle
  « s'endort ». Le premier visiteur la réveille en ~30 s en cliquant un bouton.
- **Version de Python** : fixée à 3.13 par le fichier `.python-version` (les
  paquets `arch`/`statsmodels` y sont stables).
- **Limite gratuite** : 1 Go de mémoire, largement suffisant pour ces tests.
