import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib
import pandas as pd

# Chargement du modèle final XGBoost et du scaler
model = joblib.load("final_xgboost_model.pkl")
scaler = joblib.load("scaler_xgboost.pkl")

# Configuration de la page
st.set_page_config(page_title="Portail Diabète", layout="wide")

# Menu latéral de navigation
with st.sidebar:
    choix = option_menu(
        menu_title="Navigation",
        options=["🏠 Accueil", "📘 Conseils santé", "👨‍⚕️ Consultation", "📚 Infos Médicales", "🌍 Infos Web"],
        icons=["house", "book", "person-circle", "book-half", "globe"],
        default_index=0
    )

# Page 1 : Accueil
if choix == "🏠 Accueil":
    st.title("🩺 Évaluation du Risque de Diabète")
    st.markdown("**Remplissez les informations ci-dessous pour estimer votre risque.**")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Grossesses", 0, 20, step=1)
        glucose = st.number_input("Taux de glucose (mg/dL)", 0, 200)
        blood_pressure = st.number_input("Pression artérielle (mm Hg)", 0, 140)
        skin_thickness = st.number_input("Épaisseur du pli cutané (mm)", 0, 100)
    with col2:
        insulin = st.number_input("Insuline (mu U/ml)", 0, 900)
        bmi = st.number_input("IMC", 0.0, 70.0)
        dpf = st.number_input("Antécédents familiaux (DPF)", 0.0, 3.0)
        age = st.number_input("Âge", 10, 100)

    predict = st.button("🔍 Prédire")

    if predict:
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        proba = model.predict_proba(input_scaled)[0][1]
        threshold = 0.65

        if proba >= threshold:
            st.error(f"⚠️ Risque ÉLEVÉ de diabète ({proba*100:.1f} %)")
        else:
            st.success(f"✅ Risque FAIBLE de diabète ({(1-proba)*100:.1f} %)")

        # Expander pour afficher les détails techniques de manière discrète
        with st.expander("🔍 Voir les détails de la prédiction"):
            st.markdown("### 📊 Détails techniques")

            st.markdown("**🔢 Données d'entrée (avant mise à l'échelle)**")
            st.write(pd.DataFrame(input_data, columns=[
                "Grossesses", "Glucose", "Pression Artérielle", "Épaisseur Peau",
                "Insuline", "IMC", "DPF", "Âge"
            ]))

            st.markdown("**🧪 Données transformées (standardisées)**")
            st.write(pd.DataFrame(input_scaled, columns=[
                "Grossesses", "Glucose", "Pression Artérielle", "Épaisseur Peau",
                "Insuline", "IMC", "DPF", "Âge"
            ]))

            st.markdown(f"**📈 Probabilité prédite :** `{proba:.4f}`")
            st.markdown(f"**📍 Seuil utilisé :** `{threshold}` (≥ {threshold} → Risque élevé)")

            # Importance des features
            st.markdown("### 📌 Importance des variables selon le modèle")
            importances = model.feature_importances_
            features = ["Grossesses", "Glucose", "Pression Artérielle", "Épaisseur Peau",
                        "Insuline", "IMC", "DPF", "Âge"]
            importance_df = pd.DataFrame({
                "Variable": features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Variable"))

# Page 2 : Conseils santé
elif choix == "📘 Conseils santé":
    st.title("📘 Conseils pour Prévenir le Diabète")
    st.markdown("""
    ### 🍽️ Alimentation équilibrée
    - Réduisez les sucres ajoutés
    - Privilégiez les graisses saines
    - Consommez des fibres

    ### 🚶 Activité physique régulière
    - 30 minutes d'exercice modéré par jour
    - Renforcement musculaire 2-3 fois/semaine

    ### 🧘‍♂️ Gestion du stress
    - Méditation, respiration, yoga
    - Sommeil de qualité

    ### 🩺 Suivi médical régulier
    - Surveillance de la glycémie
    - Contrôle de l'IMC
    """)

# Page 3 : Consultation
elif choix == "👨‍⚕️ Consultation":
    st.title("👨‍⚕️ Consultation Virtuelle")
    st.markdown("""
    ### 👩‍⚕️ Dr. Claire Dupont  
    **Spécialité :** Diabétologue  
    **Statut :** En ligne  

    ---
    #### 📋 Compte rendu :
    - Risque modéré détecté
    - Suivi recommandé
    - Consultez un professionnel pour confirmation
    """)

# Page 4 : Infos Médicales
elif choix == "📚 Infos Médicales":
    st.title("📚 Informations Médicales sur le Diabète")
    st.markdown("""
    Le diabète est une maladie chronique :
    
    - **Types** : Type 1, Type 2, Gestationnel  
    - **Symptômes** : Soif, fatigue, perte de poids  
    - **Prévention** : Alimentation, exercice, suivi régulier
    """)

# Page 5 : Infos Web
elif choix == "🌍 Infos Web":
    st.title("🌍 Informations sur le Web")
    st.markdown("""
    - [Assurance Maladie - Diabète](https://www.ameli.fr/assure/sante/themes/diabete)  
    - [Fédération des Diabétiques](https://www.federationdesdiabetiques.org/)  
    - [Diabetes Canada](https://www.diabetes.ca/fr)  
    - [OMS - Diabète](https://www.who.int/fr/news-room/fact-sheets/detail/diabetes)  
    - [INSERM](https://www.inserm.fr/)
    """)
