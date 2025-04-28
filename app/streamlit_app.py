import streamlit as st
import requests
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000/predict"  # À remplacer par l'URL de déploiement plus tard

# --- Interface Utilisateur ---
st.set_page_config(page_title="API Keyword Predictor", layout="wide")
st.title("📊 Interface de Prédiction de Mots-Clés")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("Paramètres")
    max_keywords = st.slider("Nombre de mots-clés", 1, 10, 5)
    st.markdown("---")
    st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Zone de texte principale
user_input = st.text_area("Entrez votre texte ici :", height=200, placeholder="Ex: Le machine learning révolutionne la médecine...")

# Bouton de prédiction
if st.button("Prédire les mots-clés", type="primary"):
    if not user_input.strip():
        st.warning("Veuillez entrer un texte valide.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                # Appel à l'API FastAPI
                response = requests.post(
                    API_URL,
                    json={"text": user_input, "max_keywords": max_keywords},
                    timeout=10
                )
                response.raise_for_status()
                
                results = response.json()
                
                # Affichage des résultats
                st.success("Prédiction terminée !")
                
                # Layout en colonnes
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Mots-clés prédits")
                    for keyword in results["keywords"]:
                        st.markdown(f"- **{keyword}**")
                
                with col2:
                    st.subheader("Scores de confiance")
                    for score in results["probabilities"]:
                        st.progress(score)
                        st.caption(f"{score:.1%}")  # Format pourcentage
                        
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur API : {str(e)}")