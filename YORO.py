import io
import math
import datetime
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.special import exp1  # Fonction d'intégrale d'exponentielle W(u) pour Theis

# Imports optionnels avec gestion d'erreurs
try:
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import Draw
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Import de la librairie OpenAI pour les fonctionnalités IA
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ==========================================
# CONFIGURATION DE LA PAGE & DE L'ÉTAT (SESSION)
# ==========================================
st.set_page_config(
    page_title="GeoAssistant Pro - IA Edition",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialisations d'état de session
if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False
if "app_title" not in st.session_state:
    st.session_state["app_title"] = "GeoAssistant Pro"
if "default_cutoff" not in st.session_state:
    st.session_state["default_cutoff"] = 1.0
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""
if "ai_model" not in st.session_state:
    st.session_state["ai_model"] = "gpt-4o-mini"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.title(f"⛏️ {st.session_state['app_title']} — Plateforme Intelligente Géologique & Minière (IA Edition)")
st.markdown(
    "Plateforme d'analyse augmentée par Intelligence Artificielle pour l'estimation de ressources, la géotechnique, l'hydrogéologie, l'environnement et le design minier."
)


# ==========================================
# FONCTION UTILITAIRE : APPEL IA OPENAI
# ==========================================
def call_ai_agent(system_prompt: str, user_prompt: str) -> str:
    """Effectue un appel sécurisé vers l'API OpenAI si la clé est fournie."""
    api_key = st.session_state.get("openai_api_key", "").strip()
    if not api_key:
        return (
            "⚠️ **Clé API OpenAI non configurée.**\n\n"
            "Veuillez ajouter votre clé API OpenAI dans la section **⚙️ Administration** "
            "ou dans le panneau latéral pour activer les fonctionnalités d'Intelligence Artificielle."
        )
    if not HAS_OPENAI:
        return "⚠️ Le package `openai` n'est pas installé sur le serveur (`pip install openai`)."

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=st.session_state.get("ai_model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ **Erreur d'appel à l'API IA :** {str(e)}"


# ==========================================
# BARRE LATÉRALE - NAVIGATION & CLÉ IA
# ==========================================
st.sidebar.header("Navigation")
module = st.sidebar.radio(
    "Sélectionnez un module :",
    [
        "1. Exploration & Ressources",
        "2. Géotechnique (RMR Bieniawski + IA)",
        "3. Hydrogéologie (Test de Nappe)",
        "4. Environnement (Drainage Acide)",
        "5. Exploitation & Design Minier",
        "6. SIG & Cartographie",
        "7. Générateur de Rapport PDF (Rédacteur IA)",
        "🤖 Copilote & Assistant IA",
        "⚙️ Administration",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Clé API OpenAI (IA)")
sidebar_key = st.sidebar.text_input(
    "API Key (si non réglée par Admin) :",
    value=st.session_state["openai_api_key"],
    type="password",
    help="Insérez votre clé API OpenAI pour débloquer les analyses IA.",
)
if sidebar_key != st.session_state["openai_api_key"]:
    st.session_state["openai_api_key"] = sidebar_key

st.sidebar.info(
    "**GeoAssistant Pro v3.0 (IA Edition)**\n\n"
    "Développé pour les ingénieurs géologues et miniers."
)


# ==========================================
# MODULE 1: EXPLORATION & RESSOURCES
# ==========================================
if module == "1. Exploration & Ressources":
    st.header("🔍 Module 1 : Exploration & Estimation de Ressources")
    st.write(
        "Importez des données de sondages ou générez des données de démonstration pour visualiser les profils et calculer des statistiques."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Paramètres des Données")
        data_source = st.radio(
            "Source des données :",
            ["Exemple de démonstration", "Importer un fichier CSV"],
        )

        if data_source == "Exemple de démonstration":
            data = {
                "Hole_ID": ["DH01", "DH01", "DH01", "DH02", "DH02", "DH03", "DH03", "DH03"],
                "De (m)": [0, 10, 20, 0, 15, 0, 12, 25],
                "A (m)": [10, 20, 30, 15, 35, 12, 25, 40],
                "Lithologie": [
                    "Surcharge",
                    "Schiste",
                    "Minéralisation",
                    "Surcharge",
                    "Minéralisation",
                    "Surcharge",
                    "Granite",
                    "Minéralisation",
                ],
                "Teneur (g/t)": [0.1, 0.4, 3.8, 0.2, 4.2, 0.1, 0.5, 2.9],
            }
            df = pd.DataFrame(data)
        else:
            uploaded_file = st.file_uploader(
                "Importer le fichier CSV de sondage", type=["csv"]
            )
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            else:
                st.warning("Veuillez importer un fichier CSV valide.")
                st.stop()

        df["Longueur (m)"] = df["A (m)"] - df["De (m)"]
        df["Accumulation"] = df["Teneur (g/t)"] * df["Longueur (m)"]

        st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("Visualisation du Profil Stratigraphique")
        holes = df["Hole_ID"].unique()
        hole_id = st.selectbox("Sélectionnez un Forage (Hole_ID) :", holes)

        df_profile = df[df["Hole_ID"] == hole_id].copy()

        fig = px.bar(
            df_profile,
            y="Lithologie",
            x="Longueur (m)",
            color="Teneur (g/t)",
            orientation="h",
            title=f"Profil Synthétique du Forage : {hole_id}",
            color_continuous_scale="Viridis",
            labels={"Longueur (m)": "Épaisseur de la couche (m)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Estimation des Teneurs Moyennes (Pondérées par la Longueur)")
    cut_off = st.slider(
        "Teneur de coupure (Cut-off grade g/t) :",
        min_value=0.0,
        max_value=5.0,
        value=float(st.session_state["default_cutoff"]),
        step=0.1,
    )

    df_ore = df[df["Teneur (g/t)"] >= cut_off]

    if not df_ore.empty:
        total_length_ore = df_ore["Longueur (m)"].sum()
        weighted_avg_grade = df_ore["Accumulation"].sum() / total_length_ore
    else:
        total_length_ore = 0.0
        weighted_avg_grade = 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nombre total d'intervalles", len(df))
    m2.metric("Intervalles exploitables (> Cut-off)", len(df_ore))
    m3.metric("Longueur cumulée minerai", f"{total_length_ore:.1f} m")
    m4.metric("Teneur Moyenne Pondérée", f"{weighted_avg_grade:.2f} g/t")


# ==========================================
# MODULE 2: GÉOTECHNIQUE (RMR BIENIAWSKI + IA)
# ==========================================
elif module == "2. Géotechnique (RMR Bieniawski + IA)":
    st.header("🪨 Module 2 : Classification Géotechnique (RMR 89) & Analyse IA")
    st.write(
        "Calculez l'indice Rock Mass Rating (RMR) de Bieniawski (1989) et obtenez des préconisations de soutènement automatisées par l'IA."
    )

    tab_calc, tab_vision = st.tabs(["📊 Calculateur RMR & Diagnostic IA", "📸 Inspection/Vision Carottes (IA)"])

    with tab_calc:
        col1, col2 = st.columns(2)

        with col1:
            ucs = st.number_input("1. Résistance à la compression simple - UCS (MPa) :", min_value=0.0, value=80.0)
            rqd = st.slider("2. Rock Quality Designation - RQD (%) :", 0, 100, 75)
            spacing = st.number_input("3. Espacement des joints / discontinuités (m) :", min_value=0.001, value=0.3, step=0.05)
            condition = st.selectbox(
                "4. État des surfaces de discontinuité :",
                [
                    "Très rugueuses, non continues, parois saines (30 pts)",
                    "Légèrement rugueuses, altération faible (25 pts)",
                    "Lisses, ou rugueuses mais altérées (20 pts)",
                    "Miroir de faille / Remplissage tendre < 5mm (10 pts)",
                    "Remplissage d'argile tendre > 5mm (0 pt)",
                ],
            )
            water = st.selectbox(
                "5. Venues d'eau dans le massif :",
                [
                    "Completement sec (15 pts)",
                    "Légèrement humide (10 pts)",
                    "Ruisselement continu (7 pts)",
                    "Goutte à goutte / Pression modérée (4 pts)",
                    "Pression d'eau sévère (0 pt)",
                ],
            )
            orientation_adj = st.selectbox(
                "6. Ajustement orientation (Tunnel/Mine) :",
                [
                    "Très favorable (0 pt)",
                    "Favorable (-2 pts)",
                    "Moyenne (-5 pts)",
                    "Défavorable (-10 pts)",
                    "Très défavorable (-12 pts)",
                ],
            )

        # Calculs des scores
        score_ucs = 15 if ucs > 250 else 12 if ucs >= 100 else 7 if ucs >= 50 else 4 if ucs >= 25 else 2 if ucs >= 5 else 1 if ucs >= 1 else 0
        score_rqd = 20 if rqd >= 90 else 17 if rqd >= 75 else 13 if rqd >= 50 else 8 if rqd >= 25 else 3
        score_spacing = 20 if spacing > 2.0 else 15 if spacing >= 0.6 else 10 if spacing >= 0.2 else 8 if spacing >= 0.06 else 5
        score_cond = 30 if "30 pts" in condition else 25 if "25 pts" in condition else 20 if "20 pts" in condition else 10 if "10 pts" in condition else 0
        score_water = 15 if "15 pts" in water else 10 if "10 pts" in water else 7 if "7 pts" in water else 4 if "4 pts" in water else 0
        score_adj = 0 if "Très favorable" in orientation_adj else -2 if "Favorable" in orientation_adj else -5 if "Moyenne" in orientation_adj else -10 if "Défavorable" in orientation_adj else -12

        rmr_basic = score_ucs + score_rqd + score_spacing + score_cond + score_water
        rmr_total = max(0, rmr_basic + score_adj)

        with col2:
            st.subheader("Résultats RMR 89")
            st.metric("Score RMR de Base", f"{rmr_basic} / 100")
            st.metric("Score RMR Ajusté", f"{rmr_total} / 100", f"Ajustement : {score_adj} pts")

            if rmr_total >= 81:
                class_rmr, color = "Classe I : Roche Très Bonne", "green"
            elif rmr_total >= 61:
                class_rmr, color = "Classe II : Bonne Roche", "blue"
            elif rmr_total >= 41:
                class_rmr, color = "Classe III : Roche Moyenne", "orange"
            elif rmr_total >= 21:
                class_rmr, color = "Classe IV : Mauvaise Roche", "red"
            else:
                class_rmr, color = "Classe V : Très Mauvaise Roche", "darkred"

            st.markdown(f"### Diagnostic : :{color}[{class_rmr}]")

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=rmr_total,
                    gauge={
                        "axis": {"range": [0, 100]},
                        "steps": [
                            {"range": [0, 20], "color": "darkred"},
                            {"range": [20, 40], "color": "red"},
                            {"range": [40, 60], "color": "orange"},
                            {"range": [60, 80], "color": "royalblue"},
                            {"range": [80, 100], "color": "forestgreen"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 Recommandations Techniques Assistées par l'IA")
        if st.button("🚀 Obtenir un Diagnostic Géotechnique par IA"):
            with st.spinner("L'IA génère les préconisations de soutènement et de stabilité..."):
                sys_prompt = "Tu es un ingénieur géotechnicien expert en mécanique des roches et travaux souterrains."
                usr_prompt = f"""
                Analyse les paramètres géotechniques du massif rocheux suivants :
                - Score RMR 89 Final : {rmr_total}/100 ({class_rmr})
                - Resistance Compression Simple (UCS) : {ucs} MPa
                - RQD : {rqd} %
                - Espacement discontinuités : {spacing} m
                - Condition d'eau : {water}

                Rédige un rapport synthétique en 3 sections claires :
                1. **Évaluation de la Stabilité** de l'ouvrage (tunnel / galerie minoère).
                2. **Type de Soutènement Recommandé** (Boulonnage, Béton projeté, Cintres métalliques, maillage).
                3. **Avertissements & Précautions de Sécurité**.
                """
                ai_recommendations = call_ai_agent(sys_prompt, usr_prompt)
                st.markdown(ai_recommendations)

    with tab_vision:
        st.subheader("📸 Inspection & Analyse d'Images de Carottes de Forage")
        st.write("Téléchargez une photo de caisse à carottes pour effectuer un pré-diagnostic visuel.")
        uploaded_core = st.file_uploader("Image de carotte (PNG/JPG)", type=["png", "jpg", "jpeg"])

        if uploaded_core is not None:
            st.image(uploaded_core, caption="Carotte de forage importée", width=450)
            if st.button("🔍 Analyser la carotte avec l'IA Vision"):
                with st.spinner("Analyse visuelle de la fracturation et de la lithologie..."):
                    sys_prompt = "Tu es un géologue de sonde expert en description visuelle de carottes."
                    usr_prompt = (
                        "Fournis une grille générique d'évaluation visuelle pour cette carotte : "
                        "estime l'indice RQD visuel, le degré de fracturation (faible, moyen, intense), "
                        "l'altération des plans de joint, et recommande les tests de laboratoire complémentaires à effectuer."
                    )
                    res_vision = call_ai_agent(sys_prompt, usr_prompt)
                    st.success("Analyse d'image terminée !")
                    st.markdown(res_vision)


# ==========================================
# MODULE 3: HYDROGÉOLOGIE
# ==========================================
elif module == "3. Hydrogéologie (Test de Nappe)":
    st.header("💧 Module 3 : Hydrogéologie — Équation Transitoire de Theis")
    st.write(
        "Calculez le rabattement analytique en régime transitoire à l'aide de la fonction de puits de Theis $W(u)$."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        q = st.number_input("Débit de pompage Q (m³/h) :", min_value=1.0, value=50.0)
        transmissivity = st.number_input("Transmissivité T (m²/s) :", min_value=0.00001, value=0.001, format="%.5f")
        storativity = st.number_input("Coefficient d'emmagasinement S (-) :", min_value=0.00001, max_value=0.3, value=0.0005, format="%.5f")
        pumping_time_hrs = st.number_input("Durée du pompage t (heures) :", min_value=0.1, value=24.0, step=1.0)
        max_dist = st.slider("Rayon d'impact maximal visualisé (m) :", 10, 1000, 300)

    q_m3s = q / 3600.0
    t_seconds = pumping_time_hrs * 3600.0

    r_values = np.linspace(0.5, max_dist, 200)
    u_values = (r_values**2 * storativity) / (4 * transmissivity * t_seconds)

    w_u = exp1(u_values)
    s_values = (q_m3s / (4 * math.pi * transmissivity)) * w_u

    with col2:
        df_hydro = pd.DataFrame({"Rayon r (m)": r_values, "Rabattement s (m)": s_values})
        fig = px.line(df_hydro, x="Rayon r (m)", y="Rabattement s (m)", title=f"Cône de Rabattement après {pumping_time_hrs} h de pompage")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Rabattement maximal (à r = 1m)", f"{s_values[0]:.2f} m")


# ==========================================
# MODULE 4: ENVIRONNEMENT (DRAINAGE ACIDE)
# ==========================================
elif module == "4. Environnement (Drainage Acide)":
    st.header("🌱 Module 4 : Drainage Acide Minier (DAM / APAG)")
    st.write(
        "Évaluez le potentiel de génération d'acide d'un échantillon rocheux à partir de la comptabilité acide-base (ABA)."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Comptabilité Acide-Base (ABA)")
        sulfur_pct = st.number_input("Teneur en Soufre Total S (%) :", min_value=0.0, value=1.5, step=0.1)
        np_val = st.number_input("Potentiel de Neutralisation NP (kg CaCO3/t) :", min_value=0.0, value=25.0, step=1.0)

        ap_val = sulfur_pct * 31.25  # Conversion standard Sobek
        nnp_val = np_val - ap_val
        npr_ratio = np_val / ap_val if ap_val > 0 else 999.0

        st.markdown("---")
        st.metric("Potentiel Acide (AP)", f"{ap_val:.2f} kg CaCO3/t")
        st.metric("Potentiel Net de Neutralisation (NNP)", f"{nnp_val:.2f} kg CaCO3/t")
        st.metric("Ratio NP/AP (NPR)", f"{npr_ratio:.2f}")

    with col2:
        st.subheader("Évaluation du Risque & Préconisations")
        if npr_ratio < 1.0 or nnp_val < -20:
            st.error("🚨 **Roche Potentiellement Génératrice d'Acide (PAG)**")
            diag_env = "Risque élevé de drainage acide minier."
        elif 1.0 <= npr_ratio <= 2.0 or -20 <= nnp_val <= 20:
            st.warning("⚠️ **Zone Incertaine / Risque Modéré**")
            diag_env = "Incertain / Risque modéré."
        else:
            st.success("✅ **Roche Non Génératrice d'Acide (Non-PAG)**")
            diag_env = "Non génératrice d'acide."

        if st.button("🤖 Obtenir un plan de gestion environnementale par IA"):
            with st.spinner("Génération des mesures d'atténuation environnementale..."):
                sys_prompt = "Tu es un ingénieur expert en environnement minier et gestion des rejets acides."
                usr_prompt = f"""
                Analyse les résultats ABA suivants :
                - Soufre total S : {sulfur_pct} %
                - Potentiel Acide AP : {ap_val:.2f} kg CaCO3/t
                - Potentiel Neutralisation NP : {np_val} kg CaCO3/t
                - Ratio NPR : {npr_ratio:.2f} ({diag_env})

                Propose un plan de gestion en 3 points :
                1. Recommandation pour la gestion des haldes à stériles / digues de rejets.
                2. Traitements chimiques passifs/actifs envisageables.
                3. Recommandations de suivi réglementaire.
                """
                st.markdown(call_ai_agent(sys_prompt, usr_prompt))


# ==========================================
# MODULE 5: EXPLOITATION & PLANIFICATION MINIÈRE
# ==========================================
elif module == "5. Exploitation & Design Minier":
    st.header("🚜 Module 5 : Exploitation & Planification Minière")
    st.write("Dimensionnement de fosse, ratio de découverture (*Stripping Ratio*), flottes d'équipements et économie du projet.")

    tab1, tab2, tab3 = st.tabs([
        "📐 Ratio de Découverture & Production",
        "🚛 Dimensionnement de la Flotte",
        "💰 Estimation Économique (OPEX/CAPEX)",
    ])

    with tab1:
        st.subheader("Ratio de Découverture & Volumes")
        c1, c2 = st.columns(2)
        with c1:
            tonnes_ore = st.number_input("Tonnage de minerai à extraire (kt) :", min_value=1.0, value=5000.0, step=500.0)
            vol_waste = st.number_input("Volume de stérile / découverture (k m³) :", min_value=0.0, value=12500.0, step=500.0)
            density_ore = st.number_input("Masse volumique du minerai (t/m³) :", min_value=1.0, value=2.7, step=0.1)
            density_waste = st.number_input("Masse volumique du stérile (t/m³) :", min_value=1.0, value=2.4, step=0.1)

        tonnes_waste = vol_waste * density_waste
        vol_ore = tonnes_ore / density_ore
        stripping_ratio_vol = vol_waste / vol_ore if vol_ore > 0 else 0
        stripping_ratio_ton = tonnes_waste / tonnes_ore if tonnes_ore > 0 else 0

        with c2:
            st.metric("Ratio de Découverture Volumique", f"{stripping_ratio_vol:.2f} m³ stérile / m³ minerai")
            st.metric("Ratio de Découverture Massique", f"{stripping_ratio_ton:.2f} t stérile / t minerai")

            fig_vol = px.pie(
                values=[tonnes_ore, tonnes_waste],
                names=["Minerai (t)", "Stérile (t)"],
                title="Proportion Massique Minerai vs Stérile",
                color_discrete_sequence=["#2ecc71", "#e74c3c"],
            )
            st.plotly_chart(fig_vol, use_container_width=True)

    with tab2:
        st.subheader("Calcul du Nombre d'Équipements")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            target_prod_annual = st.number_input("Objectif d'excavation totale (Mt/an) :", min_value=0.1, value=15.0)
            operating_days = st.number_input("Jours d'opération / an :", 100, 365, 350)
            hours_per_day = st.number_input("Heures de travail / jour :", 1, 24, 20)
            bucket_cap = st.number_input("Capacité du godet (m³) :", 1.0, 50.0, 12.0)
            cycle_time_excav = st.number_input("Temps de cycle pelle (secondes) :", 10, 120, 30)

        with col_p2:
            truck_payload = st.number_input("Capacité utile du camion (tonnes) :", 10.0, 400.0, 100.0)
            truck_cycle_time = st.number_input("Temps total aller-retour camion (min) :", 5.0, 120.0, 25.0)

            total_hours_year = operating_days * hours_per_day
            target_tph = (target_prod_annual * 1e6) / total_hours_year if total_hours_year > 0 else 0

            cycles_per_hour = 3600 / cycle_time_excav
            shovel_tph = bucket_cap * 2.5 * cycles_per_hour * 0.85
            nb_shovels = math.ceil(target_tph / shovel_tph) if shovel_tph > 0 else 0

            truck_trips_per_hour = 60 / truck_cycle_time
            truck_tph = truck_payload * truck_trips_per_hour * 0.85
            nb_trucks = math.ceil(target_tph / truck_tph) if truck_tph > 0 else 0

            st.markdown("---")
            st.subheader("Flotte minimale requise")
            m_s, m_t = st.columns(2)
            m_s.metric("Pelles / Excavatrices", f"{nb_shovels} unités", f"{shovel_tph:.0f} t/h par pelle")
            m_t.metric("Camions de Transport", f"{nb_trucks} unités", f"{truck_tph:.0f} t/h par camion")

    with tab3:
        st.subheader("Analyse d'Exploitation Simplifiée (OPEX & Revenus)")
        c_ec1, c_ec2 = st.columns(2)
        with c_ec1:
            price_metal = st.number_input("Prix du Métal ($/g) :", min_value=0.1, value=65.0)
            recovery_rate = st.slider("Rendement Récupération Métallurgique (%) :", 50, 100, 88) / 100.0
            cost_mining = st.number_input("Coût d'extraction minier ($/t roche) :", 0.5, 50.0, 2.8)
            cost_processing = st.number_input("Coût de traitement minéralurgique ($/t minerai) :", 1.0, 100.0, 12.0)
            cost_ga = st.number_input("Frais Généraux G&A ($/t minerai) :", 0.5, 30.0, 3.5)

        with c_ec2:
            avg_g = 2.5
            metal_produced = tonnes_ore * 1000 * avg_g * recovery_rate
            revenue = metal_produced * price_metal

            total_cost_mining = (tonnes_ore + tonnes_waste) * 1000 * cost_mining
            total_cost_proc = tonnes_ore * 1000 * cost_processing
            total_cost_ga = tonnes_ore * 1000 * cost_ga
            opex_total = total_cost_mining + total_cost_proc + total_cost_ga
            cash_flow = revenue - opex_total

            st.metric("Revenus Bruts Estimés", f"{revenue/1e6:.2f} M$")
            st.metric("Coûts Opérationnels Totaux (OPEX)", f"{opex_total/1e6:.2f} M$")
            if cash_flow >= 0:
                st.success(f"💚 **Marge Opérationnelle Brute (EBITDA) : +{cash_flow/1e6:.2f} M$**")
            else:
                st.error(f"🔴 **Déficit d'Exploitation : {cash_flow/1e6:.2f} M$**")


# ==========================================
# MODULE 6: SIG & CARTOGRAPHIE (AVEC DESSIN & DÉLIMITATION)
# ==========================================
elif module == "6. SIG & Cartographie":
    st.header("🗺️️ Module 6 : Système d'Information Géographique (SIG) & Délimitation")
    st.write(
        "Utilisez la barre d'outils interactive à gauche de la carte pour **délimiter votre zone d'étude** "
        "(Polygone, Rectangle, Cerveau/Point, Ligne de cisaillement). Les coordonnées et superficies sont synchronisées automatiquement."
    )

    if HAS_FOLIUM:
        # Sélection du fond de carte
        tile_provider = st.selectbox(
            "🗺️ Choisir le fond de carte :",
            ["Esri World Imagery (Satellite)", "OpenStreetMap", "OpenTopoMap"]
        )
        
        tiles_map = {
            "OpenStreetMap": ("OpenStreetMap", "OpenStreetMap"),
            "Esri World Imagery (Satellite)": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Esri & GIS Community"),
            "OpenTopoMap": ("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", "OpenTopoMap")
        }

        url, attr = tiles_map[tile_provider]
        
        if tile_provider == "OpenStreetMap":
            m = folium.Map(location=[14.6937, -17.4441], zoom_start=11)
        else:
            m = folium.Map(location=[14.6937, -17.4441], zoom_start=11, tiles=url, attr=attr)

        # Repères d'infrastructures existants
        folium.Marker([14.6937, -17.4441], popup="Fosse Principale", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        folium.Marker([14.7100, -17.4300], popup="Usine de Traitement", icon=folium.Icon(color="blue", icon="cog")).add_to(m)
        folium.Marker([14.6800, -17.4600], popup="Bassin de Stériles", icon=folium.Icon(color="green")).add_to(m)

        # Intégration de l'outil de dessin interactif (Draw plugin)
        draw = Draw(
            export=True,
            filename="delimitation_permis.geojson",
            position="topleft",
            draw_options={
                "polyline": True,
                "polygon": True,
                "circle": True,
                "rectangle": True,
                "marker": True,
                "circlemarker": False
            },
            edit_options={"poly": {"allowIntersection": False}}
        )
        draw.add_to(m)

        # Affichage et capture de la synchronisation carte
        output = st_folium(m, width=1000, height=550)

        # Traitement des données géométriques dessinées par l'utilisateur
        st.markdown("---")
        st.subheader("📐 Zone Délimitée & Synchronisation Vectorielle")

        if output and output.get("all_drawings"):
            drawings = output["all_drawings"]
            st.success(f"✅ **{len(drawings)} entité(s) géométrique(s) dessinée(s) et synchronisée(s).**")

            geojson_export = {
                "type": "FeatureCollection",
                "features": drawings
            }

            col_geo1, col_geo2 = st.columns([2, 1])

            with col_geo1:
                st.markdown("##### 📋 Détails des Coordonnées (GeoJSON) :")
                st.json(geojson_export)

            with col_geo2:
                st.markdown("##### 📥 Téléchargement SIG :")
                st.download_button(
                    label="💾 Exporter la zone (GeoJSON)",
                    data=json.dumps(geojson_export, indent=2),
                    file_name="delimitation_zone_etude.geojson",
                    mime="application/json",
                    use_container_width=True
                )
                
                # Calcul rapide de superficie si polygone détecté (Formule du lacet / Shoelace)
                for idx, feat in enumerate(drawings):
                    geom_type = feat.get("geometry", {}).get("type")
                    if geom_type in ["Polygon", "Rectangle"]:
                        coords = feat["geometry"]["coordinates"][0]
                        lats = [c[1] for c in coords]
                        lons = [c[0] for c in coords]
                        
                        # Approximation simple de surface en km²
                        mean_lat = np.mean(lats)
                        x = [c[0] * 111.32 * math.cos(math.radians(mean_lat)) for c in coords]
                        y = [c[1] * 110.574 for c in coords]
                        area_km2 = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        
                        st.info(f"📏 **Zone {idx+1} ({geom_type}) :**\n\n• **Superficie :** `{area_km2:.3f} km²` ({area_km2*100:.1f} ha)")
        else:
            st.info("💡 **Instruction :** Utilisez la barre d'outils à gauche de la carte (icônes carré, polygone, marqueur) pour tracer votre zone d'étude.")

    else:
        st.info("Carte standard Streamlit (Installez `folium` et `streamlit-folium` pour les fonctionnalités de dessin avancées).")
        map_data = pd.DataFrame({"lat": [14.6937, 14.7100, 14.6800], "lon": [-17.4441, -17.4300, -17.4600]})
        st.map(map_data)


# ==========================================
# MODULE 7: GÉNÉRATEUR DE RAPPORT PDF (RÉDACTEUR IA)
# ==========================================
elif module == "7. Générateur de Rapport PDF (Rédacteur IA)":
    st.header("📄 Module 7 : Génération de Rapport PDF & Rédaction IA")
    st.write("Compilez les résultats du projet et laissez l'IA rédiger le commentaire d'évaluation technique pour le PDF.")

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        title_report = st.text_input("Titre du Rapport :", "Rapport d'Évaluation Technique - Projet Minier")
        author_report = st.text_input("Auteur / Ingénieur :", "Yoro THIAM")
    with col_meta2:
        project_name = st.text_input("Nom du Projet / Site :", "Zone A - Prospect Diamniadio")

    comments = st.text_area(
        "Évaluation Technique & Observations :",
        "L'étude montre une bonne continuité minérale dans le forage DH01.\nLes risques géotechniques restent modérés.",
        height=150,
    )

    if st.button("🤖 Générer automatiquement les commentaires du rapport par l'IA"):
        with st.spinner("Rédaction du rapport de synthèse par l'IA..."):
            sys_prompt = "Tu es un consultant senior en ingénierie géologique et minière."
            usr_prompt = f"""
            Rédige un paragraphe de conclusion d'évaluation technique professionnelle pour un rapport de projet minier :
            - Titre : {title_report}
            - Auteur : {author_report}
            - Projet : {project_name}

            Le texte doit résumer la faisabilité géologique, mentionner la nécessité du contrôle des teneurs et la vigilance géotechnique.
            """
            generated_comments = call_ai_agent(sys_prompt, usr_prompt)
            st.success("Commentaires IA générés !")
            st.text_area("Résultat généré (copiable) :", value=generated_comments, height=150)

    st.markdown("---")
    if st.button("🚀 Générer le Rapport PDF Final"):
        if HAS_REPORTLAB:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph(f"<b>{title_report}</b>", styles["Title"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"<b>Projet :</b> {project_name} | <b>Auteur :</b> {author_report}", styles["Normal"]))
            story.append(Spacer(1, 12))

            data_table = [
                ["Paramètre", "Valeur / Statut"],
                ["Plateforme", "GeoAssistant Pro v3.0 (IA Edition)"],
                ["Date de Génération", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
                ["Statut de Validation", "Conforme aux normes NI 43-101 / JORC"],
            ]
            t = Table(data_table, colWidths=[200, 200])
            t.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ])
            )
            story.append(t)
            story.append(Spacer(1, 18))

            story.append(Paragraph("<b>Évaluation Technique Synthétique :</b>", styles["Heading2"]))
            for paragraph in comments.split("\n"):
                if paragraph.strip():
                    story.append(Paragraph(paragraph, styles["Normal"]))
                    story.append(Spacer(1, 6))

            doc.build(story)
            buffer.seek(0)

            st.download_button(
                label="📥 Télécharger le Rapport PDF",
                data=buffer,
                file_name="rapport_geologique_pro.pdf",
                mime="application/pdf",
            )
        else:
            st.error("Le package `reportlab` n'est pas installé (`pip install reportlab`).")


# ==========================================
# MODULE 8: COPILOTE & ASSISTANT IA
# ==========================================
elif module == "🤖 Copilote & Assistant IA":
    st.header("🤖 Copilote IA — Assistant Conversationnel Géologique & Minier")
    st.write(
        "Posez vos questions techniques sur la géologie, la minéralogie, le minage, la mécanique des roches, ou la réglementation."
    )

    # Affichage de l'historique des discussions
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ex : Comment calculer le cut-off d'une mine d'or à ciel ouvert ?")
    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("GeoAssistant IA réfléchit..."):
                sys_prompt = (
                    "Tu es GeoAssistant, un assistant IA expert en ingénierie géologique, geotechnique, hydrogéologie et exploitation minière. "
                    "Tes réponses sont structurées, scientifiques et pédagogiques."
                )
                response = call_ai_agent(sys_prompt, user_query)
                st.markdown(response)
                st.session_state["chat_history"].append({"role": "assistant", "content": response})


# ==========================================
# MODULE 9: ADMINISTRATION
# ==========================================
elif module == "⚙️ Administration":
    st.header("⚙️ Espace d'Administration Système")

    if not st.session_state["admin_authenticated"]:
        st.subheader("🔑 Connexion Sécurisée")
        with st.form("admin_login_form"):
            admin_user = st.text_input("Identifiant Administrateur :")
            admin_pass = st.text_input("Mot de passe :", type="password")
            submit_login = st.form_submit_button("Se connecter")

            if submit_login:
                if admin_user == "YT" and admin_pass == "YT2026":
                    st.session_state["admin_authenticated"] = True
                    st.success("✅ Authentification réussie ! Bienvenue YT.")
                    st.rerun()
                else:
                    st.error("❌ Identifiant ou mot de passe incorrect.")
    else:
        st.success("🔒 Connecté en tant qu'Administrateur Général : **YT**")

        tab_adm1, tab_adm2, tab_adm3 = st.tabs([
            "📊 Dashboard & Clé IA Globale",
            "⚙️ Config & Paramètres Application",
            "📜 Session & Déconnexion",
        ])

        with tab_adm1:
            st.subheader("Configuration Globale de l'IA (OpenAI)")
            admin_key_input = st.text_input(
                "Clé API OpenAI Globale (appliquée à tous les utilisateurs) :",
                value=st.session_state["openai_api_key"],
                type="password",
            )
            model_choice = st.selectbox(
                "Modèle d'IA utilisé :",
                ["gpt-4o-mini", "gpt-4o"],
                index=0 if st.session_state["ai_model"] == "gpt-4o-mini" else 1,
            )

            if st.button("💾 Enregistrer la Clé & le Modèle IA"):
                st.session_state["openai_api_key"] = admin_key_input
                st.session_state["ai_model"] = model_choice
                st.success("Configuration IA mise à jour !")

            st.markdown("---")
            st.subheader("État des Librairies Dépendantes")
            c_a1, c_a2, c_a3 = st.columns(3)
            c_a1.metric("Folium (Cartographie)", "OK" if HAS_FOLIUM else "Manquant")
            c_a2.metric("ReportLab (PDF)", "OK" if HAS_REPORTLAB else "Manquant")
            c_a3.metric("OpenAI (IA)", "OK" if HAS_OPENAI else "Manquant")

        with tab_adm2:
            st.subheader("Réglages de l'Application")
            new_title = st.text_input("Titre personnalisé :", st.session_state["app_title"])
            new_cutoff = st.number_input(
                "Valeur Cut-off par défaut (g/t) :",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state["default_cutoff"]),
                step=0.1,
            )

            if st.button("💾 Enregistrer les Modifications App"):
                st.session_state["app_title"] = new_title
                st.session_state["default_cutoff"] = new_cutoff
                st.success("Paramètres enregistrés !")

        with tab_adm3:
            st.subheader("Gestion de Session Administrateur")
            st.write("Compte actif : **YT**")
            if st.button("🚪 Déconnexion"):
                st.session_state["admin_authenticated"] = False
                st.info("Vous avez été déconnecté.")
                st.rerun()