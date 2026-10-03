import io
import math
import datetime
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.special import exp1

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


# ==========================================
# CONFIGURATION DE LA PAGE & MASQUAGE DU HEADER NATIF
# ==========================================
st.set_page_config(
    page_title="GeoAssistant Pro",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Injection de CSS pour cacher le menu Streamlit natif (Share, GitHub, 3 dots, etc.) et styliser l'UI en format XXXL
st.markdown(
    """
    <style>
    /* Cacher le header natif Streamlit, le bouton Share, l'icône GitHub et le footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppHeader {display: none;}
    
    /* Style global et espacements XXXL */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 98% !important;
    }
    
    /* Style de la barre d'accès / login */
    .login-card {
        background-color: #1e293b;
        padding: 45px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialisations d'état de session
if "user_authenticated" not in st.session_state:
    st.session_state["user_authenticated"] = False
if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False
if "app_title" not in st.session_state:
    st.session_state["app_title"] = "GeoAssistant Pro"
if "default_cutoff" not in st.session_state:
    st.session_state["default_cutoff"] = 1.0


# ==========================================
# PAGE D'ACCÈS PRIVÉ (LOGIN A L'ENTRÉE)
# ==========================================
if not st.session_state["user_authenticated"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c_left, c_main, c_right = st.columns([1, 2, 1])
    
    with c_main:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 25px;">
                <h1 style="font-size: 3rem;">🔒 Accès Privé — GeoAssistant Pro</h1>
                <p style="font-size: 1.2rem; color: #64748b;">Plateforme d'analyse et d'ingénierie géologique & minière</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.form("login_form"):
            username_input = st.text_input("Identifiant :", placeholder="Entrez votre identifiant")
            password_input = st.text_input("Mot de passe :", type="password", placeholder="••••••••")
            submit_login = st.form_submit_button("🔑 Se connecter à la plateforme", use_container_width=True)
            
            if submit_login:
                if username_input == "YT" and password_input == "YT2026":
                    st.session_state["user_authenticated"] = True
                    st.session_state["admin_authenticated"] = True
                    st.success("✅ Accès autorisé !")
                    st.rerun()
                else:
                    st.error("❌ Identifiant ou mot de passe incorrect.")
    st.stop()


# ==========================================
# EN-TÊTE & BARRE DE NAVIGATION EN HAUT (XXXL DESIGN)
# ==========================================
st.markdown(
    f"""
    <div style="background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%); padding: 20px 30px; border-radius: 12px; margin-bottom: 20px; color: white;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin:0; font-size: 2.2rem; font-weight: 700;">⛏️ {st.session_state['app_title']}</h1>
                <p style="margin:5px 0 0 0; color: #94a3b8; font-size: 1.05rem;">Plateforme d'analyse pour l'estimation de ressources, géotechnique, hydrogéologie et design minier.</p>
            </div>
            <div style="text-align: right;">
                <span style="background-color: #334155; padding: 8px 16px; border-radius: 20px; font-weight: 600;">👤 Session Active</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation horizontale en haut
modules_list = [
    "1. Exploration",
    "2. Géotechnique",
    "3. Hydrogéologie",
    "4. Environnement",
    "5. Design Minier",
    "6. SIG & Carto",
    "7. Rapport PDF",
    "⚙️ Admin"
]

module = st.radio(
    "Navigation Principale",
    modules_list,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<hr style='margin-top: 5px; margin-bottom: 25px; border-color: #334155;'>", unsafe_allow_html=True)


# ==========================================
# MODULE 1: EXPLORATION & RESSOURCES
# ==========================================
if module == "1. Exploration":
    st.header("🔍 Module 1 : Exploration & Estimation de Ressources")
    st.write("Importez des données de sondages ou générez des données de démonstration pour visualiser les profils et calculer des statistiques.")

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
                    "Surcharge", "Schiste", "Minéralisation",
                    "Surcharge", "Minéralisation",
                    "Surcharge", "Granite", "Minéralisation",
                ],
                "Teneur (g/t)": [0.1, 0.4, 3.8, 0.2, 4.2, 0.1, 0.5, 2.9],
            }
            df = pd.DataFrame(data)
        else:
            uploaded_file = st.file_uploader("Importer le fichier CSV de sondage", type=["csv"])
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
# MODULE 2: GÉOTECHNIQUE
# ==========================================
elif module == "2. Géotechnique":
    st.header("🪨 Module 2 : Classification Géotechnique (RMR 89)")
    st.write("Calculez l'indice Rock Mass Rating (RMR) de Bieniawski (1989) pour évaluer la qualité du massif rocheux.")

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


# ==========================================
# MODULE 3: HYDROGÉOLOGIE
# ==========================================
elif module == "3. Hydrogéologie":
    st.header("💧 Module 3 : Hydrogéologie — Équation Transitoire de Theis")
    st.write("Calculez le rabattement analytique en régime transitoire à l'aide de la fonction de puits de Theis $W(u)$.")

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
# MODULE 4: ENVIRONNEMENT
# ==========================================
elif module == "4. Environnement":
    st.header("🌱 Module 4 : Drainage Acide Minier (DAM / APAG)")
    st.write("Évaluez le potentiel de génération d'acide d'un échantillon rocheux à partir de la comptabilité acide-base (ABA).")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Comptabilité Acide-Base (ABA)")
        sulfur_pct = st.number_input("Teneur en Soufre Total S (%) :", min_value=0.0, value=1.5, step=0.1)
        np_val = st.number_input("Potentiel de Neutralisation NP (kg CaCO3/t) :", min_value=0.0, value=25.0, step=1.0)

        ap_val = sulfur_pct * 31.25
        nnp_val = np_val - ap_val
        npr_ratio = np_val / ap_val if ap_val > 0 else 999.0

        st.markdown("---")
        st.metric("Potentiel Acide (AP)", f"{ap_val:.2f} kg CaCO3/t")
        st.metric("Potentiel Net de Neutralisation (NNP)", f"{nnp_val:.2f} kg CaCO3/t")
        st.metric("Ratio NP/AP (NPR)", f"{npr_ratio:.2f}")

    with col2:
        st.subheader("Évaluation du Risque")
        if npr_ratio < 1.0 or nnp_val < -20:
            st.error("🚨 **Roche Potentiellement Génératrice d'Acide (PAG)**")
            st.write("Risque élevé de drainage acide minier. Un plan de confinement et de traitement des effluents est nécessaire.")
        elif 1.0 <= npr_ratio <= 2.0 or -20 <= nnp_val <= 20:
            st.warning("⚠️ **Zone Incertaine / Risque Modéré**")
            st.write("Risque modéré. Des essais cinétiques complémentaires en colonne sont recommandés.")
        else:
            st.success("✅ **Roche Non Génératrice d'Acide (Non-PAG)**")
            st.write("La roche dispose d'une capacité de neutralisation suffisante.")


# ==========================================
# MODULE 5: EXPLOITATION & PLANIFICATION MINIÈRE
# ==========================================
elif module == "5. Design Minier":
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
# MODULE 6: SIG & CARTOGRAPHIE
# ==========================================
elif module == "6. SIG & Carto":
    st.header("🗺 Module 6 : Système d'Information Géographique (SIG)")
    st.write("Utilisez la barre d'outils interactive à gauche de la carte pour délimiter votre zone d'étude.")

    if HAS_FOLIUM:
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

        folium.Marker([14.6937, -17.4441], popup="Fosse Principale", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        folium.Marker([14.7100, -17.4300], popup="Usine de Traitement", icon=folium.Icon(color="blue", icon="cog")).add_to(m)
        folium.Marker([14.6800, -17.4600], popup="Bassin de Stériles", icon=folium.Icon(color="green")).add_to(m)

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

        output = st_folium(m, width=1200, height=550)

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
        else:
            st.info("💡 Utilisez la barre d'outils à gauche de la carte pour tracer votre zone d'étude.")

    else:
        st.info("Carte standard Streamlit (Installez `folium` et `streamlit-folium` pour les fonctionnalités de dessin avancées).")
        map_data = pd.DataFrame({"lat": [14.6937, 14.7100, 14.6800], "lon": [-17.4441, -17.4300, -17.4600]})
        st.map(map_data)


# ==========================================
# MODULE 7: GENERATEUR DE RAPPORT PDF (SANS INFOS PRIVÉES)
# ==========================================
elif module == "7. Rapport PDF":
    st.header("📄 Module 7 : Génération de Rapport PDF Technique")
    st.write("Compilez les résultats du projet pour générer un rapport PDF professionnel **excluant toute donnée confidentielle d'accès ou d'administration**.")

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        title_report = st.text_input("Titre du Rapport :", "Rapport d'Évaluation Technique - Projet Minier")
        author_report = st.text_input("Ingénieur / Responsable :", "Département Géologique & Ingénierie")
    with col_meta2:
        project_name = st.text_input("Nom du Projet / Site :", "Zone A - Prospect Diamniadio")

    comments = st.text_area(
        "Évaluation Technique & Observations :",
        "L'étude montre une bonne continuité minérale dans le forage DH01.\nLes risques géotechniques restent modérés.",
        height=150,
    )

    st.markdown("---")
    if st.button("🚀 Générer le Rapport PDF Technique"):
        if HAS_REPORTLAB:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Entête et titre (Sans identifiants privées)
            story.append(Paragraph(f"<b>{title_report}</b>", styles["Title"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"<b>Projet :</b> {project_name} | <b>Auteur :</b> {author_report}", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Table technique pure (Aucun identifiant/code d'accès n'est affiché)
            data_table = [
                ["Paramètre Technique", "Valeur / Statut"],
                ["Plateforme d'Analyse", "GeoAssistant Pro"],
                ["Date de Génération", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
                ["Conformité Technique", "Conforme aux normes standard de l'industrie"],
                ["Cut-off appliqué", f"{st.session_state.get('default_cutoff', 1.0)} g/t"],
            ]
            t = Table(data_table, colWidths=[200, 200])
            t.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ])
            )
            story.append(t)
            story.append(Spacer(1, 18))

            story.append(Paragraph("<b>Évaluation & Conclusions Techniques :</b>", styles["Heading2"]))
            story.append(Spacer(1, 8))
            
            for paragraph in comments.split("\n"):
                if paragraph.strip():
                    story.append(Paragraph(paragraph, styles["Normal"]))
                    story.append(Spacer(1, 6))

            doc.build(story)
            buffer.seek(0)

            st.download_button(
                label="📥 Télécharger le Rapport PDF",
                data=buffer,
                file_name="rapport_technique_projet.pdf",
                mime="application/pdf",
            )
        else:
            st.error("Le package `reportlab` n'est pas installé (`pip install reportlab`).")


# ==========================================
# MODULE 8: ADMINISTRATION
# ==========================================
elif module == "⚙️ Admin":
    st.header("⚙️ Espace d'Administration Système")

    tab_adm1, tab_adm2 = st.tabs([
        "⚙️ Config & Paramètres Application",
        "📜 Déconnexion",
    ])

    with tab_adm1:
        st.subheader("Réglages de l'Application")
        new_title = st.text_input("Titre personnalisé :", st.session_state["app_title"])
        new_cutoff = st.number_input(
            "Valeur Cut-off par défaut (g/t) :",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state["default_cutoff"]),
            step=0.1,
        )

        if st.button("💾 Enregistrer les Modifications"):
            st.session_state["app_title"] = new_title
            st.session_state["default_cutoff"] = new_cutoff
            st.success("Paramètres enregistrés !")

        st.markdown("---")
        st.subheader("État des Librairies Dépendantes")
        c_a1, c_a2 = st.columns(2)
        c_a1.metric("Folium (Cartographie)", "OK" if HAS_FOLIUM else "Manquant")
        c_a2.metric("ReportLab (PDF)", "OK" if HAS_REPORTLAB else "Manquant")

    with tab_adm2:
        st.subheader("Gestion de Session")
        if st.button("🚪 Déconnexion"):
            st.session_state["user_authenticated"] = False
            st.session_state["admin_authenticated"] = False
            st.rerun()
