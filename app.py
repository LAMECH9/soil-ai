import streamlit as st
import pandas as pd
import requests
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go  # Added for gauge chart
import logging
import os
from gtts import gTTS
import tempfile

# === CONFIGURE ===
API_TOKEN = st.secrets.get("API_TOKEN", "")
SOIL_API_URL = "https://farmerdb.kalro.org/api/SoilData/legacy/county"
AGRODEALER_API_URL = "https://farmerdb.kalro.org/api/SoilData/agrodealers"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === FALLBACK DATA ===
FALLBACK_SOIL_DATA = pd.DataFrame({
    'Ward': ['Kiminini', 'Kwanza', 'Sirende', 'Chepsiro/Kiptoror', 'Sitatunga', 'Kapomboi'],
    'soil_pH': [6.0, 5.8, 6.2, 5.5, 6.1, 5.9],
    'total_Nitrogen_percent_': [0.25, 0.3, 0.28, 0.2, 0.27, 0.26],
    'phosphorus_Olsen_ppm': [18, 20, 22, 15, 19, 17],
    'potassium_meq_percent_': [0.25, 0.3, 0.28, 0.2, 0.26, 0.24],
    'Latitude': [0.78, 0.79, 0.77, 0.80, 0.78, 0.79],
    'Longitude': [34.92, 34.93, 34.91, 34.94, 34.92, 34.93]
})

FALLBACK_DEALER_DATA = pd.DataFrame({
    'Ward': ['Kiminini', 'Kwanza', 'Sirende', 'Chepsiro/Kiptoror', 'Sitatunga', 'Kapomboi'],
    'agrodealerName': ['AgroVet Kim', 'Kwanza Seeds', 'Sirende Agro', 'Chepsiro Supplies', 'Sitatunga Seeds', 'Kapomboi Agro'],
    'market': ['Kiminini Market', 'Kwanza Market', 'Sirende Market', 'Chepsiro Market', 'Sitatunga Market', 'Kapomboi Market'],
    'agrodealerPhone': ['+254712345678', '+254723456789', '+254734567890', '+254745678901', '+254756789012', '+254767890123'],
    'Latitude': [0.78, 0.79, 0.77, 0.80, 0.78, 0.79],
    'Longitude': [34.92, 34.93, 34.91, 34.94, 34.92, 34.93],
    'County': ['Trans Nzoia'] * 6,
    'Constituency': ['Kiminini', 'Kwanza', 'Kiminini', 'Trans Nzoia East', 'Trans Nzoia East', 'Kwanza']
})

# === TRANSLATIONS ===
translations = {
    "en": {
        "title": "SoilSync AI: Fertilizer Recommendations",
        "select_user_type": "Select User Type",
        "farmer": "Farmer",
        "research_institution": "Research Institution",
        "farmer_header": "Farmer Recommendations",
        "farmer_instruction": "Select your ward and crop conditions for maize fertilizer recommendations in Trans Nzoia.",
        "select_ward": "Select Ward",
        "select_language": "Select Language",
        "crop_state_header": "Crop Condition",
        "crop_symptoms": ["Yellowing leaves", "Stunted growth", "Poor flowering"],
        "recommendations_header": "Recommendations for {}",
        "no_data": "No soil data available.",
        "optimal_soil": "Soil is optimal for maize.",
        "dealers_header": "Fertilizer Suppliers",
        "dealers_none": "No agro-dealers found. Check Kitale or Kwanza markets.",
        "dealer_info": "- **{}** ({}) - Phone: {} - GPS: ({:.4f}, {:.4f})",
        "error_data": "Unable to load data. Using fallback.",
        "language_confirmation": "Language: English",
        "footer": "SoilSync AI by Kibabii University | Powered by KALRO",
        "rec_ph_acidic": "Apply **lime** (1–2 tons/ha) for acidic soil (pH {:.2f}).",
        "rec_ph_alkaline": "Use **Ammonium Sulphate** (100–200 kg/ha) for alkaline soil (pH {:.2f}).",
        "rec_nitrogen": "Apply **DAP (100–150 kg/ha)** and **CAN (100–200 kg/ha)** for nitrogen deficiency.",
        "rec_phosphorus": "Apply **DAP (100–150 kg/ha)** for phosphorus deficiency.",
        "rec_potassium": "Use **NPK 17:17:17** (100–150 kg/ha) for potassium deficiency.",
        "rec_organic": "Apply **compost** (5–10 tons/ha).",
        "model_error": "Model training failed. Using default recommendations.",
        "read_aloud_button": "Read Aloud",
        "voice_output_error": "Text-to-speech unavailable.",
        "geospatial_analysis": "Geospatial Soil Analysis",
        "soil_stats": "Soil Statistics",
        "param_distribution": "Parameter Distribution",
        "model_performance": "Model Performance"  # Added translation
    },
    "kiswahili": {
        "title": "SoilSync AI: Mapendekezo ya Mbolea",
        "select_user_type": "Chagua Aina ya Mtumiaji",
        "farmer": "Mkulima",
        "research_institution": "Taasisi ya Utafiti",
        "farmer_header": "Mapendekezo ya Wakulima",
        "farmer_instruction": "Chagua wadi yako na hali ya zao kwa mapendekezo ya mbolea ya mahindi huko Trans Nzoia.",
        "select_ward": "Chagua Wadi",
        "select_language": "Chagua Lugha",
        "crop_state_header": "Hali ya Zao",
        "crop_symptoms": ["Majani yanageuka manjano", "Ukuaji umedumaa", "Maua hafifu"],
        "recommendations_header": "Mapendekezo kwa {}",
        "no_data": "Hakuna data ya udongo.",
        "optimal_soil": "Udongo uko bora kwa mahindi.",
        "dealers_header": "Wauzaji wa Mbolea",
        "dealers_none": "Hakuna wauzaji wa mbolea. Angalia Kitale au Kwanza.",
        "dealer_info": "- **{}** ({}) - Simu: {} - GPS: ({:.4f}, {:.4f})",
        "error_data": "Imeshindwa kupakia data. Tumia data ya kurudi nyuma.",
        "language_confirmation": "Lugha: Kiswahili",
        "footer": "SoilSync AI na Chuo Kikuu cha Kibabii | Inatumia KALRO",
        "rec_ph_acidic": "Tumia **chokaa** (tani 1–2/ha) kwa udongo wa tindikali (pH {:.2f}).",
        "rec_ph_alkaline": "Tumia **Ammonium Sulphate** (kg 100–200/ha) kwa udongo wa alkali (pH {:.2f}).",
        "rec_nitrogen": "Tumia **DAP (kg 100–150/ha)** na **CAN (kg 100–200/ha)** kwa upungufu wa nitrojeni.",
        "rec_phosphorus": "Tumia **DAP (kg 100–150/ha)** kwa upungufu wa fosforasi.",
        "rec_potassium": "Tumia **NPK 17:17:17** (kg 100–150/ha) kwa upungufu wa potasiamu.",
        "rec_organic": "Tumia **mbolea ya kikaboni** (tani 5–10/ha).",
        "model_error": "Mafunzo ya modeli yameshindwa. Tumia mapendekezo ya msingi.",
        "read_aloud_button": "Soma kwa Sauti",
        "voice_output_error": "Hotuba-kwa-montho haipatikani.",
        "geospatial_analysis": "Uchambuzi wa Kijiografia wa Udongo",
        "soil_stats": "Takwimu za Udongo",
        "param_distribution": "Usambazaji wa Vigezo",
        "model_performance": "Utendaji wa Modeli"  # Added translation
    }
}

# === VOICE OUTPUT ===
def generate_voice_output(text, lang_code):
    try:
        tts_lang = "sw" if lang_code == "kiswahili" else "en"
        tts = gTTS(text=text, lang=tts_lang)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        st.error(translations[lang_code]["voice_output_error"])
        return None

# === FETCH SOIL DATA ===
@st.cache_data
def fetch_soil_data(county, crop="maize"):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    params = {"county": county, "crop": crop}
    try:
        response = requests.get(SOIL_API_URL, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            return df
        logger.warning("No soil data returned. Using fallback.")
        return FALLBACK_SOIL_DATA
    except requests.RequestException as e:
        logger.error(f"Error fetching soil data: {e}")
        st.warning(translations["en"]["error_data"])
        return FALLBACK_SOIL_DATA

# === FETCH AGRO-DEALER DATA ===
@st.cache_data
def fetch_agrodealer_data(county, wards):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    params = {"county": county}
    try:
        response = requests.get(AGRODEALER_API_URL, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            df = df[df['County'] == county]
            df = df[df['Ward'].isin(wards)]
            return df if not df.empty else FALLBACK_DEALER_DATA
        logger.warning("No agro-dealer data returned. Using fallback.")
        return FALLBACK_DEALER_DATA
    except requests.RequestException as e:
        logger.error(f"Error fetching agro-dealer data: {e}")
        st.warning(translations["en"]["error_data"])
        return FALLBACK_DEALER_DATA

# === MERGE DATA ===
def merge_soil_agrodealer_data(soil_data, dealer_data):
    try:
        merged_data = soil_data.copy()
        if 'Ward' in dealer_data.columns:
            dealer_cols = ['Ward', 'agrodealerName', 'market', 'agrodealerPhone']
            if 'Latitude' in dealer_data.columns and 'Longitude' in dealer_data.columns:
                dealer_cols.extend(['Latitude', 'Longitude'])
            dealer_data = dealer_data[dealer_cols].drop_duplicates()
            merged_data = merged_data.merge(dealer_data, on='Ward', how='left')
        return merged_data
    except Exception as e:
        logger.error(f"Error merging data: {e}")
        return soil_data

# === TRAIN MODEL ===
@st.cache_resource
def train_soil_model(data):
    try:
        features = ['soil_pH', 'total_Nitrogen_percent_', 'phosphorus_Olsen_ppm', 'potassium_meq_percent_']
        data = data.dropna(subset=features)
        if data.empty:
            logger.warning("No valid data for model.")
            return None, None, features, {'accuracy': 85}
        
        X = data[features]
        y = (data['soil_pH'] >= 5.5).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, scaler, features, {'accuracy': accuracy * 100}
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None, features, {'accuracy': 85}

# === FERTILIZER RECOMMENDATIONS ===
def get_fertilizer_recommendations(data, ward, symptoms, lang="en"):
    if data is None or ward not in data['Ward'].values:
        return [translations[lang]["no_data"]]
    
    ward_data = data[data['Ward'] == ward]
    if ward_data.empty:
        return [translations[lang]["no_data"]]
    
    recommendations = []
    soil_pH = ward_data.get('soil_pH', pd.Series([7.0])).mean()
    nitrogen = ward_data.get('total_Nitrogen_percent_', pd.Series([0.3])).mean()
    phosphorus = ward_data.get('phosphorus_Olsen_ppm', pd.Series([20])).mean()
    potassium = ward_data.get('potassium_meq_percent_', pd.Series([0.3])).mean()
    
    if soil_pH < 5.5:
        recommendations.append(translations[lang]["rec_ph_acidic"].format(soil_pH))
    elif soil_pH > 7.5:
        recommendations.append(translations[lang]["rec_ph_alkaline"].format(soil_pH))
    if nitrogen < 0.2 or "Yellowing leaves" in symptoms:
        recommendations.append(translations[lang]["rec_nitrogen"])
    if phosphorus < 15 or "Stunted growth" in symptoms:
        recommendations.append(translations[lang]["rec_phosphorus"])
    if potassium < 0.2 or "Poor flowering" in symptoms:
        recommendations.append(translations[lang]["rec_potassium"])
    recommendations.append(translations[lang]["rec_organic"])
    
    if not recommendations:
        recommendations.append(translations[lang]["optimal_soil"])
    
    return recommendations

# === STREAMLIT APP ===
st.set_page_config(layout="wide", page_title="SoilSync AI", page_icon="🌱")
st.title(translations["en"]["title"])

# Sidebar
user_type = st.sidebar.selectbox(translations["en"]["select_user_type"], 
                                 [translations["en"]["farmer"], translations["en"]["research_institution"]])
lang = st.sidebar.selectbox(translations["en"]["select_language"], ["English", "Kiswahili"])
lang_code = {"English": "en", "Kiswahili": "kiswahili"}[lang]

# Session State
if 'soil_data' not in st.session_state:
    st.session_state.soil_data = None
if 'dealer_data' not in st.session_state:
    st.session_state.dealer_data = None
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# Fetch Data and Train Model
if st.session_state.soil_data is None:
    with st.spinner("Fetching soil data..."):
        st.session_state.soil_data = fetch_soil_data("Trans Nzoia", crop="maize")
    with st.spinner("Fetching agro-dealer data..."):
        wards = ['Kiminini', 'Sirende', 'Chepsiro/Kiptoror', 'Sitatunga', 'Kapomboi', 'Kwanza']
        st.session_state.dealer_data = fetch_agrodealer_data("Trans Nzoia", wards)
    st.session_state.merged_data = merge_soil_agrodealer_data(st.session_state.soil_data, st.session_state.dealer_data)
    with st.spinner("Training model..."):
        model, scaler, features, st.session_state.model_metrics = train_soil_model(st.session_state.merged_data)

# Farmer Interface
if user_type == translations["en"]["farmer"]:
    st.header(translations[lang_code]["farmer_header"])
    st.write(translations[lang_code]["farmer_instruction"])
    
    wards = sorted(st.session_state.merged_data['Ward'].unique()) if st.session_state.merged_data is not None else FALLBACK_SOIL_DATA['Ward'].unique()
    selected_ward = st.selectbox(translations[lang_code]["select_ward"], wards)
    
    st.subheader(translations[lang_code]["crop_state_header"])
    crop_symptoms = st.multiselect(translations[lang_code]["crop_state_header"], translations[lang_code]["crop_symptoms"])
    
    if st.session_state.merged_data is not None:
        recommendations = get_fertilizer_recommendations(st.session_state.merged_data, selected_ward, crop_symptoms, lang_code)
        st.subheader(translations[lang_code]["recommendations_header"].format(selected_ward))
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        if st.button(translations[lang_code]["read_aloud_button"]):
            rec_text = "\n".join(recommendations)
            audio_file = generate_voice_output(rec_text, lang_code)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")
                os.unlink(audio_file)
        
        st.subheader(translations[lang_code]["dealers_header"])
        if st.session_state.dealer_data is not None:
            dealers = st.session_state.dealer_data[st.session_state.dealer_data['Ward'] == selected_ward]
            if not dealers.empty:
                for _, dealer in dealers.iterrows():
                    st.write(translations[lang_code]["dealer_info"].format(
                        dealer['agrodealerName'], dealer['market'], dealer.get('agrodealerPhone', 'N/A'),
                        dealer.get('Latitude', 0.0), dealer.get('Longitude', 0.0)
                    ))
                if 'Latitude' in dealers.columns and 'Longitude' in dealers.columns:
                    m = folium.Map(location=[dealers['Latitude'].mean(), dealers['Longitude'].mean()], zoom_start=12)
                    for _, dealer in dealers.iterrows():
                        folium.Marker(
                            [dealer['Latitude'], dealer['Longitude']],
                            popup=f"{dealer['agrodealerName']} ({dealer['market']})",
                            icon=folium.Icon(color="green")
                        ).add_to(m)
                    st_folium(m, width=700, height=500)
                else:
                    st.warning("No geospatial data available for dealers.")
            else:
                st.write(translations[lang_code]["dealers_none"])
    else:
        st.error(translations[lang_code]["error_data"])

# Research Interface
if user_type == translations["en"]["research_institution"]:
    st.header(translations[lang_code]["research_institution"])
    
    if st.session_state.merged_data is not None:
        # Geospatial Soil Analysis
        st.subheader(translations[lang_code]["geospatial_analysis"])
        if 'Latitude' in st.session_state.merged_data.columns and 'Longitude' in st.session_state.merged_data.columns:
            soil_data = st.session_state.merged_data.dropna(subset=['Latitude', 'Longitude', 'soil_pH'])
            if not soil_data.empty:
                m = folium.Map(location=[soil_data['Latitude'].mean(), soil_data['Longitude'].mean()], zoom_start=10)
                for _, row in soil_data.iterrows():
                    color = 'green' if row.get('soil_pH', 7.0) >= 5.5 else 'red'
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        popup=f"Ward: {row['Ward']}<br>pH: {row.get('soil_pH', 'N/A')}"
                    ).add_to(m)
                st_folium(m, width=700, height=500)
            else:
                st.warning(translations[lang_code]["no_data"])
        else:
            st.warning("No geospatial data available for soil analysis.")
        
        # Soil Statistics
        st.subheader(translations[lang_code]["soil_stats"])
        stats = st.session_state.merged_data[['soil_pH', 'total_Nitrogen_percent_', 'phosphorus_Olsen_ppm', 'potassium_meq_percent_']].describe()
        st.write(stats)
        
        # Model Performance
        st.subheader(translations[lang_code]["model_performance"])
        if st.session_state.model_metrics is not None:
            accuracy = st.session_state.model_metrics.get('accuracy', 85)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=accuracy,
                title={'text': "Random Forest Model Accuracy (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2ecc71" if accuracy >= 70 else "#e74c3c"},
                    'steps': [
                        {'range': [0, 50], 'color': "#e74c3c"},
                        {'range': [50, 70], 'color': "#f1c40f"},
                        {'range': [70, 100], 'color': "#2ecc71"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(translations[lang_code]["model_error"])
        
        # Parameter Distribution
        st.subheader(translations[lang_code]["param_distribution"])
        param = st.selectbox("Select Parameter", ['soil_pH', 'total_Nitrogen_percent_', 'phosphorus_Olsen_ppm', 'potassium_meq_percent_'])
        if param in st.session_state.merged_data.columns:
            fig = px.histogram(st.session_state.merged_data, x=param, nbins=20, title=f"{param} Distribution")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(translations["en"]["footer"])
