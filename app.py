import streamlit as st
import pandas as pd
import requests
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import json
from datetime import datetime
import logging
import plotly.express as px
from gtts import gTTS
import os

# === CONFIGURE ===
API_TOKEN = "KLRDg3ElBVveVghcN61aScAJevKMgofJF7CWcsVwG2mYt0mUQF63DdB0n6OHqOo9WYCilH7bjJ6s9sIc4zT9zzeCyPXhvytRL4wMAtbV5fRxnAmLFtEI9KXO5tvnu0Pm3rwhAfx5tXGiQOKEm98U2lGTZOIVav2hRtGwsU8SrzUPpZA6CNSNCGkCNp3sndYsrAqeme9xsqFGNEla2PBgjZ0ertc6j8nzCVzUQ8gX2T9hFnR8SoKRA7eyRMHRMDrn"
SOIL_API_URL = "https://farmerdb.kalro.org/api/SoilData/legacy/county"
AGRODEALER_API_URL = "https://farmerdb.kalro.org/api/SoilData/agrodealers"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === TRANSLATIONS DICTIONARY ===
translations = {
    "en": {
        "title": "SoilSync AI: Precision Fertilizer Recommendations for Maize",
        "select_user_type": "Select User Type",
        "farmer": "Farmer",
        "research_institution": "Research Institution",
        "farmer_header": "Farmer-Friendly Recommendations",
        "farmer_instruction": "Select your ward and describe your crop's condition to get tailored fertilizer recommendations for maize farming in Trans Nzoia.",
        "select_ward": "Select Your Ward",
        "select_language": "Select Language",
        "crop_state_header": "Describe Your Crop's Condition",
        "crop_symptoms": ["Yellowing leaves", "Stunted growth", "Poor flowering", "Wilting", "Leaf spots"],
        "recommendations_header": "Recommendations for {}",
        "no_data": "No soil data available for recommendations.",
        "optimal_soil": "Soil parameters are within optimal ranges for maize.",
        "dealers_header": "Where to Buy Fertilizers",
        "dealers_none": "No agro-dealers found for this ward. Check county-level suppliers in Kitale or Kwanza markets.",
        "dealer_info": "- **{}** ({}) - Phone: {} - GPS: ({:.4f}, {:.4f})",
        "error_data": "Unable to load soil data. Please try again later.",
        "language_confirmation": "Language set to English.",
        "footer": "SoilSync AI by Kibabii University | Powered by KALRO Data | Contact: peter.barasa@kibu.ac.ke",
        "rec_ph_acidic": "Apply **agricultural lime** (1–2 tons/ha) to correct acidic soil (pH {:.2f}).",
        "rec_ph_alkaline": "Use **Ammonium Sulphate** (100–200 kg/ha) to lower alkaline soil (pH {:.2f}).",
        "rec_nitrogen": "Apply **DAP (100–150 kg/ha)** at planting and **CAN (100–200 kg/ha)** or **Urea (50–100 kg/ha)** for top-dressing to address nitrogen deficiency.",
        "rec_phosphorus": "Apply **DAP (100–150 kg/ha)** or **TSP (100–150 kg/ha)** at planting for phosphorus deficiency.",
        "rec_potassium": "Use **NPK 17:17:17 or 23:23:0** (100–150 kg/ha) at planting for potassium deficiency.",
        "rec_zinc": "Apply **Mavuno Maize Fertilizer** or **YaraMila Cereals** for zinc deficiency, or use zinc sulfate foliar spray (5–10 kg/ha).",
        "rec_boron": "Apply **borax** (1–2 kg/ha) for boron deficiency.",
        "rec_organic": "Apply **compost/manure (5–10 tons/ha)** or **Mazao Organic** to boost organic matter.",
        "rec_salinity": "Implement leaching with irrigation and use **Ammonium Sulphate** to manage high salinity.",
        "model_error": "Model training failed. Using threshold-based recommendations.",
        "model_accuracy": "Random Forest Model Accuracy: {:.2f}%",
        "carbon_sequestration": "Estimated Carbon Sequestration: {:.2f} tons/ha/year",
        "yield_impact": "Estimated Yield Increase: {:.2f} tons/ha ({:.0f}%)",
        "fertilizer_savings": "Fertilizer Waste Reduction: {:.1f}%",
        "prediction_header": "Soil Fertility Predictions Across Wards",
        "param_stats": "Soil Parameter Statistics",
        "feature_importance": "Feature Importance for Soil Fertility Prediction",
        "agrodealer_map": "Agro-Dealer Locations",
        "soil_parameter_dist": "Soil Parameter Distribution",
        "read_recommendations": "Read Recommendations Aloud"
    },
    "sw": {
        "title": "SoilSync AI: Mapendekezo ya Mbolea ya Usahihi kwa Mahindi",
        "select_user_type": "Chagua Aina ya Mtumiaji",
        "farmer": "Mkulima",
        "research_institution": "Taasisi ya Utafiti",
        "farmer_header": "Mapendekezo Yanayofaa Mkulima",
        "farmer_instruction": "Chagua wadi yako na elezea hali ya mazao yako ili kupata mapendekezo ya mbolea yanayofaa kwa kilimo cha mahindi huko Trans Nzoia.",
        "select_ward": "Chagua Wadi Yako",
        "select_language": "Chagua Lugha",
        "crop_state_header": "Elezea Hali ya Mazao Yako",
        "crop_symptoms": ["Majani yanageuka manjano", "Ukuaji umedumaa", "Maua duni", "Kunyauka", "Madoa kwenye majani"],
        "recommendations_header": "Mapendekezo kwa {}",
        "no_data": "Hakuna data ya udongo inayopatikana kwa mapendekezo.",
        "optimal_soil": "Vigezo vya udongo viko ndani ya viwango bora kwa mahindi.",
        "dealers_header": "Wapi pa Kununua Mbolea",
        "dealers_none": "Hakuna wauzaji wa mbolea waliopatikana kwa wadi hii. Angalia wauzaji wa ngazi ya kaunti katika soko za Kitale au Kwanza.",
        "dealer_info": "- **{}** ({}) - Simu: {} - GPS: ({:.4f}, {:.4f})",
        "error_data": "Imeshindwa kupakia data ya udongo. Tafadhali jaribu tena baadaye.",
        "language_confirmation": "Lugha imewekwa kwa Kiswahili.",
        "footer": "SoilSync AI na Chuo Kikuu cha Kibabii | Inatumia Data ya KALRO | Wasiliana: peter.barasa@kibu.ac.ke",
        "rec_ph_acidic": "Tumia **chokaa cha kilimo** (tani 1–2 kwa hekta) kurekebisha udongo wenye tindikali (pH {:.2f}).",
        "rec_ph_alkaline": "Tumia **Ammonium Sulphate** (kg 100–200 kwa hekta) kupunguza udongo wa alkali (pH {:.2f}).",
        "rec_nitrogen": "Tumia **DAP (kg 100–150 kwa hekta)** wakati wa kupanda na **CAN (kg 100–200 kwa hekta)** au **Urea (kg 50–100 kwa hekta)** kwa kurutubisha juu ili kushughulikia upungufu wa nitrojeni.",
        "rec_phosphorus": "Tumia **DAP (kg 100–150 kwa hekta)** au **TSP (kg 100–150 kg kwa hekta)** wakati wa kupanda kwa upungufu wa fosforasi.",
        "rec_potassium": "Tumia **NPK 17:17:17 au 23:23:0** (kg 100–150 kwa hekta) wakati wa kupanda kwa upungufu wa potasiamu.",
        "rec_zinc": "Tumia **Mbolea ya Mavuno Maize** au **YaraMila Cereals** kwa upungufu wa zinki, au tumia dawa ya zinki ya sulfate (kg 5–10 kwa hekta).",
        "rec_boron": "Tumia **borax** (kg 1–2 kwa hekta) kwa upungufu wa boron.",
        "rec_organic": "Tumia **mbolea ya kikaboni/samadi (tani 5–10 kwa hekta)** au **Mazao Organic** kuongeza vitu vya kikaboni.",
        "rec_salinity": "Tekeleza uchukuzi wa maji na umwagiliaji na tumia **Ammonium Sulphate** kushughulikia chumvi nyingi.",
        "model_error": "Ufundishaji wa modeli umeshindwa. Tumia mapendekezo ya msingi wa kizingiti.",
        "model_accuracy": "Usahihi wa Modeli ya Msitu wa Random: {:.2f}%",
        "carbon_sequestration": "Makadirio ya Uchukuzi wa Kaboni: {:.2f} tani/ha/mwaka",
        "yield_impact": "Makadirio ya Ongezeko la Mavuno: {:.2f} tani/ha ({:.0f}%)",
        "fertilizer_savings": "Punguzo la Upotevu wa Mbolea: {:.1f}%",
        "prediction_header": "Mapendekezo ya Uzazi wa Udongo Katika Wadi",
        "param_stats": "Takwimu za Vigezo vya Udongo",
        "feature_importance": "Umuhimu wa Vipengele kwa Utambuzi wa Uzazi wa Udongo",
        "agrodealer_map": "Maeneo ya Wauzaji wa Mbolea",
        "soil_parameter_dist": "Usambazaji wa Vigezo vya Udongo",
        "read_recommendations": "Soma Mapendekezo kwa Sauti"
    }
}

# === FUNCTION TO FETCH SOIL DATA ===
@st.cache_data
def fetch_soil_data(county_name, crop="maize"):
    url = f"{SOIL_API_URL}/{county_name}"
    headers = {"Authorization": f"Token {API_TOKEN}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        relevant_columns = [
            "county", "constituency", "ward", "latitude", "longitude", "soil_pH",
            "total_Nitrogen_percent_", "total_Org_Carbon_percent_", "phosphorus_Olsen_ppm",
            "potassium_meq_percent_", "calcium_meq_percent_", "magnesium_meq_percent_",
            "zinc_ppm", "boron_ppm", "electr_Conductivity_mS_per_cm", "crop"
        ]
        available_columns = [col for col in relevant_columns if col in df.columns]
        df_filtered = df[available_columns].copy()
        if "crop" in df_filtered.columns:
            df_filtered["crop"] = df_filtered["crop"].astype(str)
            maize_mask = df_filtered["crop"].str.lower().str.contains(crop.lower(), na=False)
            df_filtered = df_filtered[maize_mask]
        core_params = [
            "soil_pH", "total_Nitrogen_percent_", "phosphorus_Olsen_ppm", "potassium_meq_percent_"
        ]
        core_params = [col for col in core_params if col in df_filtered.columns]
        df_filtered = df_filtered.dropna(subset=core_params, how='any')
        numeric_cols = [col for col in core_params + ["total_Org_Carbon_percent_", "zinc_ppm", "boron_ppm", "electr_Conductivity_mS_per_cm", "latitude", "longitude"] if col in df_filtered.columns]
        for col in numeric_cols:
            if col == "total_Org_Carbon_percent_":
                df_filtered[col] = df_filtered[col].apply(lambda x: x.split('.')[0] + '.' + x.split('.')[1][:2] if isinstance(x, str) and x.count('.') > 1 else x)
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        df_filtered = df_filtered.rename(columns={
            "county": "County", "constituency": "Constituency", "ward": "Ward",
            "latitude": "Latitude", "longitude": "Longitude"
        })
        logger.info(f"Fetched {len(df_filtered)} soil records for {county_name}")
        return df_filtered
    except Exception as e:
        logger.error(f"Soil data fetch error: {e}")
        st.error(translations["en"]["error_data"])
        return None

# === FUNCTION TO FETCH AGRO-DEALER DATA ===
@st.cache_data
def fetch_agrodealer_data(county_name, constituencies=None, wards=None):
    headers = {"Authorization": f"Token {API_TOKEN}", "Content-Type": "application/json"}
    all_dealers = []
    try:
        if constituencies and wards:
            for constituency, ward in zip(constituencies, wards):
                url = f"{AGRODEALER_API_URL}/{county_name}/{constituency}/{ward}"
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, dict) and "dealers" in data:
                        all_dealers.extend(data["dealers"])
                except:
                    continue
        if not all_dealers:
            url = f"{AGRODEALER_API_URL}/{county_name}"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "dealers" in data:
                all_dealers.extend(data["dealers"])
        if all_dealers:
            df_dealers = pd.DataFrame(all_dealers)
            dealer_columns = ["county", "subcounty", "ward", "agrodealerName", "market", "gpsLatitude", "gpsLongitude", "agrodealerPhone"]
            df_dealers = df_dealers[[col for col in dealer_columns if col in df_dealers.columns]]
            df_dealers = df_dealers.rename(columns={
                "county": "County", "subcounty": "Constituency", "ward": "Ward",
                "gpsLatitude": "Latitude", "gpsLongitude": "Longitude"
            })
            df_dealers['Latitude'] = pd.to_numeric(df_dealers['Latitude'], errors='coerce')
            df_dealers['Longitude'] = pd.to_numeric(df_dealers['Longitude'], errors='coerce')
            logger.info(f"Fetched {len(df_dealers)} agro-dealer records for {county_name}")
            return df_dealers
        return None
    except Exception as e:
        logger.error(f"Agro-dealer fetch error: {e}")
        st.error(translations["en"]["error_data"])
        return None

# === FUNCTION TO MERGE SOIL AND AGRO-DEALER DATA ===
def merge_soil_agrodealer_data(soil_df, dealer_df):
    if soil_df is None:
        logger.error("Cannot merge: Soil dataset is empty")
        return None
    if dealer_df is None:
        logger.warning("No agro-dealer data available; proceeding with soil data")
        return soil_df
    try:
        merged_df = pd.merge(
            soil_df, dealer_df, on=["County", "Constituency", "Ward"],
            how="left", suffixes=("_soil", "_dealer")
        )
        merged_df['Latitude'] = merged_df['Latitude_soil'].fillna(merged_df['Latitude_dealer'])
        merged_df['Longitude'] = merged_df['Longitude_soil'].fillna(merged_df['Longitude_dealer'])
        merged_df = merged_df.drop(columns=['Latitude_soil', 'Longitude_soil', 'Latitude_dealer', 'Longitude_dealer'], errors='ignore')
        logger.info(f"Merged dataset contains {len(merged_df)} records")
        return merged_df
    except Exception as e:
        logger.error(f"Merge error: {e}")
        return soil_df

# === TRAIN RANDOM FOREST MODEL ===
def train_soil_model(soil_data):
    if soil_data is None or soil_data.empty:
        logger.warning("Soil data is None or empty")
        return None, None, [], None
    features = ["soil_pH", "total_Nitrogen_percent_", "phosphorus_Olsen_ppm", "potassium_meq_percent_", "zinc_ppm", "boron_ppm"]
    features = [f for f in features if f in soil_data.columns]
    if not features:
        logger.warning("No valid features in soil data")
        return None, None, features, None
    X = soil_data[features].copy()
    for col in features:
        X[col] = X[col].fillna(X[col].mean())
    if X.empty:
        logger.warning("No data after processing")
        return None, None, features, None
    y = []
    for _, row in X.iterrows():
        score = (
            (row.get("soil_pH", 7.0) >= 5.5 and row.get("soil_pH", 7.0) <= 7.0) * 1 +
            (row.get("total_Nitrogen_percent_", 0.3) >= 0.2) * 1 +
            (row.get("phosphorus_Olsen_ppm", 20) >= 15) * 1 +
            (row.get("potassium_meq_percent_", 0.3) >= 0.2) * 1
        )
        if score >= 3:
            y.append("high")
        elif score >= 1:
            y.append("medium")
        else:
            y.append("low")
    if len(set(y)) < 2:
        logger.warning("Single-class labels detected")
        return None, None, features, None
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        accuracy = scores.mean() * 100
        logger.info(f"Model trained successfully with accuracy: {accuracy:.2f}%")
        return model, scaler, features, accuracy
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return None, None, features, None

# === PREDICT SOIL FERTILITY ===
def predict_soil_fertility(model, scaler, features, input_data):
    if model is None or scaler is None or not features:
        return None, None
    try:
        input_df = pd.DataFrame([input_data], columns=features)
        for col in features:
            if col in input_df.columns and pd.isna(input_df[col].iloc[0]):
                input_df[col] = input_df[col].fillna(input_df[col].mean())
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        importance = model.feature_importances_
        explanation = {f: i for f, i in zip(features, importance)}
        return prediction, explanation
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, None

# === PREDICT FOR ALL WARDS ===
def predict_all_wards(soil_data, model, scaler, features):
    if soil_data is None or model is None or scaler is None or not features:
        logger.warning("Cannot predict: missing data or model")
        return pd.DataFrame(columns=["Ward", "Fertility"])
    predictions = []
    for ward in soil_data['Ward'].unique():
        ward_data = soil_data[soil_data['Ward'] == ward]
        if ward_data.empty:
            continue
        avg_data = ward_data[features].mean().to_dict()
        for col in features:
            if col in avg_data and pd.isna(avg_data[col]):
                avg_data[col] = ward_data[col].mean() if not ward_data[col].isna().all() else 0
        prediction, _ = predict_soil_fertility(model, scaler, features, avg_data)
        if prediction:
            predictions.append({"Ward": ward, "Fertility": prediction})
    return pd.DataFrame(predictions) if predictions else pd.DataFrame(columns=["Ward", "Fertility"])

# === ESTIMATE CARBON SEQUESTRATION ===
def estimate_carbon_sequestration(soil_data, ward):
    ward_data = soil_data[soil_data["Ward"] == ward]
    if ward_data.empty or 'total_Org_Carbon_percent_' not in ward_data.columns:
        return 0.0
    organic_carbon = ward_data["total_Org_Carbon_percent_"].mean()
    if pd.isna(organic_carbon):
        return 0.0
    sequestration_rate = organic_carbon * 0.58
    return sequestration_rate

# === ESTIMATE YIELD IMPACT ===
def estimate_yield_impact(recommendations, ward_data):
    yield_increase = 0.15 if any("DAP" in rec for rec in recommendations) else 0.1
    baseline_yield = ward_data["yield"].mean() if "yield" in ward_data.columns else 2.5
    new_yield = baseline_yield * (1 + yield_increase)
    return new_yield - baseline_yield, yield_increase * 100

# === ESTIMATE FERTILIZER SAVINGS ===
def estimate_fertilizer_savings(recommendations):
    conventional_rate = 200
    recommended_rate = sum([100 if "DAP" in rec else 0 for rec in recommendations])
    savings = (conventional_rate - recommended_rate) / conventional_rate * 100 if recommended_rate > 0 else 0
    return savings

# === TEXT-TO-SPEECH FUNCTION ===
def read_recommendations(text, lang="en"):
    try:
        tts = gTTS(text=text, lang="sw" if lang == "sw" else "en")
        audio_file = "recommendations.mp3"
        tts.save(audio_file)
        with open(audio_file, "rb") as f:
            st.audio(f, format="audio/mp3")
        os.remove(audio_file)
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        st.warning("Text-to-speech is not available. Please ensure you have an active internet connection.")

# === FERTILIZER RECOMMENDATION FUNCTION FOR FARMERS ===
def get_fertilizer_recommendations_farmer(soil_data, ward, crop_symptoms, lang="en"):
    recommendations = []
    if soil_data is None or soil_data.empty:
        return [translations[lang]["no_data"]]
    ward_data = soil_data[soil_data['Ward'] == ward]
    if ward_data.empty:
        ward_data = soil_data
    avg_data = ward_data.mean(numeric_only=True)
    
    symptom_deficiencies = {
        "Yellowing leaves": ["nitrogen", "zinc"],
        "Stunted growth": ["nitrogen", "phosphorus", "potassium"],
        "Poor flowering": ["phosphorus", "potassium"],
        "Wilting": ["potassium", "organic"],
        "Leaf spots": ["zinc", "boron"],
        "Majani yanageuka manjano": ["nitrogen", "zinc"],
        "Ukuaji umedumaa": ["nitrogen", "phosphorus", "potassium"],
        "Maua duni": ["phosphorus", "potassium"],
        "Kunyauka": ["potassium", "organic"],
        "Madoa kwenye majani": ["zinc", "boron"]
    }
    
    if 'soil_pH' in avg_data and avg_data['soil_pH'] < 5.5:
        recommendations.append(translations[lang]["rec_ph_acidic"].format(avg_data['soil_pH']))
    elif 'soil_pH' in avg_data and avg_data['soil_pH'] > 7.0:
        recommendations.append(translations[lang]["rec_ph_alkaline"].format(avg_data['soil_pH']))
    
    deficiencies = set()
    for symptom in crop_symptoms:
        deficiencies.update(symptom_deficiencies.get(symptom, []))
    
    if "nitrogen" in deficiencies or ('total_Nitrogen_percent_' in avg_data and avg_data['total_Nitrogen_percent_'] < 0.2):
        recommendations.append(translations[lang]["rec_nitrogen"])
    if "phosphorus" in deficiencies or ('phosphorus_Olsen_ppm' in avg_data and avg_data['phosphorus_Olsen_ppm'] < 15):
        recommendations.append(translations[lang]["rec_phosphorus"])
    if "potassium" in deficiencies or ('potassium_meq_percent_' in avg_data and avg_data['potassium_meq_percent_'] < 0.2):
        recommendations.append(translations[lang]["rec_potassium"])
    if "zinc" in deficiencies or ('zinc_ppm' in avg_data and avg_data['zinc_ppm'] < 1):
        recommendations.append(translations[lang]["rec_zinc"])
    if "boron" in deficiencies or ('boron_ppm' in avg_data and avg_data['boron_ppm'] < 0.5):
        recommendations.append(translations[lang]["rec_boron"])
    if "organic" in deficiencies or ('total_Org_Carbon_percent_' in avg_data and avg_data['total_Org_Carbon_percent_'] < 1):
        recommendations.append(translations[lang]["rec_organic"])
    if 'electr_Conductivity_mS_per_cm' in avg_data and avg_data['electr_Conductivity_mS_per_cm'] > 1:
        recommendations.append(translations[lang]["rec_salinity"])
    
    return recommendations if recommendations else [translations[lang]["optimal_soil"]]

# === FERTILIZER RECOMMENDATION FUNCTION FOR RESEARCH INSTITUTIONS ===
def get_fertilizer_recommendations_research(input_data, model, scaler, features, lang="en"):
    recommendations = []
    prediction, explanation = predict_soil_fertility(model, scaler, features, input_data)
    
    if prediction is None:
        recommendations.append(translations[lang]["model_error"])
    
    if input_data.get("soil_pH", 7.0) < 5.5:
        recommendations.append(translations[lang]["rec_ph_acidic"].format(input_data["soil_pH"]))
    elif input_data.get("soil_pH", 7.0) > 7.0:
        recommendations.append(translations[lang]["rec_ph_alkaline"].format(input_data["soil_pH"]))
    if input_data.get("total_Nitrogen_percent_", 0.3) < 0.2:
        recommendations.append(translations[lang]["rec_nitrogen"])
    if input_data.get("phosphorus_Olsen_ppm", 20) < 15:
        recommendations.append(translations[lang]["rec_phosphorus"])
    if input_data.get("potassium_meq_percent_", 0.3) < 0.2:
        recommendations.append(translations[lang]["rec_potassium"])
    if input_data.get("zinc_ppm", 2) < 1:
        recommendations.append(translations[lang]["rec_zinc"])
    if input_data.get("boron_ppm", 1) < 0.5:
        recommendations.append(translations[lang]["rec_boron"])
    
    advice = "No model prediction available." if prediction is None else f"Soil fertility predicted as {prediction}. "
    if prediction == "low":
        advice += "Low soil fertility detected due to deficiencies in: "
        advice += ", ".join([f"{k} ({v:.2%})" for k, v in (explanation or {}).items() if v > 0.1])
        advice += ". Recommend targeted fertilizer applications and soil management improvements."
    elif prediction == "medium":
        advice += "Moderate soil fertility. Address specific deficiencies to optimize maize yields."
    elif prediction == "high":
        advice += "High soil fertility. Maintain nutrient balance with minimal fertilizer adjustments."
    
    return recommendations if recommendations else [translations[lang]["optimal_soil"]], advice, explanation

# === STREAMLIT APP ===
st.set_page_config(layout="wide", page_title="SoilSync AI", page_icon="🌱")
st.title(translations["en"]["title"])

# Sidebar for User Type Selection
user_type = st.sidebar.selectbox(translations["en"]["select_user_type"], 
                                [translations["en"]["farmer"], translations["en"]["research_institution"]], 
                                key="user_type")

# Initialize Session State
if 'soil_data' not in st.session_state:
    st.session_state.soil_data = None
if 'dealer_data' not in st.session_state:
    st.session_state.dealer_data = None
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None

# Fetch Data Once
if st.session_state.soil_data is None:
    with st.spinner("Fetching soil data for Trans Nzoia..."):
        st.session_state.soil_data = fetch_soil_data("Trans Nzoia", crop="maize")
    with st.spinner("Fetching agro-dealer data..."):
        trans_nzoia_units = [
            {"constituency": "Kiminini", "ward": "Kiminini"},
            {"constituency": "Kiminini", "ward": "Sirende"},
            {"constituency": "Trans Nzoia East", "ward": "Chepsiro/Kiptoror"},
            {"constituency": "Trans Nzoia East", "ward": "Sitatunga"},
            {"constituency": "Kwanza", "ward": "Kapomboi"},
            {"constituency": "Kwanza", "ward": "Kwanza"}
        ]
        constituencies = [unit["constituency"] for unit in trans_nzoia_units]
        wards = [unit["ward"] for unit in trans_nzoia_units]
        st.session_state.dealer_data = fetch_agrodealer_data("Trans Nzoia", constituencies, wards)
    if st.session_state.soil_data is not None:
        st.session_state.merged_data = merge_soil_agrodealer_data(st.session_state.soil_data, st.session_state.dealer_data)
        st.session_state.model, st.session_state.scaler, st.session_state.features, st.session_state.model_accuracy = train_soil_model(st.session_state.merged_data)

# Farmer Interface
if user_type == translations["en"]["farmer"]:
    lang = st.sidebar.selectbox(translations["en"]["select_language"], ["English", "Kiswahili"], key="language")
    lang_code = {"English": "en", "Kiswahili": "sw"}[lang]
    st.sidebar.write(translations[lang_code]["language_confirmation"])
    
    st.header(translations[lang_code]["farmer_header"])
    st.write(translations[lang_code]["farmer_instruction"])
    
    required_wards = ["Kiminini", "Sirende", "Chepsiro/Kiptoror", "Sitatunga", "Kapomboi", "Kwanza"]
    data_wards = []
    if st.session_state.merged_data is not None:
        data_wards += st.session_state.merged_data['Ward'].dropna().str.lower().unique().tolist()
    if st.session_state.dealer_data is not None:
        data_wards += st.session_state.dealer_data['Ward'].dropna().str.lower().unique().tolist()
    wards = sorted(list(set(required_wards + [w.title() for w in data_wards])))
    selected_ward = st.selectbox(translations[lang_code]["select_ward"], wards)
    
    st.subheader(translations[lang_code]["crop_state_header"])
    crop_symptoms = st.multiselect(
        "Select observed crop symptoms",
        translations[lang_code]["crop_symptoms"],
        help="Choose all that apply to your maize crop."
    )
    
    if st.session_state.merged_data is not None:
        recommendations = get_fertilizer_recommendations_farmer(
            st.session_state.merged_data, selected_ward, crop_symptoms, lang=lang_code
        )
        st.subheader(translations[lang_code]["recommendations_header"].format(selected_ward))
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Text-to-Speech Button
        if st.button(translations[lang_code]["read_recommendations"]):
            recommendation_text = ". ".join(recommendations)
            read_recommendations(recommendation_text, lang=lang_code)
        
        sequestration_rate = estimate_carbon_sequestration(st.session_state.merged_data, selected_ward)
        st.write(translations[lang_code]["carbon_sequestration"].format(sequestration_rate))
        
        yield_increase, yield_pct = estimate_yield_impact(recommendations, st.session_state.merged_data[st.session_state.merged_data["Ward"] == selected_ward])
        st.write(translations[lang_code]["yield_impact"].format(yield_increase, yield_pct))
        
        savings = estimate_fertilizer_savings(recommendations)
        st.write(translations[lang_code]["fertilizer_savings"].format(savings))
        
        st.subheader(translations[lang_code]["dealers_header"])
        if st.session_state.dealer_data is not None:
            dealers = st.session_state.dealer_data[st.session_state.dealer_data['Ward'] == selected_ward]
            if not dealers.empty:
                st.write("**Available Agro-Dealers**:")
                for _, dealer in dealers.iterrows():
                    st.write(translations[lang_code]["dealer_info"].format(
                        dealer['agrodealerName'], dealer['market'], dealer.get('agrodealerPhone', 'N/A'),
                        dealer['Latitude'], dealer['Longitude']
                    ))
                
                st.subheader(translations[lang_code]["agrodealer_map"])
                m = folium.Map(location=[dealers['Latitude'].mean(), dealers['Longitude'].mean()], zoom_start=12)
                for _, dealer in dealers.iterrows():
                    if pd.notnull(dealer['Latitude']) and pd.notnull(dealer['Longitude']):
                        folium.Marker(
                            [dealer['Latitude'], dealer['Longitude']],
                            popup=f"{dealer['agrodealerName']} ({dealer['market']}) - Phone: {dealer.get('agrodealerPhone', 'N/A')}",
                            icon=folium.Icon(color="green")
                        ).add_to(m)
                st_folium(m, width=700, height=500)
            else:
                st.write(translations[lang_code]["dealers_none"])
    else:
        st.error(translations[lang_code]["error_data"])

# Research Institution Interface
elif user_type == translations["en"]["research_institution"]:
    st.header("Research Institution Dashboard")
    st.write("Conduct advanced soil fertility analysis, visualize data, and generate insights for maize farming in Trans Nzoia.")
    
    if st.session_state.merged_data is not None:
        if st.session_state.model_accuracy is not None:
            st.write(translations["en"]["model_accuracy"].format(st.session_state.model_accuracy))
        else:
            st.write("Model accuracy not available due to training failure.")
        
        required_wards = ["Kiminini", "Sirende", "Chepsiro/Kiptoror", "Sitatunga", "Kapomboi", "Kwanza"]
        data_wards = []
        if st.session_state.merged_data is not None:
            data_wards += st.session_state.merged_data['Ward'].dropna().str.lower().unique().tolist()
        if st.session_state.dealer_data is not None:
            data_wards += st.session_state.dealer_data['Ward'].dropna().str.lower().unique().tolist()
        wards = sorted(list(set(required_wards + [w.title() for w in data_wards])))
        selected_ward = st.selectbox("Select Ward for Analysis", wards)
        ward_data = st.session_state.merged_data[st.session_state.merged_data['Ward'] == selected_ward]
        
        st.subheader("Input Soil Parameters")
        input_data = {}
        with st.form("soil_input_form"):
            col1, col2 = st.columns(2)
            with col1:
                input_data["soil_pH"] = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.0, step=0.1)
                input_data["total_Nitrogen_percent_"] = st.number_input("Total Nitrogen (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
                input_data["phosphorus_Olsen_ppm"] = st.number_input("Phosphorus (Olsen, ppm)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
            with col2:
                input_data["potassium_meq_percent_"] = st.number_input("Potassium (meq%)", min_value=0.0, max_value=2.0, value=0.2, step=0.01)
                input_data["zinc_ppm"] = st.number_input("Zinc (ppm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                input_data["boron_ppm"] = st.number_input("Boron (ppm)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
            submit_button = st.form_submit_button("Submit Soil Data")
        
        if submit_button:
            recommendations, advice, explanation = get_fertilizer_recommendations_research(
                input_data, st.session_state.model, st.session_state.scaler, st.session_state.features, lang="en"
            )
            st.subheader("Model-Based Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
            st.write("**Insights for Agricultural Strategy**:")
            st.write(advice)
            
            if explanation:
                st.subheader(translations["en"]["feature_importance"])
                fig = px.bar(
                    x=list(explanation.keys()),
                    y=list(explanation.values()),
                    labels={'x': 'Soil Parameter', 'y': 'Importance'},
                    title="Feature Importance for Soil Fertility Prediction",
                    color_discrete_sequence=['#636EFA']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(translations["en"]["param_stats"])
        key_params = [
            "soil_pH", "total_Nitrogen_percent_", "total_Org_Carbon_percent_",
            "phosphorus_Olsen_ppm", "potassium_meq_percent_", "zinc_ppm", "boron_ppm",
            "electr_Conductivity_mS_per_cm"
        ]
        key_params = [col for col in key_params if col in ward_data.columns]
        if key_params:
            st.write(ward_data[key_params].describe())
            
            st.subheader(translations["en"]["soil_parameter_dist"])
            param = st.selectbox("Select Parameter to Visualize", key_params)
            if not ward_data[param].empty:
                fig = px.histogram(
                    ward_data, 
                    x=param,
                    nbins=20,
                    title=f"Distribution of {param} in {selected_ward}",
                    color_discrete_sequence=['#636EFA']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(translations["en"]["prediction_header"])
        predictions_df = predict_all_wards(st.session_state.merged_data, st.session_state.model, st.session_state.scaler, st.session_state.features)
        if not predictions_df.empty:
            st.write(predictions_df)
            
            fertility_counts = predictions_df['Fertility'].value_counts().reset_index()
            fertility_counts.columns = ['Fertility', 'Count']
            fig = px.pie(
                fertility_counts,
                values='Count',
                names='Fertility',
                title="Soil Fertility Distribution Across Wards",
                color='Fertility',
                color_discrete_map={'high': '#2ECC71', 'medium': '#F1C40F', 'low': '#E74C3C'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No predictions available due to missing model or data.")
        
        st.subheader("Agro-Dealer Network")
        if st.session_state.dealer_data is not None:
            dealers = st.session_state.dealer_data[st.session_state.dealer_data['Ward'] == selected_ward]
            if not dealers.empty:
                st.write(dealers[['agrodealerName', 'market', 'agrodealerPhone', 'Latitude', 'Longitude']])
                
                st.subheader(translations["en"]["agrodealer_map"])
                m = folium.Map(location=[dealers['Latitude'].mean(), dealers['Longitude'].mean()], zoom_start=12)
                for _, dealer in dealers.iterrows():
                    if pd.notnull(dealer['Latitude']) and pd.notnull(dealer['Longitude']):
                        folium.Marker(
                            [dealer['Latitude'], dealer['Longitude']],
                            popup=f"{dealer['agrodealerName']} ({dealer['market']})",
                            icon=folium.Icon(color="green")
                        ).add_to(m)
                st_folium(m, width=700, height=500)
            else:
                st.write("No agro-dealers found for this ward.")
        
        st.subheader("Data Export")
        csv = ward_data.to_csv(index=False)
        st.download_button("Download Ward Soil Data", csv, f"{selected_ward}_soil_data.csv", "text/csv")
        
        if st.button("Export Full Trans Nzoia Dataset"):
            full_csv = st.session_state.merged_data.to_csv(index=False)
            st.download_button("Download Full Dataset", full_csv, "trans_nzoia_soil_data.csv", "text/csv")
    else:
        st.error(translations["en"]["error_data"])

# Footer
st.markdown("---")
st.markdown(translations["en"]["footer"])
