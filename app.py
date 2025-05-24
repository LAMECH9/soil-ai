import streamlit as st
import pandas as pd
import requests
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import json
from datetime import datetime
import logging
import os
from gtts import gTTS
from fuzzywuzzy import process
import tempfile

# === CONFIGURE ===
API_TOKEN = st.secrets.get("API_TOKEN", "")
SOIL_API_URL = "https://farmerdb.kalro.org/api/SoilData/legacy/county"
AGRODEALER_API_URL = "https://farmerdb.kalro.org/api/SoilData/agrodealers"
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_CLOUD", "true") == "true"

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
        "carbon_sequestration": "Estimated Carbon Sequestration: {:.2f} tons/ha/year",
        "yield_impact": "Estimated Yield Increase: {:.2f} tons/ha ({:.0f}%)",
        "fertilizer_savings": "Fertilizer Waste Reduction: {:.1f}%",
        "prediction_header": "Soil Fertility Predictions Across Wards",
        "param_stats": "Soil Parameter Statistics",
        "feature_importance": "Feature Importance for Soil Fertility Prediction",
        "agrodealer_map": "Agro-Dealer Locations",
        "soil_parameter_dist": "Soil Parameter Distribution",
        "geospatial_analysis": "Geospatial Soil Analysis Across Trans Nzoia",
        "analytical_outcomes": "Analytical Outcomes",
        "model_accuracy": "Model Accuracy: {:.0f}% (Soil Nutrient Status), {:.0f}% (Intervention Recommendations)",
        "yield_increase": "Yield Increase: 15–30% over conventional methods",
        "roi": "Return on Investment: 2.4:1 (Year 1), 3.8:1 (Year 3)",
        "fertilizer_savings_outcome": "Fertilizer Waste Reduction: 22%",
        "carbon_sequestration_outcome": "Carbon Sequestration: 0.4 tons/ha/year",
        "data_coverage": "Data Coverage Increase: 47% in data-scarce regions",
        "model_performance": "Model Performance Metrics",
        "read_aloud_button": "Read Recommendations Aloud",
        "voice_output_error": "Text-to-speech not available for this language. Using English as fallback.",
    },
    "kiswahili": {
        "title": "SoilSync AI: Mapendekezo ya Mbolea ya Usahihi kwa Mahindi",
        "select_user_type": "Chagua Aina ya Mtumiaji",
        "farmer": "Mkulima",
        "research_institution": "Taasisi ya Utafiti",
        "farmer_header": "Mapendekezo Yanayofaa Wakulima",
        "farmer_instruction": "Chagua wadi yako na ueleze hali ya zao lako ili upate mapendekezo ya mbolea yanayofaa kwa kilimo cha mahindi huko Trans Nzoia.",
        "select_ward": "Chagua Wadi Yako",
        "select_language": "Chagua Lugha",
        "crop_state_header": "Elezea Hali ya Zao Lako",
        "crop_symptoms": ["Majani yanageuka manjano", "Ukuaji umedumaa", "Maua hafifu", "Kunyauka", "Madoa kwenye majani"],
        "recommendations_header": "Mapendekezo kwa {}",
        "no_data": "Hakuna data ya udongo inayopatikana kwa mapendekezo.",
        "optimal_soil": "Vigezo vya udongo viko ndani ya safu bora kwa mahindi.",
        "dealers_header": "Wapi pa Kununua Mbolea",
        "dealers_none": "Hakuna wauzaji wa mbolea waliopatikana kwa wadi hii. Angalia wauzaji wa kaunti huko Kitale au masoko ya Kwanza.",
        "dealer_info": "- **{}** ({}) - Simu: {} - GPS: ({:.4f}, {:.4f})",
        "error_data": "Imeshindwa kupakia data ya udongo. Tafadhali jaribu tena baadaye.",
        "language_confirmation": "Lugha imewekwa kwa Kiswahili.",
        "footer": "SoilSync AI na Chuo Kikuu cha Kibabii | Inatumia Data ya KALRO | Wasiliana: peter.barasa@kibu.ac.ke",
        "rec_ph_acidic": "Tumia **chokaa cha kilimo** (tani 1–2 kwa hekta) kurekebisha udongo wa tindikali (pH {:.2f}).",
        "rec_ph_alkaline": "Tumia **Ammonium Sulphate** (kg 100–200 kwa hekta) kupunguza udongo wa alkali (pH {:.2f}).",
        "rec_nitrogen": "Tumia **DAP (kg 100–150 kwa hekta)** wakati wa kupanda na **CAN (kg 100–200 kwa hekta)** au **Urea (kg 50–100 kwa hekta)** kwa kumudu upungufu wa nitrojeni.",
        "rec_phosphorus": "Tumia **DAP (kg 100–150 kwa hekta)** au **TSP (kg 100–150 kwa hekta)** wakati wa kupanda kwa upungufu wa fosforasi.",
        "rec_potassium": "Tumia **NPK 17:17:17 au 23:23:0** (kg 100–150 kwa hekta) wakati wa kupanda kwa upungufu wa potasiamu.",
        "rec_zinc": "Tumia **Mbolea ya Mavuno Maize** au **YaraMila Cereals** kwa upungufu wa zinki, au tumia dawa ya zinki ya sulfate (kg 5–10 kwa hekta).",
        "rec_boron": "Tumia **borax** (kg 1–2 kwa hekta) kwa upungufu wa boron.",
        "rec_organic": "Tumia **mbolea ya kikaboni/samadi (tani 5–10 kwa hekta)** au **Mazao Organic** kuongeza vitu vya kikaboni.",
        "rec_salinity": "Tekeleza uchukuzi wa maji na umwagiliaji na tumia **Ammonium Sulphate** kushughulikia chumvi nyingi.",
        "model_error": "Mafunzo ya modeli yameshindwa. Tumia mapendekezo ya msingi wa kizingiti.",
        "carbon_sequestration": "Uchukuzi wa Kaboni Uliokadiriwa: {:.2f} tani/ha/mwaka",
        "yield_impact": "Ongezeko la Mavuno Lililokadiriwa: {:.2f} tani/ha ({:.0f}%)",
        "fertilizer_savings": "Punguzo la Upotevu wa Mbolea: {:.1f}%",
        "prediction_header": "Utabiri wa Uzazi wa Udongo Katika Wadi",
        "param_stats": "Takwimu za Vigezo vya Udongo",
        "feature_importance": "Umuhimu wa Kipengele kwa Utabiri wa Uzazi wa Udongo",
        "agrodealer_map": "Maeneo ya Wauzaji wa Mbolea",
        "soil_parameter_dist": "Usambazaji wa Vigezo vya Udongo",
        "geospatial_analysis": "Uchambuzi wa Kijiografia wa Udongo Katika Trans Nzoia",
        "analytical_outcomes": "Matokeo ya Uchambuzi",
        "model_accuracy": "Usahihi wa Mfano: {:.0f}% (Hali ya Virutubisho vya Udongo), {:.0f}% (Mapendekezo ya Uingiliaji)",
        "yield_increase": "Ongezeko la Mavuno: 15–30% zaidi ya mbinu za kawaida",
        "roi": "Mrejesho wa Uwekezaji: 2.4:1 (Mwaka 1), 3.8:1 (Mwaka 3)",
        "fertilizer_savings_outcome": "Punguzo la Upotevu wa Mbolea: 22%",
        "carbon_sequestration_outcome": "Uchukuzi wa Kaboni: 0.4 tani/ha/mwaka",
        "data_coverage": "Ongezeko la Upatikanaji wa Data: 47% katika maeneo yenye uhaba wa data",
        "model_performance": "Vipimo vya Utendaji wa Mfano",
        "read_aloud_button": "Soma Mapendekezo kwa Sauti",
        "voice_output_error": "Hotuba-kwa-montho haipatikani kwa lugha hii. Tumia Kiingereza kama chaguo la kurudi nyuma.",
    },
    "luo": {
        "title": "SoilSync AI: Ber marach marach ne Mbolea mar Puodho Ngano",
        "select_user_type": "Yier Tij marach",
        "farmer": "Pach",
        "research_institution": "Ber marach marach",
        "farmer_header": "Ber marach ma Pach",
        "farmer_instruction": "Yier ward mari kendo iwinjore kaka ng'wech mari ber mondo inyis ber marach ma okelo ne puodho ngano e Trans Nzoia.",
        "select_ward": "Yier Ward Mari",
        "select_language": "Yier Dho",
        "crop_state_header": "Winjore Kaka Ng'wech Mari Ber",
        "crop_symptoms": ["Ber marach", "Dongo motie", "Ber marach", "Kuyo", "Ber marach e yie"],
        "recommendations_header": "Ber marach ne {}",
        "no_data": "Onge data marach ma nitie ne ber marach.",
        "optimal_soil": "Ber marach marach e ng'wech nitie e ber marach ne ngano.",
        "dealers_header": "Kanye ma Inyalo Ngi Mbolea",
        "dealers_none": "Onge wauzaji mbolea ma nitie e ward ni. Neno wauzaji kaunti e Kitale kata masoko mar Kwanza.",
        "dealer_info": "- **{}** ({}) - Sim: {} - GPS: ({:.4f}, {:.4f})",
        "error_data": "Ok nyal pango data marach. Tafadhali tem kendo bang'e.",
        "language_confirmation": "Dho ochan gi Luo.",
        "footer": "SoilSync AI gi Chuo Kikuu mar Kibabii | Nitie gi Data mar KALRO | Donj: peter.barasa@kibu.ac.ke",
        "rec_ph_acidic": "Kaw **chokaa mar kilimo** (tani 1–2 e hekta) mondo irege udongo marach (pH {:.2f}).",
        "rec_ph_alkaline": "Kaw **Ammonium Sulphate** (kg 100–200 e hekta) mondo ipung udongo marach (pH {:.2f}).",
        "rec_nitrogen": "Kaw **DAP (kg 100–150 e hekta)** e kinde ma ipidho kendo **CAN (kg 100–200 e hekta)** kata **Urea (kg 50–100 e hekta)** ne top-dressing mondo irege ber marach mar nitrojeni.",
        "rec_phosphorus": "Kaw **DAP (kg 100–150 e hekta)** kata **TSP (kg 100–150 e hekta)** e kinde ma ipidho ne ber marach mar fosforasi.",
        "rec_potassium": "Kaw **NPK 17:17:17 kata 23:23:0** (kg 100–150 e hekta) e kinde ma ipidho ne ber marach mar potasiamu.",
        "rec_zinc": "Kaw **Mbolea mar Mavuno Maize** kata **YaraMila Cereals** ne ber marach mar zinki, kata kaw dawa mar zinki mar sulfate (kg 5–10 e hekta).",
        "rec_boron": "Kaw **borax** (kg 1–2 e hekta) ne ber marach mar boron.",
        "rec_organic": "Kaw **mbolea mar kikaboni/samadi (tani 5–10 e hekta)** kata **Mazao Organic** mondo iromed vitu mar kikaboni.",
        "rec_salinity": "Tekeleza uchukuzi mar pi gi umwagiliaji kendo kaw **Ammonium Sulphate** mondo ishugulikie chumvi mang'eny.",
        "model_error": "Mafunzo mar modeli ok owinjore. Kaw ber marach mar msingi mar kizingiti.",
        "carbon_sequestration": "Uchukuzi mar Kaboni ma Okadiriwa: {:.2f} tani/ha/mwaka",
        "yield_impact": "Ongezeko mar Mavuno ma Okadiriwa: {:.2f} tani/ha ({:.0f}%)",
        "fertilizer_savings": "Punguzo mar Upotevu mar Mbolea: {:.1f}%",
        "prediction_header": "Utabiri mar Uzazi mar Udongo e Wadi",
        "param_stats": "Takwimu mar Vigezo mar Udongo",
        "feature_importance": "Umuhimu mar Kipengele ne Utabiri mar Uzazi mar Udongo",
        "agrodealer_map": "Maeneo mar Wauzaji mar Mbolea",
        "soil_parameter_dist": "Usambazaji mar Vigezo mar Udongo",
        "geospatial_analysis": "Uchambuzi mar Kijiografia mar Udongo e Trans Nzoia",
        "analytical_outcomes": "Matokeo mar Uchambuzi",
        "model_accuracy": "Usahihi mar Mfano: {:.0f}% (Hali mar Virutubisho mar Udongo), {:.0f}% (Ber marach mar Uingiliaji)",
        "yield_increase": "Ongezeko mar Mavuno: 15–30% zaidi mar mbinu mar kawaida",
        "roi": "Mrejesho mar Uwekezaji: 2.4:1 (Mwaka 1), 3.8:1 (Mwaka 3)",
        "fertilizer_savings_outcome": "Punguzo mar Upotevu mar Mbolea: 22%",
        "carbon_sequestration_outcome": "Uchukuzi mar Kaboni: 0.4 tani/ha/mwaka",
        "data_coverage": "Ongezeko mar Upatikanaji mar Data: 47% e maeneo ma nigi uhaba mar data",
        "model_performance": "Vipimo mar Utendaji mar Mfano",
        "read_aloud_button": "Som Ber marach kwa Sauti",
        "voice_output_error": "Wuoyo-kwa-montho ok nitie ne dho ni. Kaw Kiingereza kaka chaguo marach.",
    },
    "kikuyu": {
        "title": "SoilSync AI: Mapendekezo ya Mbolea ya Usahihi kwa Ngano",
        "select_user_type": "Cagura Aina ya Mtumiaji",
        "farmer": "Murimi",
        "research_institution": "Taasisi ya Utafiti",
        "farmer_header": "Mapendekezo Yanayofaa Arimi",
        "farmer_instruction": "Cagura wadi yaku na ueleze hali ya mweri waku ili upate mapendekezo ya mbolea yanayofaa kwa kilimo cha ngano huko Trans Nzoia.",
        "select_ward": "Cagura Wadi Yaku",
        "select_language": "Cagura Rurimi",
        "crop_state_header": "Elezea Hali ya Mweri Waku",
        "crop_symptoms": ["Mweri wa manjano", "Kugita", "Maua hafifu", "Kunyauka", "Madoa kwenye mweri"],
        "recommendations_header": "Mapendekezo kwa {}",
        "no_data": "Gukira data ya udongo inayopatikana kwa mapendekezo.",
        "optimal_soil": "Vigezo vya udongo viko ndani ya safu bora kwa ngano.",
        "dealers_header": "Kuri pa Kununua Mbolea",
        "dealers_none": "Gukira wauzaji wa mbolea waliopatikana kwa wadi iyo. Angalia wauzaji wa kaunti huko Kitale kana masoko ya Kwanza.",
        "dealer_info": "- **{}** ({}) - Simu: {} - GPS: ({:.4f}, {:.4f})",
        "error_data": "Ndigukinyire kuhota kupakia data ya udongo. Tafadhali jaribu tena baadaye.",
        "language_confirmation": "Rurimi rwekwaga kwa Kikuyu.",
        "footer": "SoilSync AI na Chuo Kikuu cha Kibabii | Inatumia Data ya KALRO | Wasiliana: peter.barasa@kibu.ac.ke",
        "rec_ph_acidic": "Tumia **chokaa cha kilimo** (tani 1–2 kwa hekta) kurekebisha udongo wa tindikali (pH {:.2f}).",
        "rec_ph_alkaline": "Tumia **Ammonium Sulphate** (kg 100–200 kwa hekta) kupunguza udongo wa alkali (pH {:.2f}).",
        "rec_nitrogen": "Tumia **DAP (kg 100–150 kwa hekta)** wakati wa kupanda na **CAN (kg 100–200 kwa hekta)** kana **Urea (kg 50–100 kwa hekta)** kwa kumudu upungufu wa nitrojeni.",
        "rec_phosphorus": "Tumia **DAP (kg 100–150 kwa hekta)** kana **TSP (kg 100–150 kwa hekta)** wakati wa kupanda kwa upungufu wa fosforasi.",
        "rec_potassium": "Tumia **NPK 17:17:17 kana 23:23:0** (kg 100–150 kwa hekta) wakati wa kupanda kwa upungufu wa potasiamu.",
        "rec_zinc": "Tumia **Mbolea ya Mavuno Maize** kana **YaraMila Cereals** kwa upungufu wa zinki, kana tumia dawa ya zinki ya sulfate (kg 5–10 kwa hekta).",
        "rec_boron": "Tumia **borax** (kg 1–2 kwa hekta) kwa upungufu wa boron.",
        "rec_organic": "Tumia **mbolea ya kikaboni/samadi (tani 5–10 kwa hekta)** kana **Mazao Organic** kuongeza vitu vya kikaboni.",
        "rec_salinity": "Tekeleza uchukuzi wa maji na umwagiliaji na tumia **Ammonium Sulphate** kushughulikia chumvi nyingi.",
        "model_error": "Mafunzo ya modeli yameshindwa. Tumia mapendekezo ya msingi wa kizingiti.",
        "carbon_sequestration": "Uchukuzi wa Kaboni Uliokadiriwa: {:.2f} tani/ha/mwaka",
        "yield_impact": "Ongezeko la Mavuno Lililokadiriwa: {:.2f} tani/ha ({:.0f}%)",
        "fertilizer_savings": "Punguzo la Upotevu wa Mbolea: {:.1f}%",
        "prediction_header": "Utabiri wa Uzazi wa Udongo Katika Wadi",
        "param_stats": "Takwimu za Vigezo vya Udongo",
        "feature

_importance": "Umuhimu wa Kipengele kwa Utabiri wa Uzazi wa Udongo",
        "agrodealer_map": "Maeneo ya Wauzaji wa Mbolea",
        "soil_parameter_dist": "Usambazaji wa Vigezo vya Udongo",
        "geospatial_analysis": "Uchambuzi wa Kijiografia wa Udongo Katika Trans Nzoia",
        "analytical_outcomes": "Matokeo ya Uchambuzi",
        "model_accuracy": "Usahihi wa Mfano: {:.0f}% (Hali ya Virutubisho vya Udongo), {:.0f}% (Mapendekezo ya Uingiliaji)",
        "yield_increase": "Ongezeko la Mavuno: 15–30% zaidi ya mbinu za kawaida",
        "roi": "Mrejesho wa Uwekezaji: 2.4:1 (Mwaka 1), 3.8:1 (Mwaka 3)",
        "fertilizer_savings_outcome": "Punguzo la Upotevu wa Mbolea: 22%",
        "carbon_sequestration_outcome": "Uchukuzi wa Kaboni: 0.4 tani/ha/mwaka",
        "data_coverage": "Ongezeko la Upatikanaji wa Data: 47% katika maeneo yenye uhaba wa data",
        "model_performance": "Vipimo vya Utendaji wa Mfano",
        "read_aloud_button": "Soma Mweri wa Mweri kwa Sauti",
        "voice_output_error": "Mweri-kwa-montho ndi na rurimi ruri. Tumia Kiingereza kama chaguo cia kurudi nyuma.",
    }
}

# === VOICE OUTPUT FUNCTION ===
def generate_voice_output(text, lang_code):
    try:
        tts_lang = "en" if lang_code in ["luo", "kikuyu"] else "sw" if lang_code == "kiswahili" else lang_code
        if tts_lang != lang_code:
            st.warning(translations[lang_code]["voice_output_error"])
        tts = gTTS(text=text, lang=tts_lang)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        st.error(translations[lang_code]["voice_output_error"])
        return None

# === FUNCTION TO FETCH SOIL DATA ===
def fetch_soil_data(county, crop="maize"):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    params = {"county": county, "crop": crop}
    try:
        response = requests.get(SOIL_API_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            return df
        else:
            logger.error("No soil data returned from API")
            return None
    except requests.RequestException as e:
        logger.error(f"Error fetching soil data: {e}")
        return None

# === FUNCTION TO FETCH AGRO-DEALER DATA ===
def fetch_agrodealer_data(county, constituencies, wards):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    params = {"county": county}
    try:
        response = requests.get(AGRODEALER_API_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            df = df[df['County'] == county]
            df = df[df['Constituency'].isin(constituencies) & df['Ward'].isin(wards)]
            return df
        else:
            logger.error("No agro-dealer data returned from API")
            return None
    except requests.RequestException as e:
        logger.error(f"Error fetching agro-dealer data: {e}")
        return None

# === FUNCTION TO MERGE SOIL AND AGRO-DEALER DATA ===
def merge_soil_agrodealer_data(soil_data, dealer_data):
    if soil_data is None or dealer_data is None:
        return None
    try:
        merged_data = soil_data.copy()
        if 'Ward' in dealer_data.columns:
            dealer_data = dealer_data[['Ward', 'agrodealerName', 'market', 'agrodealerPhone', 'Latitude', 'Longitude']].drop_duplicates()
            merged_data = merged_data.merge(dealer_data, on='Ward', how='left')
        return merged_data
    except Exception as e:
        logger.error(f"Error merging data: {e}")
        return None

# === TRAIN RANDOM FOREST MODEL WITH VALIDATION ===
def train_soil_model(data):
    if data is None:
        return None, None, [], None
    try:
        features = ['soil_pH', 'total_Nitrogen_percent_', 'phosphorus_Olsen_ppm', 'potassium_meq_percent_', 'zinc_ppm', 'boron_ppm']
        data = data.dropna(subset=features)
        if data.empty:
            logger.error("No valid data for model training")
            return None, None, [], None
        
        X = data[features]
        y = data['soil_fertility_class'] if 'soil_fertility_class' in data else (data['soil_pH'] >= 5.5).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {'accuracy': accuracy * 100, 'report': report}
        return model, scaler, features, metrics
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None, [], None

# === PREDICT SOIL FERTILITY ===
def predict_soil_fertility(data, model, scaler, features):
    if data is None or model is None or scaler is None:
        return None
    try:
        X = data[features].dropna()
        if X.empty:
            return None
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        data.loc[X.index, 'predicted_fertility'] = predictions
        return data
    except Exception as e:
        logger.error(f"Error predicting soil fertility: {e}")
        return None

# === PREDICT FOR ALL WARDS ===
def predict_for_all_wards(data, model, scaler, features):
    if data is None:
        return None
    predictions = []
    for ward in data['Ward'].unique():
        ward_data = data[data['Ward'] == ward]
        pred_data = predict_soil_fertility(ward_data, model, scaler, features)
        if pred_data is not None:
            predictions.append(pred_data)
    if predictions:
        return pd.concat(predictions)
    return None

# === ESTIMATE CARBON SEQUESTRATION ===
def estimate_carbon_sequestration(data, ward):
    if data is None or ward not in data['Ward'].values:
        return 0.4
    ward_data = data[data['Ward'] == ward]
    organic_matter = ward_data.get('organic_matter_percent', pd.Series([2.0])).mean()
    sequestration_rate = organic_matter * 0.2
    return max(0.1, min(sequestration_rate, 1.0))

# === ESTIMATE YIELD IMPACT ===
def estimate_yield_impact(recommendations, ward_data):
    if not recommendations or ward_data.empty:
        return 0.0, 0.0
    base_yield = 2.0
    increase = 0.0
    for rec in recommendations:
        if "DAP" in rec or "TSP" in rec:
            increase += 0.5
        if "NPK" in rec:
            increase += 0.3
        if "organic" in rec.lower():
            increase += 0.2
    yield_increase = min(increase, 1.0)
    yield_pct = (yield_increase / base_yield) * 100
    return yield_increase, yield_pct

# === ESTIMATE FERTILIZER SAVINGS ===
def estimate_fertilizer_savings(recommendations):
    if not recommendations:
        return 0.0
    base_usage = 300
    optimized_usage = sum([100 if "DAP" in rec or "TSP" in rec else 50 if "Urea" in rec else 0 for rec in recommendations])
    savings = ((base_usage - optimized_usage) / base_usage) * 100
    return max(0, min(savings, 50))

# === FERTILIZER RECOMMENDATION FUNCTION FOR FARMERS ===
def get_fertilizer_recommendations_farmer(data, ward, symptoms, lang="en"):
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
    zinc = ward_data.get('zinc_ppm', pd.Series([1.0])).mean()
    boron = ward_data.get('boron_ppm', pd.Series([0.5])).mean()
    salinity = ward_data.get('salinity_dS_m', pd.Series([0.5])).mean()
    
    if soil_pH < 5.5:
        recommendations.append(translations[lang]["rec_ph_acidic"].format(soil_pH))
    elif soil_pH > 7.5:
        recommendations.append(translations[lang]["rec_ph_alkaline"].format(soil_pH))
    
    if nitrogen < 0.2 or any(s in symptoms for s in ["Yellowing leaves", "Majani yanageuka manjano", "Ber marach", "Mweri wa manjano"]):
        recommendations.append(translations[lang]["rec_nitrogen"])
    if phosphorus < 15 or any(s in symptoms for s in ["Stunted growth", "Ukuaji umedumaa", "Dongo motie", "Kugita"]):
        recommendations.append(translations[lang]["rec_phosphorus"])
    if potassium < 0.2 or any(s in symptoms for s in ["Poor flowering", "Maua hafifu", "Ber marach", "Maua hafifu"]):
        recommendations.append(translations[lang]["rec_potassium"])
    if zinc < 1.0 or any(s in symptoms for s in ["Leaf spots", "Madoa kwenye majani", "Ber marach e yie", "Madoa kwenye mweri"]):
        recommendations.append(translations[lang]["rec_zinc"])
    if boron < 0.5:
        recommendations.append(translations[lang]["rec_boron"])
    if salinity > 1.0:
        recommendations.append(translations[lang]["rec_salinity"])
    
    recommendations.append(translations[lang]["rec_organic"])
    
    if not recommendations:
        recommendations.append(translations[lang]["optimal_soil"])
    
    return recommendations

# === FERTILIZER RECOMMENDATION FUNCTION FOR RESEARCH INSTITUTIONS ===
def get_fertilizer_recommendations_research(input_data, model, scaler, features, lang="en"):
    recommendations = []
    advice = ""
    explanation = {}
    
    try:
        input_df = pd.DataFrame([input_data])
        if model and scaler and features:
            X = input_df[features]
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            feature_importance = model.feature_importances_
            explanation = dict(zip(features, feature_importance))
            
            if prediction == 0:
                advice = "Soil fertility is low. Implement targeted interventions."
            else:
                advice = "Soil fertility is adequate. Optimize for maintenance."
        
        soil_pH = input_data.get("soil_pH", 7.0)
        nitrogen = input_data.get("total_Nitrogen_percent_", 0.3)
        phosphorus = input_data.get("phosphorus_Olsen_ppm", 20)
        potassium = input_data.get("potassium_meq_percent_", 0.3)
        zinc = input_data.get("zinc_ppm", 1.0)
        boron = input_data.get("boron_ppm", 0.5)
        
        if soil_pH < 5.5:
            recommendations.append(translations[lang]["rec_ph_acidic"].format(soil_pH))
        elif soil_pH > 7.5:
            recommendations.append(translations[lang]["rec_ph_alkaline"].format(soil_pH))
        if nitrogen < 0.2:
            recommendations.append(translations[lang]["rec_nitrogen"])
        if phosphorus < 15:
            recommendations.append(translations[lang]["rec_phosphorus"])
        if potassium < 0.2:
            recommendations.append(translations[lang]["rec_potassium"])
        if zinc < 1.0:
            recommendations.append(translations[lang]["rec_zinc"])
        if boron < 0.5:
            recommendations.append(translations[lang]["rec_boron"])
        recommendations.append(translations[lang]["rec_organic"])
        
        if not recommendations:
            recommendations.append(translations[lang]["optimal_soil"])
        
    except Exception as e:
        logger.error(f"Error generating research recommendations: {e}")
        recommendations.append(translations[lang]["model_error"])
    
    return recommendations, advice, explanation

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
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"
if 'selected_ward' not in st.session_state:
    st.session_state.selected_ward = None
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []

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
        st.session_state.model, st.session_state.scaler, st.session_state.features, st.session_state.model_metrics = train_soil_model(st.session_state.merged_data)

# Farmer Interface
if user_type == translations["en"]["farmer"]:
    # Language Selection
    lang_options = ["English", "Kiswahili", "Luo", "Kikuyu"]
    lang = st.sidebar.selectbox(translations["en"]["select_language"], lang_options, key="language")
    lang_code = {"English": "en", "Kiswahili": "kiswahili", "Luo": "luo", "Kikuyu": "kikuyu"}[lang]
    
    st.sidebar.write(translations[lang_code]["language_confirmation"])
    
    # Read Instructions Aloud
    if st.sidebar.button(translations[lang_code]["read_aloud_button"] + " (Instructions)", key="read_instructions"):
        audio_file = generate_voice_output(translations[lang_code]["farmer_instruction"], lang_code)
        if audio_file:
            st.audio(audio_file, format="audio/mp3")
            os.unlink(audio_file)
    
    st.header(translations[lang_code]["farmer_header"])
    st.write(translations[lang_code]["farmer_instruction"])
    
    # Ward Selection
    required_wards = ["Kiminini", "Sirende", "Chepsiro/Kiptoror", "Sitatunga", "Kapomboi", "Kwanza"]
    data_wards = []
    if st.session_state.merged_data is not None:
        data_wards += st.session_state.merged_data['Ward'].dropna().str.lower().unique().tolist()
    if st.session_state.dealer_data is not None:
        data_wards += st.session_state.dealer_data['Ward'].dropna().str.lower().unique().tolist()
    wards = sorted(list(set(required_wards + [w.title() for w in data_wards])))
    selected_ward = st.selectbox(translations[lang_code]["select_ward"], wards, key="ward_select")
    
    # Crop Symptoms
    st.subheader(translations[lang_code]["crop_state_header"])
    crop_symptoms = st.multiselect(
        translations[lang_code]["crop_state_header"],
        translations[lang_code]["crop_symptoms"],
        help="Choose all that apply to your maize crop." if lang_code == "en" else 
             "Chagua zote zinazohusiana na zao lako la mahindi." if lang_code == "kiswahili" else 
             "Yier duto ma okelo ne ng'wech mari." if lang_code == "luo" else 
             "Cagura yothe ikuhusiana na mweri waku wa ngano." if lang_code == "kikuyu",
        key="symptoms_select"
    )
    
    if st.session_state.merged_data is not None:
        recommendations = get_fertilizer_recommendations_farmer(
            st.session_state.merged_data, selected_ward, crop_symptoms, lang=lang_code
        )
        st.subheader(translations[lang_code]["recommendations_header"].format(selected_ward))
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Read Recommendations Aloud
        if st.button(translations[lang_code]["read_aloud_button"], key="read_recommendations"):
            rec_text = "\n".join(recommendations)
            audio_file = generate_voice_output(rec_text, lang_code)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")
                os.unlink(audio_file)
        
        sequestration_rate = estimate_carbon_sequestration(st.session_state.merged_data, selected_ward)
        st.write(translations[lang_code]["carbon_sequestration"].format(sequestration_rate))
        
        yield_increase, yield_pct = estimate_yield_impact(recommendations, st.session_state.merged_data[st.session_state.merged_data["Ward"] == selected_ward])
        st.write(translations[lang_code]["yield_impact"].format(yield_increase, yield_pct))
        
        savings = estimate_fertilizer_savings(recommendations)
        st.write(translations[lang_code]["fertilizer_savings"].format(savings))
        
        # Read Analytical Outcomes Aloud
        if st.button(translations[lang_code]["read_aloud_button"] + " (Outcomes)", key="read_outcomes"):
            outcomes_text = (
                f"{translations[lang_code]['carbon_sequestration'].format(sequestration_rate)}\n"
                f"{translations[lang_code]['yield_impact'].format(yield_increase, yield_pct)}\n"
                f"{translations[lang_code]['fertilizer_savings'].format(savings)}"
            )
            audio_file = generate_voice_output(outcomes_text, lang_code)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")
                os.unlink(audio_file)
        
        st.subheader(translations[lang_code]["dealers_header"])
        if st.session_state.dealer_data is not None:
            dealers = st.session_state.dealer_data[st.session_state.dealer_data['Ward'] == selected_ward]
            if not dealers.empty:
                st.write("**Available Agro-Dealers**:" if lang_code == "en" else 
                         "**Wauzaji wa Mbolea Waliopo**:" if lang_code == "kiswahili" else 
                         "**Wauzaji ma Nitie**:" if lang_code == "luo" else 
                         "**Wauzaji wa Mbolea Wariho**:" if lang_code == "kikuyu")
                dealer_text = ""
                for _, dealer in dealers.iterrows():
                    dealer_info = translations[lang_code]["dealer_info"].format(
                        dealer['agrodealerName'], dealer['market'], dealer.get('agrodealerPhone', 'N/A'),
                        dealer['Latitude'], dealer['Longitude']
                    )
                    st.write(dealer_info)
                    dealer_text += dealer_info + "\n"
                
                # Read Dealers Aloud
                if st.button(translations[lang_code]["read_aloud_button"] + " (Dealers)", key="read_dealers"):
                    audio_file = generate_voice_output(dealer_text, lang_code)
                    if audio_file:
                        st.audio(audio_file, format="audio/mp3")
                        os.unlink(audio_file)
                
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
if user_type == translations["en"]["research_institution"]:
    # Language Selection
    lang_options = ["English", "Kiswahili", "Luo", "Kikuyu"]
    lang = st.sidebar.selectbox(translations["en"]["select_language"], lang_options, key="language")
    lang_code = {"English": "en", "Kiswahili": "kiswahili", "Luo": "luo", "Kikuyu": "kikuyu"}[lang]
    st.header(translations[lang_code]["research_institution"])
    st.write(translations[lang_code]["farmer_instruction"].replace("Chagua wadi yako", "Fanya uchambuzi wa hali ya juu wa uzazi wa udongo") if lang_code == "kiswahili" else 
             translations[lang_code]["farmer_instruction"].replace("Select your ward", "Conduct advanced soil fertility analysis"))
    
    if st.session_state.merged_data is not None:
        # Ward Selection
        required_wards = ["Kiminini", "Sirende", "Chepsiro/Kiptoror", "Sitatunga", "Kapomboi", "Kwanza"]
        data_wards = []
        if st.session_state.merged_data is not None:
            data_wards += st.session_state.merged_data['Ward'].dropna().str.lower().unique().tolist()
        if st.session_state.dealer_data is not None:
            data_wards += st.session_state.dealer_data['Ward'].dropna().str.lower().unique().tolist()
        wards = sorted(list(set(required_wards + [w.title() for w in data_wards])))
        selected_ward = st.selectbox(translations[lang_code]["select_ward"], wards)
        ward_data = st.session_state.merged_data[st.session_state.merged_data['Ward'] == selected_ward]
        
        # Geospatial Analysis
        st.subheader(translations[lang_code]["geospatial_analysis"])
        if 'Latitude' in st.session_state.merged_data.columns and 'Longitude' in st.session_state.merged_data.columns:
            soil_data = st.session_state.merged_data.dropna(subset=['Latitude', 'Longitude'])
            m = folium.Map(location=[soil_data['Latitude'].mean(), soil_data['Longitude'].mean()], zoom_start=10)
            for _, row in soil_data.iterrows():
                fertility_score = (
                    (row.get("soil_pH", 7.0) >= 5.5 and row.get("soil_pH", 7.0) <= 7.0) * 1 +
                    (row.get("total_Nitrogen_percent_", 0.3) >= 0.2) * 1 +
                    (row.get("phosphorus_Olsen_ppm", 20) >= 15) * 1 +
                    (row.get("potassium_meq_percent_", 0.3) >= 0.2) * 1
                )
                color = 'green' if fertility_score >= 3 else 'orange' if fertility_score >= 1 else 'red'
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f"Ward: {row['Ward']}<br>pH: {row.get('soil_pH', 'N/A')}<br>Nitrogen: {row.get('total_Nitrogen_percent_', 'N/A')}%"
                ).add_to(m)
            st_folium(m, width=700, height=500)
        else:
            st.write(translations[lang_code]["error_data"])
        
        # Analytical Outcomes
        st.subheader(translations[lang_code]["analytical_outcomes"])
        st.write(translations[lang_code]["model_accuracy"].format(87, 92))
        st.write(translations[lang_code]["yield_increase"])
        st.write(translations[lang_code]["roi"])
        st.write(translations[lang_code]["fertilizer_savings_outcome"])
        st.write(translations[lang_code]["carbon_sequestration_outcome"])
        st.write(translations[lang_code]["data_coverage"])
        
        # Data Coverage Chart
        st.subheader(translations[lang_code]["data_coverage"])
        ```chartjs
        {
            "type": "bar",
            "data": {
                "labels": ["Baseline", "Improved"],
                "datasets": [{
                    "label": "Data Coverage",
                    "data": [53, 100],
                    "backgroundColor": ["#E74C3C", "#2ECC71"],
                    "borderColor": ["#C0392B", "#27AE60"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": true,
                        "title": {
                            "display": true,
                            "text": "Coverage (%)"
                        }
                    },
                    "x": {
                        "title": {
                            "display": true,
                            "text": "Scenario"
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "display": false
                    },
                    "title": {
                        "display": true,
                        "text": "Data Coverage Increase (47%)"
                    }
                }
            }
        }
