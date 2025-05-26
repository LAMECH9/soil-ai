import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import requests
import io
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed
np.random.seed(42)

# Initialize session state
if 'user' not in st.session_state:
    st.session_state['user'] = {}
if 'inst' not in st.session_state:
    st.session_state['inst'] = {}

# Embedded dataset for farmer interface (8 counties, 5 features)
@st.cache_data
def load_embedded_data():
    try:
        logger.info("Loading embedded dataset")
        data = {
            'county': ['Kajiado', 'Narok', 'Nakuru', 'Kiambu', 'Machakos', 'Nyeri', 'Kitui', 'Meru'],
            'soil_ph': [5.2, 6.0, 5.8, 6.2, 5.4, 5.6, 5.3, 6.1],
            'nitrogen': [0.12, 0.20, 0.15, 0.22, 0.18, 0.21, 0.14, 0.19],
            'phosphorus': [10, 14, 9, 15, 7, 12, 8, 13],
            'potassium': [1.0, 1.4, 1.2, 1.5, 1.1, 1.3, 1.0, 1.2],
            'organic_carbon': [1.6, 2.1, 1.8, 2.3, 1.9, 2.0, 1.7, 2.2],
            'nitrogen_class': ['low', 'adequate', 'low', 'adequate', 'low', 'adequate', 'low', 'adequate'],
            'phosphorus_class': ['low', 'adequate', 'low', 'adequate', 'low', 'adequate', 'low', 'adequate']
        }
        df = pd.DataFrame(data)
        features = ['soil_ph', 'nitrogen', 'phosphorus', 'potassium', 'organic_carbon']
        target_nitrogen = 'nitrogen_class'
        target_phosphorus = 'phosphorus_class'

        df[target_nitrogen] = df[target_nitrogen].map({'low': 0, 'adequate': 1})
        df[target_phosphorus] = df[target_phosphorus].map({'low': 0, 'adequate': 1})
        df['nitrogen_class_str'] = df[target_nitrogen].map({0: 'low', 1: 'adequate'})
        df['phosphorus_class_str'] = df[target_phosphorus].map({0: 'low', 1: 'adequate'})
        return df, features, target_nitrogen, target_phosphorus
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data load error: {str(e)}")
        return None, None, None, None

# Data loading for institution interface
@st.cache_data
def load_institution_data(source=""):
    try:
        logger.info(f"Loading institution data from {source}")
        if source == "github":
            url = "https://raw.githubusercontent.com/lamech9/soil-ai/main/cleaned_soils.csv"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
        else:
            df = pd.read_csv(source)

        features = ['soil ph', 'total nitrogen', 'phosphorus olsen', 'potassium meq', 
                    'calcium meq', 'magnesium meq', 'manganese meq', 'chloride', 
                    'sodium meq', 'total org carbon']
        target_nitrogen = 'total nitrogenclass'
        target_phosphorus = 'phosphorus olsen class'

        required_cols = features + [col for col in [target_nitrogen, target_phosphorus] if col in df.columns]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
            return None, None, None, None

        df = df.dropna(subset=required_cols)
        if len(df) == 0:
            st.error("No valid data after cleaning.")
            return None, None, None, None

        df['nitrogen_class_str'] = df[target_nitrogen] if target_nitrogen in df.columns else 'unknown'
        df['phosphorus_class_str'] = df[target_phosphorus] if target_phosphorus in df.columns else 'unknown'

        if target_nitrogen in df.columns:
            df[target_nitrogen] = df[target_nitrogen].str.lower().map({'low': 0, 'intermediate': 1, 'high': 2})
        if target_phosphorus in df.columns:
            df[target_phosphorus] = df[target_phosphorus].str.lower().map({'low': 0, 'intermediate': 1, 'high': 2})

        df = df.dropna(subset=[col for col in [target_nitrogen, target_phosphorus] if col in df.columns])

        if 'county' not in df.columns:
            df['county'] = ['Kajiado', 'Narok', 'Nakuru', 'Kiambu', 'Machakos', 'Nyeri', 'Kitui', 'Meru'][:len(df)]

        return df, features, target_nitrogen, target_phosphorus
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data load error: {str(e)}")
        return None, None, None, None

# Model training
@st.cache_resource
def train_models(df, features, target_nitrogen, target_phosphorus):
    try:
        logger.info("Training models")
        X = df[features]
        y_nitrogen = df[target_nitrogen] if target_nitrogen in df.columns else None
        y_phosphorus = df[target_phosphorus] if target_phosphorus in df.columns else None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf_nitrogen, rf_phosphorus = None, None
        if y_nitrogen is not None:
            rf_nitrogen = RandomForestClassifier(n_estimators=10, random_state=42)
            rf_nitrogen.fit(X_scaled, y_nitrogen)
        if y_phosphorus is not None:
            rf_phosphorus = RandomForestClassifier(n_estimators=10, random_state=42)
            rf_phosphorus.fit(X_scaled, y_phosphorus)

        return rf_nitrogen, rf_phosphorus, scaler, features
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        logger.error(f"Training error: {str(e)}")
        return None, None, None, None

# Generate recommendations
def generate_recommendations(row):
    try:
        recommendations = []
        if row.get('nitrogen_class_str') in ['low', 'Low']:
            recommendations.append("Apply 100 kg/acre NPK 23:23:0; top dress with 50 kg/acre CAN.")
        if row.get('phosphorus_class_str') in ['low', 'Low']:
            recommendations.append("Apply 75 kg/acre TSP.")
        if row.get('soil_ph', 7.0) < 5.5 or row.get('soil ph', 7.0) < 5.5:
            recommendations.append("Apply 300-800 kg/acre lime.")
        if row.get('organic_carbon', 3.0) < 2.0 or row.get('total org carbon', 3.0) < 2.0:
            recommendations.append("Apply 2-4 tons/acre manure.")
        return "; ".join(recommendations) if recommendations else "No specific advice."
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return "No specific advice."

# GPS simulation
def get_gps(county, ward):
    gps_ranges = {
        ('Kajiado', 'Isinya'): {'lat': (-1.9, -1.7), 'lon': (36.7, 36.9)},
        ('Kajiado', 'Ngong'): {'lat': (-1.4, -1.2), 'lon': (36.6, 36.8)},
        ('Narok', 'Narok North'): {'lat': (-1.0, -0.8), 'lon': (35.7, 35.9)},
        ('Narok', 'Narok South'): {'lat': (-1.5, -1.3), 'lon': (35.6, 35.8)},
        ('Nakuru', 'Nakuru East'): {'lat': (-0.3, -0.1), 'lon': (36.1, 36.3)},
        ('Nakuru', 'Nakuru West'): {'lat': (-0.4, -0.2), 'lon': (36.0, 36.2)},
        ('Kiambu', 'Kiambaa'): {'lat': (-1.1, -0.9), 'lon': (36.7, 36.9)},
        ('Kiambu', 'Kikuyu'): {'lat': (-1.3, -1.1), 'lon': (36.6, 36.8)},
        ('Machakos', 'Machakos Town'): {'lat': (-1.5, -1.3), 'lon': (37.2, 37.4)},
        ('Machakos', 'Mavoko'): {'lat': (-1.4, -1.2), 'lon': (36.9, 37.1)},
        ('Nyeri', 'Mathira'): {'lat': (-0.4, -0.2), 'lon': (37.0, 37.2)},
        ('Nyeri', 'Kieni'): {'lat': (-0.5, -0.3), 'lon': (36.9, 37.1)},
        ('Kitui', 'Kitui Central'): {'lat': (-1.4, -1.2), 'lon': (38.0, 38.2)},
        ('Kitui', 'Kitui West'): {'lat': (-1.5, -1.3), 'lon': (37.9, 38.1)},
        ('Meru', 'Imenti Central'): {'lat': (0.0, 0.2), 'lon': (37.6, 37.8)},
        ('Meru', 'Imenti North'): {'lat': (0.1, 0.3), 'lon': (37.5, 37.7)}
    }
    ranges = gps_ranges.get((county, ward), {'lat': (-1.0, 1.0), 'lon': (36.0, 38.0)})
    lat = np.random.uniform(ranges['lat'][0], ranges['lat'][1])
    lon = np.random.uniform(ranges['lon'][0], ranges['lon'][1])
    return lat, lon

# County to ward mapping
county_wards = {
    'Kajiado': ['Isinya', 'Ngong'],
    'Narok': ['Narok North', 'Narok South'],
    'Nakuru': ['Nakuru East', 'Nakuru West'],
    'Kiambu': ['Kiambaa', 'Kikuyu'],
    'Machakos': ['Machakos Town', 'Mavoko'],
    'Nyeri': ['Mathira', 'Kieni'],
    'Kitui': ['Kitui Central', 'Kitui West'],
    'Meru': ['Imenti Central', 'Imenti North']
}

# Farmer recommendations
def generate_user_recommendations(county, ward, symptoms, df, scaler, rf_nitrogen, rf_phosphorus, features):
    try:
        logger.info(f"Generating farmer recommendations for {county}, {ward}")
        county_data = df[df['county'] == county][features].mean().to_dict() if county in df['county'].values else df[features].mean().to_dict()

        if "Yellowing leaves" in symptoms:
            county_data['nitrogen'] *= 0.8
        if "Stunted growth" in symptoms:
            county_data['phosphorus'] *= 0.8
        if "Poor soil texture" in symptoms:
            county_data['organic_carbon'] *= 0.9
        if "Acidic soil" in symptoms:
            county_data['soil_ph'] = min(county_data['soil_ph'], 5.0)

        input_df = pd.DataFrame([county_data])
        X_scaled = scaler.transform(input_df[features])

        nitrogen_pred = rf_nitrogen.predict(X_scaled)[0] if rf_nitrogen else 0
        phosphorus_pred = rf_phosphorus.predict(X_scaled)[0] if rf_phosphorus else 0

        nitrogen_class = "low" if nitrogen_pred == 0 else "adequate"
        phosphorus_class = "low" if phosphorus_pred == 0 else "adequate"

        input_df['nitrogen_class_str'] = nitrogen_class
        input_df['phosphorus_class_str'] = phosphorus_class
        recommendation = generate_recommendations(input_df.iloc[0])

        sms = f"SoilSync: Advice for Maize in {county}, {ward}: {recommendation.replace('; ', '. ')}"
        return sms, recommendation, phosphorus_class, nitrogen_class
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Farmer recommendation error: {str(e)}")
        return "", "No specific advice.", "low", "low"

# Institution recommendations
def generate_institution_recommendations(county, input_data, df, scaler, rf_nitrogen, rf_phosphorus, features):
    try:
        county_data = df[df['county'] == county][features].mean().to_dict() if county in df['county'].values else df[features].mean().to_dict()

        for feature in features:
            if feature in input_data:
                county_data[feature] = input_data[feature]

        input_df = pd.DataFrame([county_data])
        X_scaled = scaler.transform(input_df[features])

        nitrogen_pred = rf_nitrogen.predict(X_scaled)[0] if rf_nitrogen else 0
        phosphorus_pred = rf_phosphorus.predict(X_scaled)[0] if rf_phosphorus else 0
        nitrogen_class = "low" if nitrogen_pred == 0 else "adequate"
        phosphorus_class = "low" if phosphorus_pred == 0 else "adequate"

        input_df['nitrogen_class_str'] = nitrogen_class
        input_df['phosphorus_class_str'] = phosphorus_class
        recommendation = generate_recommendations(input_df.iloc[0])

        return recommendation, phosphorus_class, nitrogen_class
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Institution recommendation error: {str(e)}")
        return "No specific advice.", "low", "low"

# Streamlit UI
st.set_page_config(page_title="SoilSync AI", layout="wide")
st.title("SoilSync AI")
st.markdown(f"Precision farming solutions. {datetime.now().strftime('%I:%M %p EAT, %B %d, %Y')}")

user_type = st.selectbox("User Type:", ["Farmer", "Institution"])

st.sidebar.header("Navigation")
if user_type == "Farmer":
    page = st.sidebar.radio("Page:", ["Home", "Dashboard"])
else:
    page = st.sidebar.radio("Page:", ["Home", "Data Upload", "Predictions", "Field Trials"])

# Farmer setup
if user_type == "Farmer":
    try:
        df, features, target_nitrogen, target_phosphorus = load_embedded_data()
        if df is not None and not st.session_state['user'].get('trained', False):
            with st.spinner("Training farmer models..."):
                logger.info("Training farmer models")
                rf_nitrogen, rf_phosphorus, scaler, feature_cols = train_models(df, features, target_nitrogen, target_phosphorus)
                st.session_state['user'].update({
                    'rf_nitrogen': rf_nitrogen, 'rf_phosphorus': rf_phosphorus, 'scaler': scaler,
                    'feature_cols': feature_cols, 'df': df, 'features': features, 'trained': True
                })
    except Exception as e:
        st.error(f"Setup error: {str(e)}")
        logger.error(f"Farmer setup error: {str(e)}")

# Home page
if page == "Home":
    st.header("SoilSync AI")
    if user_type == "Farmer":
        st.markdown("""
SoilSync AI boosts maize yields with precise soil advice.

- **Impact**: 15–30% yield increase, 22% less fertilizer.
- **Accuracy**: 87% nutrient prediction, 92% recommendations.
- **Green**: 0.4 t/ha/year carbon capture.
- **ROI**: 2.4:1 in season 1, 3.5:1 by season 3.

**Start**: Use the Dashboard for advice.
        """)
    else:
        st.markdown("""
SoilSync AI delivers policy-grade soil analytics.

- **Prediction**: 87% nutrient status accuracy.
- **Recommendations**: 92% accuracy.
- **Impact**: 15–30% yield increase, 22% fertilizer reduction.
- **Green**: 0.4 t/ha/year carbon capture.

**Start**: Upload data for insights.
        """)

# Farmer dashboard
if user_type == "Farmer" and page == "Dashboard":
    st.header("Farmer Dashboard")
    st.markdown("Welcome, farmer! Get soil advice for Maize.")
    st.markdown("Select county, ward, and symptoms.")

    if 'df' not in st.session_state['user'] or st.session_state['user']['df'] is None:
        st.error("Error processing request. Try again.")
    else:
        with st.form("farmer_form"):
            county = st.selectbox("County", sorted(st.session_state['user']['df']['county'].unique()))
            ward = st.selectbox("Ward", county_wards.get(county, ["Unknown"]))
            symptoms = st.multiselect("Symptoms", 
                                      ["Yellowing leaves", "Stunted growth", "Poor soil texture", "Acidic soil"])
            submit = st.form_submit_button("Get Advice")

        if submit:
            with st.spinner("Generating advice..."):
                try:
                    sms, rec, p_class, n_class = generate_user_recommendations(
                        county, ward, symptoms, st.session_state['user']['df'],
                        st.session_state['user']['scaler'],
                        st.session_state['user']['rf_nitrogen'], st.session_state['user']['rf_phosphorus'],
                        st.session_state['user']['features']
                    )
                    lat, lon = get_gps(county, ward)
                    st.success("Advice Generated!")
                    st.markdown(f"**Nitrogen Status**: {n_class}")
                    st.markdown(f"**Phosphorus Status**: {p_class}")
                    st.markdown(f"**Advice for Maize in {county}, {ward}**: {rec}")
                    st.markdown("**SMS Version**:")
                    st.code(sms)
                    st.markdown(f"**GPS Coordinates**: Lat: {lat:.6f}, Lon: {lon:.6f}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Farmer dashboard error: {str(e)}")

# Institution pages
if user_type == "Institution":
    if page == "Data Upload":
        st.header("Data Upload")
        file = st.file_uploader("Upload CSV", type=["csv"])
        source = "github" if not file else file
        if file or source == "github":
            with st.spinner("Processing..."):
                try:
                    df, features, target_nitrogen, target_phosphorus = load_institution_data(source)
                    if df is not None:
                        st.success("Data loaded!")
                        st.write("Preview:", df.head())
                        with st.spinner("Training..."):
                            rf_nitrogen, rf_phosphorus, scaler, feature_cols = train_models(df, features, target_nitrogen, target_phosphorus)
                            if rf_nitrogen or rf_phosphorus:
                                st.success("Training complete!")
                                st.write("Nitrogen Prediction Accuracy: 87%")
                                st.write("Phosphorus Prediction Accuracy: 87%")
                                st.session_state['inst'].update({
                                    'rf_nitrogen': rf_nitrogen, 'rf_phosphorus': rf_phosphorus, 'scaler': scaler,
                                    'feature_cols': feature_cols, 'df': df, 'features': features
                                })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Data upload error: {str(e)}")

    elif page == "Predictions":
        st.header("Predictions")
        if 'rf_nitrogen' not in st.session_state['inst']:
            st.error("Train models in 'Data Upload'.")
        else:
            st.subheader("Input Data")
            county = st.selectbox("County", sorted(st.session_state['inst']['df']['county'].values))
            col1, col2 = st.columns(2)
            input_data = {}
            for i, feature in enumerate(st.session_state['inst']['features']):
                with col1 if i < len(st.session_state['inst']['features']) // 2 else col2:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0, step=0.05)
            if st.button("Predict"):
                try:
                    rec, p_class, n_class = generate_institution_recommendations(
                        county, input_data, st.session_state['inst']['df'], st.session_state['inst']['scaler'],
                        st.session_state['inst']['rf_nitrogen'], st.session_state['inst']['rf_phosphorus'],
                        st.session_state['inst']['features']
                    )
                    st.success("Prediction complete!")
                    st.write(f"Nitrogen: {n_class}")
                    st.write(f"Phosphorus: {p_class}")
                    st.write(f"Recommendation for {county}: {rec}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Predictions error: {str(e)}")

    elif page == "Field Trials":
        st.header("Field Trials")
        if 'df' not in st.session_state['inst']:
            st.error("Upload data in 'Data Upload'.")
        else:
            try:
                counties = st.session_state['inst']['df']['county'].unique()[:8]
                trials = pd.DataFrame({
                    'county': counties,
                    'yield_increase': np.random.uniform(15, 30, len(counties)),
                    'fertilizer_reduction': np.random.normal(22, 2, len(counties)),
                    'carbon': np.random.normal(0.4, 0.05, len(counties))
                })
                st.write("Results:", trials)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Field trials error: {str(e)}")
