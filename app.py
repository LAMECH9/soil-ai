import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Set page configuration
st.set_page_config(page_title="SoilSync AI: Precision Agriculture Platform", layout="wide")

# Function to load data with error handling
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Validate required columns
        required_columns = ['county', 'description of the farm', 'crop', 'soil ph', 'total nitrogen',
                           'phosphorus olsen', 'fertilizer recommendation']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}")
            return None
        return df
    except FileNotFoundError:
        st.error("Error: 'cleaned_soilsync_dataset.csv' not found in the current directory.")
        return None
    except pd.errors.EmptyDataError:
        st.error("Error: The CSV file is empty.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to get current time in EAT
def get_current_time_eat():
    eat_tz = pytz.timezone('Africa/Nairobi')
    return datetime.now(eat_tz).strftime("%I:%M %p EAT, %B %d, %Y")

# Load the dataset
data = load_data("cleaned_soilsync_dataset.csv")

# Main application
def main():
    # Header
    st.title("SoilSync AI: Precision Agriculture Platform")
    st.markdown("Empowering farmers and institutions with AI-driven soil fertility solutions.")
    st.write(f"Current time: {get_current_time_eat()}")

    # Navigation
    st.sidebar.header("Navigation")
    language = st.sidebar.selectbox("Select Language", ["English"])
    page = st.sidebar.selectbox("Select Page", ["Home", "Farmer Dashboard"])

    # User Type Selection
    st.sidebar.header("Select User Type")
    user_type = st.sidebar.radio("", ["User"])

    if page == "Home":
        st.header("Welcome to SoilSync AI")
        st.write("""
        SoilSync AI is designed to provide farmers with precise soil fertility recommendations based on
        advanced soil analysis. Select 'Farmer Dashboard' to get tailored advice for your farm.
        """)
        st.image("https://via.placeholder.com/800x300.png?text=SoilSync+AI+Banner", use_column_width=True)

    elif page == "Farmer Dashboard":
        st.header("Farmer Dashboard")
        st.write("Welcome, farmer! Get tailored soil recommendations for your farm.")

        if data is None or data.empty:
            st.error("Unable to load data. Please ensure the CSV file is available and try again.")
            return

        # Input form
        with st.form("farmer_input_form"):
            col1, col2 = st.columns(2)
            with col1:
                # County selection (unique counties from dataset)
                counties = sorted(data['county'].unique())
                county = st.selectbox("Select your county", counties)
                
                # Ward selection (based on selected county)
                wards = sorted(data[data['county'] == county]['description of the farm'].unique())
                ward = st.selectbox("Select your ward", wards)

            with col2:
                # Crop selection (based on selected ward)
                ward_data = data[(data['county'] == county) & (data['description of the farm'] == ward)]
                if not ward_data.empty:
                    # Extract crops, handling comma-separated values
                    crops = []
                    for crop_list in ward_data['crop']:
                        if isinstance(crop_list, str):
                            crops.extend([c.strip() for c in crop_list.split(',')])
                    crops = sorted(list(set(crops)))
                else:
                    crops = []
                crop = st.selectbox("Select your crop", crops if crops else ["No crops available"])

                # Symptoms selection
                symptoms = st.multiselect("Select observed symptoms",
                                        ["Yellowing leaves", "Stunted growth", "Poor yield", "Wilting",
                                         "Nutrient deficiency", "Other"])

            submitted = st.form_submit_button("Get Recommendations")

        if submitted:
            if not crops:
                st.warning("No crops available for the selected ward. Please try a different ward.")
                return

            if crop == "No crops available":
                st.warning("Please select a valid crop.")
                return

            # Filter data for recommendations
            try:
                # Get relevant data
                recommendation_data = data[
                    (data['county'] == county) &
                    (data['description of the farm'] == ward) &
                    (data['crop'].str.contains(crop, case=False, na=False))
                ]

                if recommendation_data.empty:
                    st.warning(f"No specific recommendations found for {crop} in {ward}, {county}. Showing general advice.")
                    recommendation_data = data[
                        (data['county'] == county) &
                        (data['description of the farm'] == ward)
                    ]

                if not recommendation_data.empty:
                    st.subheader("Soil Analysis and Recommendations")
                    for _, row in recommendation_data.iterrows():
                        with st.expander(f"Recommendation for Farm ID {row['id']}"):
                            st.write(f"**Soil pH**: {row['soil ph']} ({row['soil ph class']})")
                            st.write(f"**Total Nitrogen**: {row['total nitrogen']} ({row['total nitrogenclass']})")
                            st.write(f"**Phosphorus (Olsen)**: {row['phosphorus olsen']} ({row['phosphorus olsen class']})")
                            st.write("**Fertilizer Recommendation**:")
                            st.markdown(row['fertilizer recommendation'])
                            
                            # Symptom-based advice
                            if symptoms:
                                st.write("**Symptom-Based Advice**:")
                                for symptom in symptoms:
                                    if symptom == "Yellowing leaves":
                                        st.write("- **Yellowing leaves**: Often indicates nitrogen or iron deficiency. Consider applying nitrogen-rich fertilizers like CAN or foliar sprays with iron.")
                                    elif symptom == "Stunted growth":
                                        st.write("- **Stunted growth**: May be due to phosphorus or potassium deficiency. Apply fertilizers like TSP or N:P:K 23:23:0.")
                                    elif symptom == "Poor yield":
                                        st.write("- **Poor yield**: Could result from multiple nutrient deficiencies. Ensure balanced fertilization and soil testing.")
                                    elif symptom == "Wilting":
                                        st.write("- **Wilting**: Check irrigation and drainage. May also indicate potassium deficiency.")
                                    elif symptom == "Nutrient deficiency":
                                        st.write("- **Nutrient deficiency**: Conduct a full soil test to identify specific deficiencies.")
                                    elif symptom == "Other":
                                        st.write("- **Other**: Please consult an agricultural extension officer for detailed diagnosis.")
                else:
                    st.error("No data available for the selected inputs. Please try different selections or contact support.")
            except Exception as e:
                st.error(f"Error processing request: {str(e)}. Please try again or contact support.")

if __name__ == "__main__":
    main()
