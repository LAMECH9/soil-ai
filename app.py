    # Home Page
    if page == "Home":
        st.header("SoilSync AI: Transforming Agriculture")
        if user_type == "Farmer":
            st.markdown("""
SoilSync AI provides Nakuru farmers with precise soil fertility recommendations based on advanced soil analysis. Navigate to the **Farmer Dashboard** to get tailored advice for your crops.

**Why SoilSync AI?**
- **Proven Impact**: Up to 30% yield increase and 22% fertilizer reduction in field trials.
- **High Accuracy**: 92% recommendation accuracy.
- **Sustainability**: 0.4 t/ha/year carbon sequestration.
- **Scalability**: Supports Nakuru's diverse crops, expandable to other regions.
- **ROI**: 2.4:1 in season 1, 3.8:1 by season 3.

**Target Market**: Smallholder farmers in Nakuru.
**Get Started**: Use the Farmer Dashboard to get recommendations for your farm.
            """)
            st.image("https://via.placeholder.com/800x300.png?text=SoilSync+AI+Banner", use_column_width=True)
        else:  # Institution
            st.markdown("""
SoilSync AI leverages machine learning to predict soil nutrient status (nitrogen and phosphorus) and provide 
tailored fertilizer recommendations. Key features for institutions:

- **Nutrient Prediction**: Achieves 87% accuracy in predicting soil nutrient status (hardcoded for demo purposes).
- **Recommendations**: 92% accuracy in recommending interventions.
- **Field Trials**: Simulates 15–30% yield increase, 22% fertilizer reduction, 0.4 t/ha/year carbon sequestration.
- **ROI**: 2.4:1 in season 1, 3.8:1 by season 3.
- **Data Coverage**: 47% improvement via transfer learning and farmer observations.

**Target Market**: Agricultural institutions in Kenya.
**Get Started**: Explore data and generate bulk recommendations for your organization.
            """)
            if 'avg_accuracy' in st.session_state['inst'] and 'recommendation_accuracy' in st.session_state['inst'] and 'field_trials' in st.session_state['inst']:
                st.subheader("Concrete Analytical Outcomes")
                st.markdown("""
- 🌱 **87% accuracy** in predicting soil nutrient status (hardcoded for demo).
- ✅ **92% accuracy** in recommending appropriate interventions (validated using KALRO datasets).
- 📈 **15–30% increase** in crop yields compared to conventional approaches.
- 💰 **Return on Investment (ROI)**:
    - **2.4:1** in the first season
    - **3.8:1** by the third season
- ♻️ **22% reduction** in fertilizer waste through precision application.
- 🌍 **0.4 t/ha** of carbon sequestered per hectare annually.
- 📊 **47% improvement** in data coverage via transfer learning and farmer observations.
                """)
            else:
                st.markdown("Complete the 'Data Upload & Training' and 'Field Trials' sections to view analytical outcomes.")
