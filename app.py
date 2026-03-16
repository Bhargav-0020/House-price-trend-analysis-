import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1, h2, h3 {
    color: #1f4e79;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* KPI Cards */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    text-align: center;
}

.card h3 {
    margin-bottom: 5px;
    color: #2E86C1;
}

.card p {
    font-size: 22px;
    font-weight: bold;
}

/* Prediction Box */
.success-box {
    background-color: #e8f8f5;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #1abc9c;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
df = pd.read_csv("dataset/housing.csv")

st.set_page_config(
    page_title="House Price Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)
selected = option_menu(
    menu_title=None,
    options=["Home", "Data Analysis", "Trend Analysis", "Price Prediction"],
    icons=["house", "bar-chart", "graph-up", "currency-rupee"],
    default_index=0,
    orientation="horizontal",
)

#Home page
if selected == "Home":

    st.markdown(
        "<h1 style='text-align:center;'>🏠 House Price Trends Analysis Using Real Estate Data</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h4 style='text-align:center; color:gray;'>Data-Driven Property Insights & Price Prediction System</h4>",
        unsafe_allow_html=True
    )

    st.write("")

    # Overview Section
    st.subheader("📌 Project Overview")

    st.write("""
    This dashboard provides comprehensive analysis of real estate data and predicts
    property prices using a Machine Learning model (Linear Regression).

    It helps users understand:
    • Market price trends  
    • Feature influence on house prices  
    • Location-based pricing differences  
    • Real-time property valuation  
    """)

    st.write("")

    # How It Works Section
    st.subheader("⚙️ How It Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Step 1 — Data Analysis**
        - Explore dataset statistics  
        - Visualize correlations  
        - Identify key price drivers  
        """)

    with col2:
        st.markdown("""
        **Step 2 — Machine Learning**
        - Linear Regression model  
        - Feature scaling applied  
        - Model trained & evaluated  
        - Real-time price prediction  
        """)

    st.write("")

    # Technology Section
    st.subheader("🛠 Technologies Used")

    st.markdown("""
    - Python  
    - Pandas & NumPy  
    - Matplotlib & Seaborn  
    - Plotly  
    - Scikit-learn  
    - Streamlit  
    """)

    st.write("")

    st.info("Use the navigation menu above to explore Data Analysis, Trends, and Price Prediction.")

    # KPI Cards
    col1, col2, col3 = st.columns(3)

    avg_price = int(df["price"].mean())
    max_price = int(df["price"].max())
    total_houses = len(df)

    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>Average Price</h3>
            <p>₹ {avg_price:,}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h3>Highest Price</h3>
            <p>₹ {max_price:,}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <h3>Total Properties</h3>
            <p>{total_houses}</p>
        </div>
        """, unsafe_allow_html=True)
        
#DATA ANALYSIS PAGE
elif selected == "Data Analysis":

    st.title("📊 Data Analysis & Feature Relationships")

    # Dataset preview
    st.subheader("Dataset Preview (First 5 Records)")
    st.dataframe(df.head())

    # Statistical summary
    st.subheader("Statistical Summary")
    st.write(
        "This table shows mean, minimum, maximum and spread of numerical features in the housing dataset."
    )
    st.dataframe(df.describe())

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    st.write(
        "The heatmap shows how strongly each feature affects house price. "
        "Darker color means stronger relationship with price."
    )

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['int64','float64'])

    # Create centered layout
    left, center, right = st.columns([1,2,1])

    with center:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5,
            fmt=".2f",
            annot_kws={"size":8},
            ax=ax
        )
        ax.set_title("Correlation Matrix", fontsize=12)
        st.pyplot(fig)

#TREND ANALYSIS
# ================= TREND ANALYSIS =================
elif selected == "Trend Analysis":

    st.title("📈 Real Estate Market Trend Analysis")
    st.write("This section visualizes how different features influence house prices.")

    # -------- ROW 1 --------
    col1, col2 = st.columns(2)

    # Area vs Price
    with col1:
        st.subheader("Area vs Price Relationship")
        st.caption("Larger houses generally have higher prices.")

        fig1, ax1 = plt.subplots(figsize=(5,3))
        sns.scatterplot(x="area", y="price", data=df, ax=ax1, color="#2E86C1")
        ax1.set_xlabel("Area (sqft)")
        ax1.set_ylabel("Price")
        st.pyplot(fig1)

    # Price Distribution
    with col2:
        st.subheader("Price Distribution")
        st.caption("Shows how property prices are distributed in the dataset.")

        fig2, ax2 = plt.subplots(figsize=(5,3))
        sns.histplot(df["price"], kde=True, ax=ax2, color="#1ABC9C")
        ax2.set_xlabel("House Price")
        st.pyplot(fig2)

    # -------- ROW 2 --------
    col3, col4 = st.columns(2)

    # Bedrooms vs Price
    with col3:
        st.subheader("Bedrooms vs Price")
        st.caption("Houses with more bedrooms tend to cost more.")

        fig3, ax3 = plt.subplots(figsize=(5,3))
        sns.boxplot(x="bedrooms", y="price", data=df, ax=ax3)
        ax3.set_xlabel("Number of Bedrooms")
        ax3.set_ylabel("Price")
        st.pyplot(fig3)

    # Location vs Price
    with col4:
        st.subheader("Location vs Price")
        st.caption("Urban properties are generally more expensive.")

        fig4 = px.box(
            df,
            x="location",
            y="price",
            color="location",
            title="Property Price by Location"
        )
        fig4.update_layout(template="plotly_white")
        st.plotly_chart(fig4, width="stretch")

    # -------- YEAR-WISE TREND --------
    st.markdown("---")
    st.subheader("Year-wise Property Price Growth")
    st.caption("Shows how average property price changes over time.")

    # Check if year column exists
    if "year" in df.columns:

        yearly_price = df.groupby("year")["price"].mean().reset_index()

        fig5 = px.line(
            yearly_price,
            x="year",
            y="price",
            markers=True,
            title="Average Property Price Growth Over Years"
        )

        fig5.update_layout(
            template="plotly_white",
            xaxis_title="Year",
            yaxis_title="Average Price"
        )

        st.plotly_chart(fig5, width="stretch")

    else:
        st.warning("Year column not found in dataset. Year-wise trend cannot be displayed.")

# PRICE PREDICTION
elif selected == "Price Prediction":

    st.title("🏠 House Price Prediction")

    st.write("Enter the property details to estimate the house price.")

    # Numeric inputs
    area = st.number_input("Area (in square feet)", min_value=300, max_value=20000, value=1000)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
    stories = st.number_input("Stories", min_value=1, max_value=4, value=2)
    parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

    # Categorical inputs
    mainroad = st.selectbox("Main Road Access", ["yes","no"])
    guestroom = st.selectbox("Guest Room", ["yes","no"])
    basement = st.selectbox("Basement", ["yes","no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes","no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes","no"])
    location = st.selectbox("Location", ["Rural","Semi-Urban","Urban"])

    # Encoding
    yn = {'yes':1, 'no':0}
    loc = {'Rural':0, 'Semi-Urban':1, 'Urban':2}

    # Create dataframe for model
    input_df = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': yn[mainroad],
        'guestroom': yn[guestroom],
        'basement': yn[basement],
        'hotwaterheating': yn[hotwaterheating],
        'airconditioning': yn[airconditioning],
        'location': loc[location]
    }])

    # Scale input
    input_data = scaler.transform(input_df)

    # Prediction
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        price = int(prediction[0])

        st.success(f"Estimated House Price: ₹ {price:,}")

        # Price category (extra professional touch)
        if price < 5000000:
            st.info("Category: Budget House 🟢")
        elif price < 9000000:
            st.info("Category: Mid-Range House 🟡")
        else:
            st.info("Category: Premium House 🔴")