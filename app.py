import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

MODEL_PATH = "pollution_model.pkl"
COLUMNS_PATH = "model_columns.pkl"
DATA_PATH = "PB_All_2000_2021.csv"

TARGETS = ["O2", "NO3", "NO2", "SO4", "PO4", "CL"]


st.set_page_config(
    page_title="Water Quality Intelligence System",
    page_icon="💧",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #0b1f2a, #102a43);
    color: white;
}
.main-title {
    font-size: 46px;
    font-weight: 800;
    color: #7dd3fc;
}
.sub-title {
    font-size: 18px;
    color: #cbd5e1;
}
.metric-card {
    padding: 20px;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(125, 211, 252, 0.25);
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.good {
    color: #22c55e;
    font-weight: 800;
}
.moderate {
    color: #facc15;
    font-weight: 800;
}
.poor {
    color: #ef4444;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def get_risk_level(pred):
    o2 = pred["O2"]
    no2 = pred["NO2"]
    so4 = pred["SO4"]
    cl = pred["CL"]

    score = 0

    if o2 < 4:
        score += 2
    elif o2 < 6:
        score += 1

    if no2 > 10:
        score += 2
    elif no2 > 5:
        score += 1

    if so4 > 500:
        score += 1

    if cl > 500:
        score += 1

    if score <= 1:
        return "Good", "good"
    elif score <= 3:
        return "Moderate", "moderate"
    else:
        return "Poor / Polluted", "poor"
def get_recommendation(risk):

    if risk == "Good":
        return "Water quality appears stable. Continue regular monitoring."

    elif risk == "Moderate":
        return "Moderate pollution risk detected. Further laboratory testing is recommended."

    else:
        return "High pollution risk detected. Immediate inspection and pollution source analysis is recommended."

def calculate_wqi(pred):

    score = 0

    score += pred["NO3"] * 0.2
    score += pred["NO2"] * 0.2
    score += pred["SO4"] * 0.15
    score += pred["PO4"] * 0.15
    score += pred["CL"] * 0.15

    # oxygen inverse effect
    score += max(0, 20 - pred["O2"]) * 0.15

    wqi = min(100, score)

    return round(wqi, 2)

try:
    model, model_columns = load_model()
    data = load_data()

    try:
        model_comparison = joblib.load("model_comparison.pkl")
    except:
        model_comparison = None

    try:
        feature_importance = joblib.load("feature_importance.pkl")
    except:
        feature_importance = None

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


st.markdown("""
<div style="
padding:25px;
border-radius:20px;
background: linear-gradient(135deg,#0f172a,#1e293b,#0c4a6e);
box-shadow:0 8px 30px rgba(0,0,0,0.4);
margin-bottom:20px;
">
<h1 style="color:#7dd3fc;font-size:48px;">
💧 Water Quality Intelligence System
</h1>

<p style="color:#cbd5e1;font-size:18px;">
AI-powered multi-pollutant prediction and intelligent water quality monitoring platform.
</p>
</div>
""", unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict multiple water pollutants and estimate pollution risk using machine learning.</div>',
    unsafe_allow_html=True
)

st.divider()

with st.sidebar:
    st.header("⚙️ Input Parameters")

    years = sorted(data["year"].dropna().astype(int).unique())
    stations = sorted(data["id"].dropna().astype(int).unique())

    selected_year = st.selectbox("Select Year", years, index=len(years) - 1)
    selected_month = st.selectbox("Select Month", list(range(1, 13)))
    selected_station = st.selectbox("Select Station ID", stations)

    st.subheader("🧪 Water Sample Measurements")
    nh4 = st.number_input(
    "NH4 (Ammonium Concentration)",
    min_value=0.0,
    value=0.50,
    step=0.01
    )
    bsk5 = st.number_input(
        "BSK5 / BOD5 (Organic Pollution Level)",       
        min_value=0.0,
        value=3.00,
        step=0.01
    )
    suspended = st.number_input(
        "Suspended Solids",
        min_value=0.0,
        value=20.00,
        step=0.01
    )

    predict_btn = st.button("🔮 Predict Water Quality", use_container_width=True)


colA, colB, colC = st.columns(3)

with colA:
    st.metric("Dataset Rows", len(data))

with colB:
    st.metric("Stations", data["id"].nunique())

with colC:
    st.metric("Year Range", f"{int(data['year'].min())} - {int(data['year'].max())}")


st.divider()

if predict_btn or "last_result" in st.session_state:

    input_df = pd.DataFrame([{
        "id": selected_station,
        "year": selected_year,
        "month": selected_month,
        "NH4": nh4,
        "BSK5": bsk5,
        "Suspended": suspended
    }])

    input_df = input_df[model_columns]

    prediction = model.predict(input_df)[0]

    result = pd.DataFrame({
        "Pollutant": TARGETS,
        "Predicted Value": prediction
    })
    st.session_state["last_result"] = result
    st.session_state["last_pred_dict"] = dict(zip(TARGETS, prediction))

    pred_dict = dict(zip(TARGETS, prediction))
    risk, risk_class = get_risk_level(pred_dict)
    recommendation = get_recommendation(risk)
    wqi_score = calculate_wqi(pred_dict)

    st.subheader("📌 Prediction Summary")

    st.markdown(
        f"""
        <div class="metric-card">
            <h3>Overall Water Quality Status:
            <span class="{risk_class}">{risk}</span></h3>
            <p>{recommendation}</p>        
            </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
    f"""
    <div class="metric-card">
        <h3>🌊 Water Quality Index (WQI)</h3>
        <h1>{wqi_score}</h1>
    </div>
    """,
    unsafe_allow_html=True
    )

    if wqi_score <= 25:
        st.success("Excellent Water Quality")

    elif wqi_score <= 50:
        st.info("Good Water Quality")

    elif wqi_score <= 75:
        st.warning("Poor Water Quality")

    else:
        st.error("Hazardous Water Quality")

    st.write("")

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    cards = [c1, c2, c3, c4, c5, c6]

    for i, pollutant in enumerate(TARGETS):
        with cards[i]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>{pollutant}</h4>
                    <h2>{pred_dict[pollutant]:.2f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.subheader("📊 Predicted Pollutant Comparison")

    fig = px.bar(
        result,
        x="Pollutant",
        y="Predicted Value",
        color="Predicted Value",
        text_auto=".2f",
        title="Predicted Pollutant Levels"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        title_x=0.3
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌊 Water Quality Risk Gauge")

    risk_score = wqi_score

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,

        title={'text': "Pollution Risk Level"},

        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#38bdf8"},

            'steps': [
                {'range': [0, 35], 'color': "#22c55e"},
                {'range': [35, 70], 'color': "#facc15"},
                {'range': [70, 100], 'color': "#ef4444"}
            ]
        }
    ))

    gauge_fig.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(gauge_fig, use_container_width=True)
    st.subheader("📈 Historical Trend Analysis")

    trend_pollutant = st.selectbox(
        "Select pollutant for historical trend",
        TARGETS,
        key="trend_pollutant"
    )

    station_data = data[
    (data["id"] == selected_station) &
    (data[trend_pollutant].notna())
    ].copy()

    if not station_data.empty:
        yearly_trend = (
        station_data.groupby("year")[trend_pollutant]
        .mean()
        .reset_index()
        .sort_values("year")
    )

        trend_fig = px.line(
            yearly_trend,
            x="year",
            y=trend_pollutant,
            markers=True,
            title=f"{trend_pollutant} Trend for Station {selected_station}"
        )

        trend_fig.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Year",
            yaxis_title=f"{trend_pollutant} Level"
        )

        st.plotly_chart(trend_fig, use_container_width=True)

    else:
        st.warning("No historical data available for this station.")

    st.subheader("📄 Prediction Table")
    st.dataframe(result, use_container_width=True)

    csv = result.to_csv(index=False).encode("utf-8")

    download_col, empty_col = st.columns([1, 2])

    with download_col:
        st.download_button(
            label="⬇️ Download Prediction Report as CSV",
            data=csv,
            file_name="water_quality_prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("Enter values from the sidebar and click Predict Water Quality.")

st.divider()

with st.expander("📚 Dataset Preview"):
    st.dataframe(data.head(20), use_container_width=True)

st.markdown("## 🤖 Machine Learning Performance Analysis")
if model_comparison is not None:
    st.dataframe(model_comparison, use_container_width=True)

    fig_model = px.bar(
        model_comparison,
        x="Model",
        y="R2 Score",
        text_auto=".3f",
        title="R² Score Comparison of Machine Learning Models"
    )

    fig_model.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(fig_model, use_container_width=True)
    best_model_name = model_comparison.iloc[0]["Model"]
    best_r2 = model_comparison.iloc[0]["R2 Score"]

    st.success(
        f"Best performing model: {best_model_name} with R² Score = {best_r2:.3f}"
    )

else:
    st.info("Model comparison data not available. Please run the model comparison notebook cell first.")

st.markdown("## 📊 Feature Importance Analysis")

if feature_importance is not None:

    fig_feature = px.bar(
        feature_importance,
        x="Feature",
        y="Importance",
        color="Importance",
        text_auto=".3f",
        title="Most Influential Water Quality Parameters"
    )

    fig_feature.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig_feature, use_container_width=True)

    st.dataframe(feature_importance, use_container_width=True)

# ADD HEATMAP HERE

st.markdown("## 🌡 Pollution Correlation Heatmap")

heatmap_columns = [
    "NH4",
    "BSK5",
    "Suspended",
    "O2",
    "NO3",
    "NO2",
    "SO4",
    "PO4",
    "CL"
]

heatmap_data = data[heatmap_columns].corr()

fig_heatmap = px.imshow(
    heatmap_data,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Correlation Between Water Quality Parameters"
)

fig_heatmap.update_layout(
    template="plotly_dark",
    height=700
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("## 🏭 Multi-Station Pollution Comparison")

comparison_pollutant = st.selectbox(
    "Select pollutant for station comparison",
    TARGETS,
    key="comparison_pollutant"
)

selected_stations_compare = st.multiselect(
    "Select stations to compare",
    sorted(data["id"].dropna().astype(int).unique()),
    default=sorted(data["id"].dropna().astype(int).unique())[:5]
)

comparison_data = data[
    (data["id"].isin(selected_stations_compare)) &
    (data[comparison_pollutant].notna())
].copy()

if not comparison_data.empty:
    station_comparison = (
        comparison_data.groupby("id")[comparison_pollutant]
        .mean()
        .reset_index()
        .sort_values(by=comparison_pollutant, ascending=False)
    )

    fig_station = px.bar(
        station_comparison,
        x="id",
        y=comparison_pollutant,
        color=comparison_pollutant,
        text_auto=".2f",
        title=f"Average {comparison_pollutant} Level Across Selected Stations"
    )

    fig_station.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Station ID",
        yaxis_title=f"Average {comparison_pollutant}"
    )

    st.plotly_chart(fig_station, use_container_width=True)

    st.dataframe(station_comparison, use_container_width=True)

else:
    st.warning("No data available for selected stations.")


st.markdown("## 🏆 Pollution Ranking System")

ranking_pollutant = st.selectbox(
    "Select pollutant for ranking",
    TARGETS,
    key="ranking_pollutant"
)

ranking_type = st.radio(
    "Ranking Type",
    ["Top Polluted Stations", "Top Polluted Years"],
    horizontal=True
)

ranking_data = data[data[ranking_pollutant].notna()].copy()

if ranking_type == "Top Polluted Stations":

    station_rank = (
        ranking_data.groupby("id")[ranking_pollutant]
        .mean()
        .reset_index()
        .sort_values(by=ranking_pollutant, ascending=False)
        .head(10)
    )

    fig_rank = px.bar(
        station_rank,
        x="id",
        y=ranking_pollutant,
        color=ranking_pollutant,
        text_auto=".2f",
        title=f"Top 10 Polluted Stations by {ranking_pollutant}"
    )

    fig_rank.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Station ID",
        yaxis_title=f"Average {ranking_pollutant}"
    )

    st.plotly_chart(fig_rank, use_container_width=True)
    st.dataframe(station_rank, use_container_width=True)

else:

    year_rank = (
        ranking_data.groupby("year")[ranking_pollutant]
        .mean()
        .reset_index()
        .sort_values(by=ranking_pollutant, ascending=False)
        .head(10)
    )

    fig_year_rank = px.bar(
        year_rank,
        x="year",
        y=ranking_pollutant,
        color=ranking_pollutant,
        text_auto=".2f",
        title=f"Top Polluted Years by {ranking_pollutant}"
    )

    fig_year_rank.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Year",
        yaxis_title=f"Average {ranking_pollutant}"
    )

    st.plotly_chart(fig_year_rank, use_container_width=True)
    st.dataframe(year_rank, use_container_width=True)


st.markdown("## 🔮 Future Pollution Forecasting (2022–2030)")

forecast_pollutant = st.selectbox(
    "Select pollutant for forecasting",
    TARGETS,
    key="forecast_pollutant"
)

forecast_station = st.selectbox(
    "Select station for forecasting",
    sorted(data["id"].dropna().astype(int).unique()),
    key="forecast_station"
)

forecast_data = data[
    (data["id"] == forecast_station) &
    (data[forecast_pollutant].notna())
].copy()

if not forecast_data.empty:

    yearly_forecast_data = (
        forecast_data.groupby("year")[forecast_pollutant]
        .mean()
        .reset_index()
        .sort_values("year")
    )

    from sklearn.linear_model import LinearRegression
    import numpy as np

    X_forecast = yearly_forecast_data[["year"]]
    y_forecast = yearly_forecast_data[forecast_pollutant]

    forecast_model = LinearRegression()
    forecast_model.fit(X_forecast, y_forecast)

    future_years = pd.DataFrame({
        "year": list(range(2022, 2031))
    })

    future_predictions = forecast_model.predict(future_years[["year"]])

    future_df = pd.DataFrame({
        "year": future_years["year"],
        forecast_pollutant: future_predictions,
        "Type": "Forecast"
    })

    historical_df = yearly_forecast_data.copy()
    historical_df["Type"] = "Historical"

    combined_df = pd.concat([historical_df, future_df], ignore_index=True)

    fig_forecast = px.line(
        combined_df,
        x="year",
        y=forecast_pollutant,
        color="Type",
        markers=True,
        title=f"{forecast_pollutant} Forecast for Station {forecast_station} (2022–2030)"
    )

    fig_forecast.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Year",
        yaxis_title=f"{forecast_pollutant} Level"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    st.dataframe(future_df, use_container_width=True)

else:
    st.warning("No historical data available for forecasting.")

# THEN ROADMAP
st.markdown("## 🚀 System Roadmap")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>📈 Advanced Analytics</h3>
        <hr style="border:1px solid #1e293b;">
        <p>✔ Multi-model performance comparison</p>
        <p>✔ Feature importance interpretation</p>
        <p>✔ Time-series pollutant forecasting</p>
        <p>✔ AI-driven pollution risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>🌍 Intelligent Monitoring</h3>
        <hr style="border:1px solid #1e293b;">
        <p>✔ Station-wise trend visualization</p>
        <p>✔ Water Quality Index (WQI) computation</p>
        <p>✔ Interactive geo-spatial pollution mapping</p>
        <p>✔ Research-grade reporting & visualizations</p>
    </div>
    """, unsafe_allow_html=True)
