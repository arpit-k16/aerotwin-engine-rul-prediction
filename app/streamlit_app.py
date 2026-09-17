"""
AeroTwin — Aircraft Engine Health Intelligence Dashboard
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.health_score import RISK_COLORS, generate_maintenance_insight

# Configure Streamlit page
st.set_page_config(
    page_title="AeroTwin | Engine Health",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .metric-label {
        color: #6c757d;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #856404;
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffeeba;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load preprocessed data and predictions."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "reports" / "figures"
    
    # Load test predictions and features
    try:
        preds_df = pd.read_parquet(processed_dir / "test_predictions.parquet")
        feat_df = pd.read_parquet(processed_dir / "test_processed.parquet")
        
        # Merge them
        df = pd.merge(feat_df, preds_df[["engine_id", "cycle", "predicted_rul", "health_score", "risk_category"]], 
                      on=["engine_id", "cycle"], how="left")
        
        # Load SHAP global importance (image path)
        shap_img = reports_dir / "shap_global_importance.png"
        
        return df, shap_img
    except Exception as e:
        st.error(f"Error loading data: {e}. Please run the pipeline first.")
        return None, None

def main():
    st.title("✈️ AeroTwin — Engine Health Intelligence")
    st.markdown("Predictive maintenance dashboard using NASA C-MAPSS telemetry.")
    
    df, shap_img_path = load_data()
    if df is None:
        return
        
    engine_ids = sorted(df["engine_id"].unique())
    
    # Sidebar
    st.sidebar.header("Engine Selection")
    selected_engine = st.sidebar.selectbox("Select Engine ID:", engine_ids)
    
    engine_data = df[df["engine_id"] == selected_engine].copy()
    engine_data = engine_data.sort_values("cycle")
    
    last_cycle_data = engine_data.iloc[-1]
    
    # Top Overview Metrics
    st.header(f"Overview: Engine {selected_engine}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Cycle</div>
            <div class="metric-value">{int(last_cycle_data['cycle'])}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated RUL</div>
            <div class="metric-value">{int(last_cycle_data['predicted_rul'])}</div>
            <div style="color: #6c757d; font-size: 0.9rem;">Remaining Cycles</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        score_color = RISK_COLORS.get(last_cycle_data['risk_category'], "#000000")
        st.markdown(f"""
        <div class="metric-card" style="border-bottom: 4px solid {score_color};">
            <div class="metric-label">Health Score</div>
            <div class="metric-value" style="color: {score_color};">{last_cycle_data['health_score']:.1f}</div>
            <div style="color: #6c757d; font-size: 0.9rem;">out of 100</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value" style="color: {score_color}; font-size: 2rem;">{last_cycle_data['risk_category']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📉 Degradation Trajectory", "🌡️ Sensor Health", "🧠 Explainability & Insights"])
    
    with tab1:
        st.subheader("Remaining Useful Life (RUL) Trajectory")
        
        fig = go.Figure()
        
        # Predicted RUL
        fig.add_trace(go.Scatter(
            x=engine_data["cycle"], 
            y=engine_data["predicted_rul"],
            mode="lines",
            name="Predicted RUL",
            line=dict(color="#FF5722", width=3)
        ))
        
        # Actual RUL if available
        if "rul" in engine_data.columns:
            fig.add_trace(go.Scatter(
                x=engine_data["cycle"], 
                y=engine_data["rul"],
                mode="lines",
                name="Actual RUL (Ground Truth)",
                line=dict(color="#2196F3", width=2, dash="dash")
            ))
            
        fig.update_layout(
            xaxis_title="Operating Cycle",
            yaxis_title="Remaining Cycles",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Key Sensor Trends")
        
        # Select important sensors (just picking a few illustrative ones for the demo)
        available_sensors = [col for col in engine_data.columns if col.startswith("sensor_") and "_rmean_" not in col and "_rstd_" not in col and "_slope_" not in col and "_ewma_" not in col]
        
        if available_sensors:
            sensor_to_plot = st.selectbox("Select Sensor to analyze:", available_sensors, index=0)
            
            fig2 = go.Figure()
            
            # Raw sensor
            fig2.add_trace(go.Scatter(
                x=engine_data["cycle"],
                y=engine_data[sensor_to_plot],
                mode="lines",
                name="Raw Measurement",
                line=dict(color="#BDBDBD", width=1),
                opacity=0.6
            ))
            
            # Rolling mean if available
            rmean_col = f"{sensor_to_plot}_rmean_10"
            if rmean_col in engine_data.columns:
                fig2.add_trace(go.Scatter(
                    x=engine_data["cycle"],
                    y=engine_data[rmean_col],
                    mode="lines",
                    name="Rolling Mean (10 cycles)",
                    line=dict(color="#1976D2", width=2)
                ))
                
            fig2.update_layout(
                title=f"{sensor_to_plot.replace('_', ' ').title()} Trajectory",
                xaxis_title="Operating Cycle",
                yaxis_title="Normalized Value",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
    with tab3:
        col_insight, col_shap = st.columns([1, 1.5])
        
        with col_insight:
            st.subheader("Maintenance Insight")
            
            # Calculate degradation trend
            recent_preds = engine_data["predicted_rul"].values
            # Compute slope over last 5 cycles
            if len(recent_preds) >= 5:
                x = np.arange(5)
                y = recent_preds[-5:]
                trend = np.polyfit(x, y, 1)[0]
            else:
                trend = 0.0
                
            insight = generate_maintenance_insight(
                last_cycle_data["health_score"],
                last_cycle_data["risk_category"],
                last_cycle_data["predicted_rul"],
                trend
            )
            
            st.info(insight.replace("\n\n", "\n\n**Note:** "))
            
            st.markdown("""
            <div class="disclaimer">
            <b>Disclaimer:</b> AeroTwin is a portfolio project demonstrating predictive maintenance ML. 
            The health score and insights are analytical indicators derived from model outputs and 
            do not represent certified aerospace maintenance decisions.
            </div>
            """, unsafe_allow_html=True)
            
        with col_shap:
            st.subheader("Global Feature Importance (SHAP)")
            st.markdown("Top factors driving the model's RUL predictions across the fleet.")
            
            if shap_img_path and shap_img_path.exists():
                st.image(str(shap_img_path), use_container_width=True)
            else:
                st.warning("SHAP summary plot not found. Run the pipeline to generate it.")

if __name__ == "__main__":
    main()
