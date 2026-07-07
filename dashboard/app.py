import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.simulation import TournamentSimulator

# Set page config
st.set_page_config(
    page_title="2026 FIFA World Cup Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    .main {
        background-color: #0f111a;
        color: #e0e6ed;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1d2e;
        border-radius: 4px;
        color: #a0aec0;
        padding-left: 16px;
        padding-right: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #3b82f6;
    }
    .group-card {
        background-color: #1a1d2e;
        padding: 16px;
        border-radius: 8px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_simulator():
    # Cache simulator instance to avoid repeating precomputation
    return TournamentSimulator()

# Initialize simulator
try:
    simulator = get_simulator()
except Exception as e:
    st.error(f"Failed to initialize simulator: {e}. Please ensure features are generated and model is trained.")
    st.stop()

# Sidebar controls
st.sidebar.title("🏆 WC 2026 Control Center")
st.sidebar.markdown("Configure and run the Monte Carlo simulation engine.")

n_sims = st.sidebar.slider("Number of Simulations", min_value=100, max_value=5000, value=1000, step=100)

run_sim = st.sidebar.button("🚀 Run Monte Carlo Engine", use_container_width=True)

# App header
st.title("🏆 2026 FIFA World Cup Predictor")
st.markdown("An interactive simulator leveraging machine learning, dynamic Elo ratings, and chronological team form variables.")

# Create tabs
tab_sim, tab_match, tab_groups = st.tabs([
    "📊 Tournament Simulations", 
    "🆚 Custom Match Predictor", 
    "📅 Groups & Standings"
])

# Tab 1: Tournament Simulations
with tab_sim:
    # Run simulation logic
    if run_sim or "sim_results" not in st.session_state:
        if run_sim:
            with st.spinner(f"Simulating {n_sims} World Cups..."):
                # Temporarily override sims
                old_sims = simulator.n_simulations
                simulator.n_simulations = n_sims
                results = simulator.run_monte_carlo()
                simulator.n_simulations = old_sims
                
                # Store in session state
                st.session_state["sim_results"] = results
                st.session_state["n_sims_run"] = n_sims
        else:
            # Run quick default on first load
            if "sim_results" not in st.session_state:
                with st.spinner("Initializing first run (1000 simulations)..."):
                    results = simulator.run_monte_carlo()
                    st.session_state["sim_results"] = results
                    st.session_state["n_sims_run"] = 1000

    results = st.session_state["sim_results"]
    n_sims_run = st.session_state["n_sims_run"]
    
    st.header(f"📈 Simulation Results ({n_sims_run} iterations)")
    
    # Process results into dataframe
    df_champs = pd.DataFrame(results["champions_probs"], columns=["Team", "Probability"])
    df_champs["Probability (%)"] = df_champs["Probability"] * 100
    
    # Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    best_team = df_champs.iloc[0]["Team"]
    best_prob = df_champs.iloc[0]["Probability (%)"]
    
    col_m1.metric("Tournament Favorite", best_team, f"{best_prob:.1f}% Win Probability")
    
    # Count unique champions
    unique_champs = len(df_champs)
    col_m2.metric("Unique Winners", unique_champs, "out of 48 teams")
    
    # Dark horse metric
    dark_horses = df_champs[df_champs["Probability (%)"] < 2.0]
    dark_horse_best = dark_horses.iloc[0]["Team"] if not dark_horses.empty else "N/A"
    col_m3.metric("Top Dark Horse Winner", dark_horse_best)
    
    # Champion Probability chart
    st.subheader("🥇 Top 10 World Cup Contenders")
    fig_champs = px.bar(
        df_champs.head(10),
        x="Probability (%)",
        y="Team",
        orientation="h",
        color="Probability (%)",
        color_continuous_scale="Viridis",
        labels={"Team": "Country", "Probability (%)": "Championship Odds (%)"}
    )
    fig_champs.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e6ed",
        yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig_champs, use_container_width=True)
    
    # Full Odds Table
    st.subheader("📋 All Team Championship Odds")
    st.dataframe(
        df_champs.rename(columns={"Probability": "Probability (Decimal)"}),
        column_config={
            "Team": "Country Name",
            "Probability (Decimal)": st.column_config.NumberColumn(format="%.4f"),
            "Probability (%)": st.column_config.ProgressColumn(
                "Championship Odds",
                help="Probability of winning the World Cup",
                format="%.2f%%",
                min_value=0.0,
                max_value=30.0
            )
        },
        use_container_width=True,
        hide_index=True
    )

# Tab 2: Custom Match Predictor
with tab_match:
    st.header("🆚 Custom Match Predictor")
    st.markdown("Predict the probabilities and simulate a score for any match configuration.")
    
    all_teams = sorted(list(simulator.team_states.keys()))
    
    col_t1, col_vs, col_t2 = st.columns([5, 1, 5])
    
    with col_t1:
        team_a = st.selectbox("Select Team A", all_teams, index=all_teams.index("Argentina") if "Argentina" in all_teams else 0)
        state_a = simulator.team_states[team_a]
        st.markdown(f"**FIFA Rank:** {int(state_a['rank'])} | **Elo Rating:** {int(state_a['elo'])}")
        
    with col_vs:
        st.markdown("<h3 style='text-align: center; margin-top: 25px;'>VS</h3>", unsafe_allow_html=True)
        
    with col_t2:
        team_b = st.selectbox("Select Team B", all_teams, index=all_teams.index("France") if "France" in all_teams else 1)
        state_b = simulator.team_states[team_b]
        st.markdown(f"**FIFA Rank:** {int(state_b['rank'])} | **Elo Rating:** {int(state_b['elo'])}")
        
    neutral = st.checkbox("Neutral Venue", value=True)
    
    # Calculate probabilities
    p_a, p_draw, p_b = simulator.predict_match_probabilities(team_a, team_b, neutral)
    
    # Probability Chart
    st.subheader("📊 Win-Draw-Loss Distribution")
    probs_df = pd.DataFrame({
        "Outcome": [f"{team_a} Win", "Draw", f"{team_b} Win"],
        "Probability (%)": [p_a * 100, p_draw * 100, p_b * 100]
    })
    
    fig_match = px.bar(
        probs_df,
        x="Outcome",
        y="Probability (%)",
        color="Outcome",
        color_discrete_map={
            f"{team_a} Win": "#10b981", # Green
            "Draw": "#6b7280",          # Gray
            f"{team_b} Win": "#3b82f6"  # Blue
        },
        text=probs_df["Probability (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig_match.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e6ed",
        showlegend=False
    )
    st.plotly_chart(fig_match, use_container_width=True)
    
    # Match Simulator button
    st.subheader("⚽ Live Match Simulator")
    if st.button("🎲 Simulate Score", use_container_width=True):
        winner, goals_a, goals_b = simulator.simulate_match_outcome(team_a, team_b, is_knockout=True, neutral=neutral)
        
        st.markdown(
            f"<h2 style='text-align: center; color: #3b82f6;'>"
            f"{team_a} {goals_a} - {goals_b} {team_b}"
            f"</h2>",
            unsafe_allow_html=True
        )
        
        if goals_a == goals_b:
            st.markdown(f"<p style='text-align: center;'>Tied in 90 mins. **{winner}** wins on penalties!</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='text-align: center;'>**{winner}** wins in regular time!</p>", unsafe_allow_html=True)

# Tab 3: Groups & Standings
with tab_groups:
    st.header("📅 2026 FIFA World Cup Groups")
    
    # 4 columns for groups layout
    g_cols = st.columns(3)
    
    groups_list = list(simulator.groups_2026.keys())
    
    for idx, g in enumerate(groups_list):
        col = g_cols[idx % 3]
        with col:
            st.markdown(
                f"<div class='group-card'>"
                f"<h4>Group {g}</h4>"
                f"<ol>"
                f"<li><b>{simulator.groups_2026[g][0]}</b></li>"
                f"<li><b>{simulator.groups_2026[g][1]}</b></li>"
                f"<li>{simulator.groups_2026[g][2]}</li>"
                f"<li>{simulator.groups_2026[g][3]}</li>"
                f"</ol>"
                f"</div>",
                unsafe_allow_html=True
            )
            
    # [MOCK GROUP STANDINGS VIEW FOR REFERENCE - cmt out]
    # # Render placeholder text
    # st.markdown("---")
    # st.markdown("### 📝 Alternative Group Stage View")
    # # if st.button("Generate single group table"):
    # #     st.dataframe(pd.DataFrame(simulator.simulate_group_stage()["A"]))
