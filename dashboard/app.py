import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import requests


st.set_page_config(
    page_title="Delhivery Network Intelligence",
    layout="wide"
)

st.title("🚚 Delhivery Network Intelligence Dashboard")
st.caption("Graph-Based ETA Optimization & Bottleneck Detection System")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("../data/delivery_data.csv")
    except FileNotFoundError:
        # Fallback simulation data for testing structural layout
        np.random.seed(42)
        df = pd.DataFrame({
            "trip_creation_time": pd.date_range(start="2026-01-01", periods=1000, freq="h"),
            "source_center": np.random.choice(["IND001", "IND002", "IND003", "IND004", "IND005"], 1000),
            "destination_center": np.random.choice(["IND001", "IND002", "IND003", "IND004", "IND005"], 1000),
            "osrm_time": np.random.uniform(30, 300, 1000),
            "actual_time": np.random.uniform(35, 450, 1000)
        })
        df = df[df["source_center"] != df["destination_center"]]
    return df

df = load_data()

df["delay_ratio"] = df["actual_time"] / df["osrm_time"]
df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"])
df["hour"] = df["trip_creation_time"].dt.hour

corridor_df = df.groupby(["source_center", "destination_center"])["delay_ratio"].median().reset_index()

G = nx.from_pandas_edgelist(
    corridor_df,
    source="source_center",
    target="destination_center",
    edge_attr="delay_ratio",
    create_using=nx.DiGraph()
)

betweenness = nx.betweenness_centrality(G)
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
clustering = nx.clustering(G.to_undirected())

hub_df = pd.DataFrame({
    "hub": list(betweenness.keys()),
    "betweenness": list(betweenness.values()),
    "in_degree": [in_degree.get(node, 0) for node in betweenness.keys()],
    "out_degree": [out_degree.get(node, 0) for node in betweenness.keys()],
    "clustering": [clustering.get(node, 0) for node in betweenness.keys()]
})

top_bottlenecks = hub_df.sort_values("betweenness", ascending=False)
top_corridors = corridor_df.sort_values("delay_ratio", ascending=False)


# SIDEBAR NAVIGATION
with st.sidebar:
    st.image("https://img.icons8.com/fluent/100/000000/delivery-truck.png", width=70)
    st.title("Navigation")
    section = st.radio(
        "Go To",
        ["Overview", "Bottlenecks", "Corridors", "Network Graph", "Live ETA Prediction", "Raw Data"]
    )


if section == "Overview":
    st.subheader("Network Health KPIs")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Hubs", G.number_of_nodes())
        c2.metric("Total Corridors", G.number_of_edges())
        c3.metric("Avg Delay Ratio", f"{df['delay_ratio'].mean():.2f}x")
        c4.metric("Max Delay Ratio", f"{df['delay_ratio'].max():.2f}x", delta="High Variance", delta_color="inverse")

    st.markdown("### ETA Model Baseline Metrics")
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", "29.62 min")
        m2.metric("RMSE", "63.08 min")
        m3.metric("R² Score", "0.9889")
        m4.metric("15% Bound Accuracy", "63.9%")

    st.markdown("### Delay Distribution Overview")
    fig = px.histogram(
        df, x="delay_ratio", nbins=50,
        labels={'delay_ratio': 'Delay Ratio (Actual / OSRM)'},
        color_discrete_sequence=['#FF4B4B'],
        template="plotly_white"
    )
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=350)
    st.plotly_chart(fig, use_container_width=True)


# BOTTLENECK SECTION
elif section == "Bottlenecks":
    st.subheader("Critical Network Bottlenecks")
    st.caption("Hubs with high betweenness centrality values acting as system chokepoints.")
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.dataframe(
            top_bottlenecks.head(20).style.background_gradient(subset=['betweenness'], cmap='Reds'),
            use_container_width=True, hide_index=True
        )
    with col2:
        fig = px.bar(
            top_bottlenecks.head(10), x="hub", y="betweenness",
            title="Top 10 High Centrality Hubs",
            labels={"betweenness": "Betweenness Centrality", "hub": "Hub ID"},
            color="betweenness", color_continuous_scale="Reds"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# CORRIDOR SECTION
elif section == "Corridors":
    st.subheader("High-Risk Transit Corridors")
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.dataframe(
            top_corridors.head(20).style.background_gradient(subset=['delay_ratio'], cmap='Oranges'),
            use_container_width=True, hide_index=True
        )
    with col2:
        top_corridors_plot = top_corridors.head(10).copy()
        top_corridors_plot["Route"] = top_corridors_plot["source_center"] + " → " + top_corridors_plot["destination_center"]
        fig = px.bar(
            top_corridors_plot, x="Route", y="delay_ratio",
            title="Top 10 High Delay Transits",
            labels={"delay_ratio": "Median Delay Ratio"},
            color="delay_ratio", color_continuous_scale="Oranges"
        )
        st.plotly_chart(fig, use_container_width=True)


# NETWORK GRAPH SECTION
elif section == "Network Graph":
    st.subheader("Logistics Topology Network Map")
    
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Hub: {node}<br>Degree Connections: {G.degree(node)}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(showscale=True, colorscale='YlOrRd', size=10, color=[], line_width=1.5)
    )
    
    node_trace.marker.color = [len(list(G.neighbors(n))) for n in G.nodes()]

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False, hovermode='closest',
                    margin=dict(b=10, l=5, r=5, t=10),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                 )
    st.plotly_chart(fig, use_container_width=True)


# LIVE ETA PREDICTION
elif section == "Live ETA Prediction":
    st.subheader("Pipeline Inference Engine")
    
    with st.form("eta_form"):
        st.markdown("#### Meta Parameters Matrix")
        
        # Matrix Layout
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            osrm_time = st.number_input("OSRM Time (Minutes)", value=120.0)
            source_center = st.text_input("Source Center Hub ID", "IND000000ACB")
        with r1_c2:
            osrm_distance = st.number_input("OSRM Distance (KM)", value=50.0)
            destination_center = st.text_input("Destination Center Hub ID", "IND410504AAA")
        with r1_c3:
            actual_distance = st.number_input("Actual Segment Distance (KM)", value=55.0)

        st.markdown("---")
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            segment_osrm_time = st.number_input("Segment OSRM Time Reference", value=60.0)
            hour = st.slider("Trip Generation Hour", 0, 23, 14)
        with r2_c2:
            segment_osrm_distance = st.number_input("Segment OSRM Distance Reference", value=30.0)
            day_of_week = st.slider("Day Cycle Vector (0=Mon, 6=Sun)", 0, 6, 2)

        submitted = st.form_submit_button("Run Prediction Model", type="primary")

    if submitted:
        payload = {
            "osrm_time": osrm_time,
            "osrm_distance": osrm_distance,
            "actual_distance_to_destination": actual_distance,
            "hour": hour,
            "day_of_week": day_of_week,
            "segment_osrm_time": segment_osrm_time,
            "segment_osrm_distance": segment_osrm_distance,
            "source_center": source_center,
            "destination_center": destination_center
        }

        with st.spinner("Processing structural network maps..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict_eta",
                    json=payload,
                    timeout=5
                )
                
                # Check server error codes defensively before reading JSON
                if response.status_code == 200:
                    prediction = response.json()
                    st.success("🎯 Execution Completed Successfully!")
                    
                    with st.container(border=True):
                        st.metric(
                            label="**Model Predicted Total ETA**", 
                            value=f"{prediction['predicted_eta']} Hours"
                        )
                        res_c1, res_c2 = st.columns(2)
                        res_c1.metric("Calculated Source Centrality", f"{prediction['source_centrality']:.5f}")
                        res_c2.metric("Calculated Destination Centrality", f"{prediction['destination_centrality']:.5f}")
                else:
                    st.error(f"Backend Exception Event (HTTP Status: {response.status_code})")
                    with st.expander("Diagnostic Trace Log"):
                        st.code(response.text, language="html")

            except requests.exceptions.ConnectionError:
                st.error("Interface Connection Fault: Validation Pipeline at `http://127.0.0.1:8000` is offline.")


# RAW DATA 
elif section == "Raw Data":
    st.subheader(" Registry Storage Preview")
    st.dataframe(df.head(100), use_container_width=True)

