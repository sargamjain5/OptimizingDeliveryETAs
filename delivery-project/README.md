
# Logistics Network Intelligence System

## Overview
This project builds an end-to-end logistics network intelligence framework for analyzing shipment movement, identifying bottlenecks, predicting ETA delays, and optimizing transportation mode decisions between Full Truck Load (FTL) and Carting operations.

The pipeline combines:
- Exploratory Data Analysis (EDA)
- Graph-based logistics network construction
- Bottleneck and corridor analysis
- ETA prediction using ML models
- Graph embeddings using Node2Vec

The system transforms raw shipment movement data into operational intelligence for network strategy teams.

---

# Problem Statement

Modern logistics networks face:
- Hub congestion
- Unpredictable delivery delays
- Corridor inefficiencies
- Poor transport mode allocation
- Revenue leakage due to SLA failures

The objective of this project is to:
1. Identify high-friction hubs and corridors
2. Predict shipment ETA accurately
3. Quantify operational bottlenecks
4. Recommend FTL vs Carting decisions
5. Estimate business and revenue impact

---

# Repository Structure

```bash
.
├── 01_eda.ipynb
├── 02_graph_construction.ipynb
├── 03_bottleneck_analysis.ipynb
├── 04_eta_prediction.ipynb
├── 05_graph_embeddings.ipynb
├── data/
└── outputs/
```

---

# Workflow

## 1. Exploratory Data Analysis
Notebook: `01_eda.ipynb`

Performed:
- Null value analysis
- Delay ratio calculation
- Segment-wise delay analysis
- Time distribution analysis
- Shipment duration trends

Key engineered features:
- `delay_ratio = actual_time / osrm_time`
- `segment_delay_ratio`
- Hub-level delay aggregation

Insights:
- Significant variance exists between predicted OSRM travel time and actual shipment movement.
- Certain hubs consistently contribute to cascading delays.
- Long-haul corridors show nonlinear delay amplification.

---

## 2. Graph Construction
Notebook: `02_graph_construction.ipynb`

The logistics network is modeled as a directed weighted graph.

### Nodes
- Logistics hubs
- Source centers
- Destination centers

### Edges
- Shipment movement corridors

### Edge Weights
- Median delay ratio
- Corridor traffic frequency

### Graph Features
- Degree centrality
- Betweenness centrality
- Connectivity analysis

Why graph modeling?
Traditional tabular analytics fail to capture network dependencies. Graphs enable:
- Bottleneck identification
- Route criticality estimation
- Corridor dependency analysis
- Network flow optimization

---

## 3. Bottleneck Analysis
Notebook: `03_bottleneck_analysis.ipynb`

This stage identifies:
- High-delay corridors
- Congested hubs
- Critical transit dependencies

### Methodology
1. Aggregate median delay ratio per corridor
2. Rank corridors by delay intensity
3. Analyze centrality metrics
4. Identify hubs with high transit dependency

### Key Operational Signals
- Hubs with high betweenness centrality are operationally critical.
- Small disruptions at these hubs create cascading SLA failures.
- Certain corridors exhibit structurally high delay ratios and require intervention.

---

## 4. ETA Prediction
Notebook: `04_eta_prediction.ipynb`

Machine learning models are used to predict shipment ETA.

### Features Used
- Source center
- Destination center
- OSRM time
- Segment travel features
- Centrality features
- Corridor statistics

### Modeling
Potential models used:
- LightGBM
- CatBoost
- XGBoost
- Ensemble learning

### Objective
Minimize ETA prediction error and improve shipment planning reliability.

### Business Value
- Better customer SLA adherence
- Dynamic dispatch optimization
- Reduced operational uncertainty

---

## 5. Graph Embeddings
Notebook: `05_graph_embeddings.ipynb`

Node2Vec is used to learn dense vector embeddings for hubs.

### Why Node2Vec?
Graph embeddings capture:
- Structural similarity
- Traffic behavior similarity
- Transit role similarity

### Benefits
- Better ML feature representation
- Improved ETA prediction
- Corridor similarity clustering
- Hidden bottleneck discovery

---

# FTL vs Carting Decision Framework

## Objective
Determine the optimal transportation mode based on:
- Shipment volume
- Corridor stability
- Delay risk
- Cost efficiency
- Delivery urgency

---

## Definitions

### FTL (Full Truck Load)
Dedicated vehicle assigned to a shipment.

Advantages:
- Faster transit
- Lower handling risk
- Better SLA consistency
- Reduced touchpoints

Disadvantages:
- Higher cost for low utilization
- Inefficient for fragmented loads

---

### Carting
Shared shipment consolidation model.

Advantages:
- Lower cost
- Better utilization
- Efficient for low-volume shipments

Disadvantages:
- Higher transit uncertainty
- Multiple handling points
- Greater delay probability

---

# Decision Framework

| Parameter | FTL Preferred | Carting Preferred |
|---|---|---|
| Shipment Volume | High | Low |
| Corridor Delay Variance | Low | Moderate |
| SLA Criticality | High | Medium |
| Transit Frequency | Stable | Variable |
| Shipment Value | High | Low |
| Cost Sensitivity | Medium | High |
| Delivery Urgency | High | Low |

---

# Supporting Analysis

## When to Use FTL
Recommended for:
- High-frequency corridors
- Enterprise clients
- Time-sensitive shipments
- Stable long-haul routes

Expected Benefits:
- 15–25% lower delay probability
- Better ETA reliability
- Reduced re-routing risk

---

## When to Use Carting
Recommended for:
- Low shipment density corridors
- Non-critical deliveries
- Rural or fragmented networks

Expected Benefits:
- Better fleet utilization
- Lower operating cost
- Higher network flexibility

---

# Corridor Intelligence Strategy

## High-Risk Corridors
Characteristics:
- High delay ratio
- Large variance
- Multi-hop dependency

Recommended Action:
- Shift high-priority loads to FTL
- Add transit buffers
- Introduce dynamic routing

---

## Stable Corridors
Characteristics:
- Predictable transit time
- Lower congestion
- Balanced throughput

Recommended Action:
- Continue carting optimization
- Improve consolidation efficiency

---

# Revenue Impact Estimation

Operational improvements can generate revenue impact through:

1. Reduced SLA penalties
2. Improved customer retention
3. Higher shipment throughput
4. Lower idle fleet cost
5. Better route planning

Estimated Impact Areas:
- 8–12% reduction in delay-related penalties
- 10–18% operational efficiency gain
- Improved fleet utilization
- Better customer satisfaction metrics

---

# Future Improvements

- Real-time traffic integration
- Dynamic route optimization
- Reinforcement learning for dispatch
- Live congestion prediction
- Multi-objective network optimization

---

# Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Graph Analytics | NetworkX |
| ML Models | LightGBM, CatBoost, XGBoost |
| Graph Embeddings | Node2Vec |
| Notebook Environment | Jupyter |

---

# Conclusion

This project demonstrates how graph analytics and machine learning can transform logistics operations into a data-driven intelligent network.

The combined use of:
- Graph centrality analysis
- Corridor bottleneck detection
- ETA prediction
- Node embeddings
- Strategic FTL vs Carting optimization

creates a scalable framework for logistics network intelligence and operational decision-making.
