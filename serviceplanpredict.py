import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

model = joblib.load("serviceplanmodel.pkl")

def price_plan(vehicle_brand,vehicle_type,vehicle_age,mileage):
    input_data = pd.DataFrame([{
        "vehicle_brand": vehicle_brand,
        "vehicle_type": vehicle_type,
        "vehicle_age": vehicle_age,
        "mileage": mileage
    }])

    expected_cost = model.predict(input_data)[0][0]
    risk_margin = expected_cost*0.25
    admin_cost = 2000
    price = expected_cost+risk_margin+admin_cost
    profit = price - expected_cost

    return{
        "expected_cost": expected_cost,
        "price":price,
        "profit":profit
    }

st.set_page_config(page_title="Service Plan Pricing engine",layout="wide")
st.title ("Service Plan Pricing Engine")
st.markdown("AI-powered pricing for vehicle service plans")

st.divider()

col1,col2 =st.columns(2)
with col1:
    brand = st.selectbox("vehicle_brand",["Toyota","VW","Suzuki","BMW","Hyundai","Kia"])
    vehicletype = st.selectbox("vehicle_type",["SUV","Sedan","Hatchback"])
with col2:
    age = st.slider("vehicle_age",1,10)
    mileage = st.slider("mileage",5000,100000)

st.divider()

if st.button("Calculate Plan Price"):
    result = price_plan(brand,vehicletype,age,mileage)
    st.divider()
    col1,col2,col3=st.columns(3)

    with col1:
        st.metric("Expected Repair Cost", f"R {result['expected_cost']:,.2f}")
    with col2:
        st.metric("Recommended Plan Price", f"R {result['price']:,.2f}")
    with col3:
        st.metric("Profit",f"R {result['profit']:,.2f}")
    st.divider()
    
    profit = result["price"] - result["expected_cost"]

    fig_profit = go.Figure(go.Indicator(
        mode="gauge+number",
        value=profit,
        title={'text': "Profit Gauge"},
        gauge={
            'axis': {'range': [None, 10000]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 3000], 'color': "lightgray"},
                {'range': [3000, 7000], 'color': "yellow"},
                {'range': [7000, 10000], 'color': "lightgreen"}
            ],
        }
    ))

    
    
    age_range = np.arange(1,11)
    mileage_range = np.arange(5000,100001,10000)

    risk_matrix = []

    for a in age_range:
        row = []
        for m in mileage_range:
            pred = model.predict(pd.DataFrame([{
                "vehicle_brand": brand,
                "vehicle_type": vehicletype,
                "vehicle_age": a,
                "mileage": m
            }]))[0][0]
            row.append(pred)
        risk_matrix.append(row)

    risk_matrix = pd.DataFrame(risk_matrix, index=age_range, columns=mileage_range)
    risk_normalized = (risk_matrix - risk_matrix.min().min()) / (risk_matrix.max().max() - risk_matrix.min().min())
    fig, ax = plt.subplots(figsize=(3,2))
    sns.heatmap(risk_normalized, cmap="Reds",
                linewidths=0.3,
                linecolor="white",
                annot=True,
                fmt=".2f",
                annot_kws={"size":4},
                cbar=False
                )

    ax.set_xlabel("Mileage",fontsize=6)
    ax.set_ylabel("Vehicle Age",fontsize=6)
    ax.set_title("Repair Cost Risk Heatmap",fontsize=6)
    ax.set_xticklabels([f"{int(x/1000)}k" for x in mileage_range], rotation=45)
    ax.tick_params(axis='both', labelsize=6)

    col1, col2 = st.columns([1,1.3])
    with col1:
        st.plotly_chart(fig_profit, use_container_width=True)
    with col2:
        plt.tight_layout()
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
        st.pyplot(fig)
