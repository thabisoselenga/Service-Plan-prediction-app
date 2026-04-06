import streamlit as st
import pandas as pd
import joblib

model = joblib.load("serviceplanmodel.pkl")

def price_plan(vehicle_brand,vehicle_type,vehicle_age,mileage):
    input_data = pd.DataFrame([{
        "vehicle_brand": vehicle_brand,
        "vehicle_type": vehicle_type,
        "vehicle_age": vehicle_age,
        "mileage": mileage
    }])

    expected_cost = model.predict(input_data)[0]
    risk_margin = expected_cost*0.25
    admin_cost = 2000
    price = expected_cost+risk_margin+admin_cost

    return{
        "expected_cost": expected_cost,
        "price":price
    }

st.title ("Service plan pricing app")

st.markdown("Please enter the following below and use the button to calculate the price")
st.divider()

brand = st.selectbox("vehicle_brand",["Toyota","VW","Suzuki","BMW","Hyundai","Kia"])
vehicletype = st.selectbox("vehicle_type",["SUV","Sedan","Hatchback"])
age = st.slider("vehicle_age",1,10)
mileage = st.slider("mileage",5000,100000)

st.divider()

if st.button("Calculate Plan Price"):
    result = price_plan(brand,vehicletype,age,mileage)


    st.metric("Expected Repair Cost", f"R {result['expected_cost']:,.2f}")
    st.metric("Recommended Plan Price", f"R {result['price']:,.2f}")