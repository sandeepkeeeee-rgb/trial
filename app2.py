#%%
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#%%
# Create Dummy Dataset
np.random.seed(42)

data = pd.DataFrame({
"age": np.random.randint(21, 60, 100),
"tenure": np.random.randint(0, 20, 100),
"experience": np.random.randint(0, 25, 100)
})

# Target variable (salary)
data["salary"] = (
data["age"] * 2000 +
data["tenure"] * 3000 +
data["experience"] * 5000 +
np.random.randint(10000, 50000, 100)
)

# Train Model
X = data[["age", "tenure", "experience"]]
y = data["salary"]

model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Salary Prediction App")

st.write("Use sliders to input employee details")

age = st.slider("Age", 20, 60, 30)
tenure = st.slider("Tenure (Years in Company)", 0, 20, 5)
experience = st.slider("Total Experience (Years)", 0, 25, 5)
#%%
# Prediction
if st.button("Predict Salary"):
    input_data = np.array([[age, tenure, experience]])
    salary = model.predict(input_data)[0]

    st.success(f"Predicted Salary: ₹ {round(salary, 2)}")
# %%
