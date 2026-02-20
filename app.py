import streamlit as st
from apputil import *

st.title("🚢 Titanic Survival Analysis")

st.header("Exercise 1: Survival Demographics")
st.write("How did survival rates vary across different age groups, and did the 'women and children first' protocol apply equally across all passenger classes?")

st.subheader("Survival Data by Demographics")
st.dataframe(survival_demographics())

st.subheader("Visualization")
st.plotly_chart(visualize_demographic(), use_container_width=True)

st.markdown("---")

st.header("Exercise 2: Family Groups and Fare Analysis")
st.write("Did larger families pay proportionally higher fares, or were there economies of scale in ticket pricing across different passenger classes?")

st.subheader("Family Size and Fare Data")
st.dataframe(family_groups())

st.subheader("Last Name Analysis")
st.write("Top 10 most common last names:")
st.dataframe(last_names().head(10))

st.write("The last name counts align with the family size data. Families like Andersson and Sage represent the largest groups, predominantly in 3rd class, which explains lower average fares for larger families in that class.")

st.subheader("Visualization")
st.plotly_chart(visualize_families(), use_container_width=True)

st.markdown("---")

st.header("Bonus: Age Division Analysis")
st.write("Within each passenger class, did being older or younger than the median age for that class affect survival chances?")

st.subheader("Age Division Data")
st.dataframe(determine_age_division()[['Pclass', 'Age', 'older_passenger', 'Survived']].head(20))

st.subheader("Visualization")
st.plotly_chart(visualize_age_division(), use_container_width=True)

st.write("Younger passengers in 1st and 2nd class had better survival rates, supporting the 'women and children first' protocol. In 3rd class, the pattern is less pronounced.")