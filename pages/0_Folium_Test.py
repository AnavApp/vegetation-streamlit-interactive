# import folium
# import streamlit as st

# from streamlit_folium import st_folium

# # center on Liberty Bell, add marker
# m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
# folium.Marker(
#     [39.949610, -75.150282], popup="Liberty Bell", tooltip="Liberty Bell"
# ).add_to(m)

# # call to render Folium map in Streamlit
# st_data = st_folium(m, width=725)


### DRAW SUPPORT:
import folium
import streamlit as st
from folium.plugins import Draw

from streamlit_folium import st_folium

m = folium.Map(location=[39.949610, -75.150282], zoom_start=5)
Draw(export=True).add_to(m)

c1, c2 = st.columns(2)
with c1:
    output = st_folium(m, width=700, height=500)

with c2:
    st.write(output)