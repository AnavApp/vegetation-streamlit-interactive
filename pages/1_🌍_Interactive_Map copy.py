import ee
import random
import numpy as np
import os
import streamlit as st
import leafmap.foliumap as leafmap
import requests
import datetime
import pickle
from datetime import date
import pandas as pd
import plotly.express as px
import time
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from datetime import datetime
ee.Authenticate()
ee.Initialize(project='vegetation-2023-408901')


print('page got reloaded')

@st.cache_resource
def load_model(path):
    with open (path, 'rb') as loaded_model:
        model = pickle.load(loaded_model)
    return model

def get_steps(steps):
  benchmark_year=2023
  today=str(date.today())
  year=today.split('-')[0]

  month=today.split('-')[1]
  gap=int(year)-benchmark_year
  gap_month=gap*12 + int(month)
  total_steps=gap_month+steps
  return total_steps

def generate_timestamps(num_steps):
    months = 12
    year = 2023
    month_ = 1

    dates = []

    for num in range(num_steps):
        year_change = num // 12
        year_ = year + year_change

        if num %12 == 0:
            month_ = 1

        date = f"{year_}-{month_}-01"
        dates.append(date)

        month_ += 1
    return dates

def predict(model, num_steps):
    prediction=model.forecast(num_steps)
    timestamp=generate_timestamps(num_steps)
    data_dict={
        'predictions':list(prediction.values),
        'real_values': np.random.random(num_steps),
        'timesteps': timestamp,
    }
    
    df=pd.DataFrame(data_dict)
    fig=px.line(df, x="timesteps", y=df.columns[0:2], title='Predicted Mean NDVI value for the next 24 months', width=700, height=500)

    st.plotly_chart(fig)
    #st.plotly_chart()

def center():
    print('Made it to center()')
    if st.session_state['valleySelect']=='Sacramento': # Sacramento Valley
        print('Made it to if')
        #st.session_state['m'].set_center(-120.283654, 37.011918, 10)
        #m.set_center(-120.283654, 37.011918, 10)
    else:
        print('Made it to else')
        #st.session_state['m'].set_center(-122.011105, 39.129029, 20)
        m.set_center(139.745438, 35.658581, 24)
        #print('st.session_state[\'m\']:',st.session_state['m'])
    st.session_state['m'].to_streamlit(height=700)
CENTER_START = [37.011918, -120.283654] 
ZOOM_START = 10


def initialize_session_state():
    if "center" not in st.session_state:
        st.session_state["center"] = CENTER_START
    if "zoom" not in st.session_state:
        st.session_state["zoom"] = ZOOM_START
    if "markers" not in st.session_state:
        st.session_state["markers"] = []
    if "map_data" not in st.session_state:
        st.session_state["map_data"] = {}
    if "all_drawings" not in st.session_state["map_data"]:
        st.session_state["map_data"]["all_drawings"] = None
    if "upload_file_button" not in st.session_state:
        st.session_state["upload_file_button"] = False

#@title { vertical-output: true}
def collect(region,desc,fold):
  #print("Region:", region.getInfo())
  print("Description:", desc)
  print("Folder:", fold)
  roi = region


  def printType(prompt, object):
    print(prompt, type(object))
  def format_date(timestamp):
      """
      Convert the UTC timestamps to date time

      @parameters
      timestamp: UTC timestamps in milliseconds

      @return
      None
      """
      # get the seconds by dividing 1000
      #print(timestamp)
      timestamp = timestamp/1000
      # Convert the UTC timestamp to a datetime object
      datetime_object = datetime.utcfromtimestamp(timestamp)
      # Format the datetime object as a string (optional)
      formatted_datetime = datetime_object.strftime("%Y-%m-%d %H:%M:%S UTC")
      #print("Formatted Datetime:", formatted_datetime)
      return formatted_datetime
  def print_dict(dictionary):
    for k, v in dictionary.items():
      print(k, v)
  def mapFunctionNDVI(specific_image):
      Red = specific_image.select('B4')
      NIR = specific_image.select('B8')

      ndviPalette = ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
                  '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
                  '012E01', '011D01', '011301']

      NDVI_temp = specific_image
      NDVI = NDVI_temp.addBands(((NIR.subtract(Red)).divide(NIR.add(Red))).rename('NDVI'))  #ee.Image
      #nameOfBands = NDVI.bandNames().getInfo()
      #nameOfBands.remove("B2")
      #print(nameOfBands) # Check if everything in order

      NDVI = NDVI.select('NDVI') # Select all bands except the one you wanna remove
      #NDVI.copyProperties(specific_image)
      return NDVI
  def calculateNDVIStatsForImage(ndvi_image):
    #image = sentinel2ImageCollection.first()
    #print('Image type is :', type(ndvi_image))

    reducers = ee.Reducer.min() \
    .combine(
      ee.Reducer.max(),
      sharedInputs = True
    ).combine(
      ee.Reducer.mean(),
      sharedInputs = True
    ).combine(
      ee.Reducer.stdDev(),
      sharedInputs = True
    )

    multi_stats = ndvi_image.reduceRegion(
        reducer=reducers,
        geometry=roi,
        scale=30,
        crs='EPSG:32610'
    )

    return ndvi_image.set('stats', multi_stats.values())
  def calculateNDVIStatsForImageAsDictionary(ndvi_image):
    #image = sentinel2ImageCollection.first()
    #print('Image type is :', type(ndvi_image))

    reducers = ee.Reducer.min() \
    .combine(
      ee.Reducer.max(),
      sharedInputs = True
    ).combine(
      ee.Reducer.mean(),
      sharedInputs = True
    ).combine(
      ee.Reducer.stdDev(),
      sharedInputs = True
    )

    multi_stats = ndvi_image.reduceRegion(
        reducer=reducers,
        geometry=roi,
        scale=30,
        crs='EPSG:32610'
    )
    #dateStart = format_date(ndvi_image.get('system:time_start').getInfo())

    #multi_stats.set('dateStart','something')
    return ndvi_image.set('stats_dictionary', multi_stats)


  WANTED_BANDS = ['B2', 'B3', 'B4', 'B8']

  print('ROI:', roi)
  sentinel2ImageCollection = (
      ee.ImageCollection('COPERNICUS/S2')
      .select(WANTED_BANDS)
      .filterBounds(roi)
      .filterDate('2017-01-01', '2023-12-31')
      .filter(
          ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',10)
        )
      #.limit(10)
      #.filter(ee.Filter.calendarRange(1, 6, 'day_of_week'))
  )


  firstImage=sentinel2ImageCollection.first()

  Map.addLayer(
      sentinel2ImageCollection,
      {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1},
      'Sentinel-2',
  )

  firstImage=sentinel2ImageCollection.first()

  ndviCollection = sentinel2ImageCollection.map(mapFunctionNDVI) #collection of ndvi sentinel images
  #print(ndviCollection.first().getInfo())

  statsCollection = ndviCollection.map(calculateNDVIStatsForImage)
  #printType('statsCollection', statsCollection)
  statsList = statsCollection.toList(statsCollection.size())
  statsLength = statsList.length().getInfo()
  image = statsCollection.first()
  dateStart = format_date(image.get('system:time_start').getInfo())
  print('DONE!')


  dictionaryCollection = ndviCollection.map(calculateNDVIStatsForImageAsDictionary)
  dictionaryList = dictionaryCollection.toList(dictionaryCollection.size())
  dictionaryLength = dictionaryList.length().getInfo()
  featureList = []
  print('Start time =', datetime.now())
  for index in range(dictionaryLength):
    image = ee.Image(dictionaryList.get(index))
    dateStart = format_date(image.get('system:time_start').getInfo())
    print('dateStart:',dateStart)
    dictionary = image.get('stats_dictionary').getInfo()
    dictionary['dateStart'] = dateStart

    feature = ee.Feature(None, dictionary)
    featureList.append(feature)

  featureCollection = ee.FeatureCollection(featureList);
  ee.batch.Export.table.toDrive(
      collection=featureCollection,
      description=desc,
      folder=fold,
      fileFormat='CSV',
  ).start()
  print('Task Started')
  print('End =', datetime.now())

markdown = """
Web App URL: <https://geotemplate.streamlit.app>
GitHub Repository: <https://github.com/giswqs/streamlit-multipage-template>
"""
st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "https://i.imgur.com/UbOXYAU.png"
st.sidebar.image(logo)
st.title("Interactive Map")

col1, col2 = st.columns([4, 1])
options = list(leafmap.basemaps.keys())
valleys = ['Sacramento', 'San Joaquin']
model=None
sa_model = load_model('sa_model')
sj_model = load_model("sj_model")

initialize_session_state()

#m = leafmap.Map(locate_control=True, latlon_control=True, draw_export=True, minimap_control=True,center=(35.658581,139.745438),zoom=15,google_map="SATELLITE")
m=folium.Map(location=[37.011918, -120.283654], zoom_start=10)


basemaps = {
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    )
}
basemaps['Google Satellite'].add_to(m)
Draw(export=True).add_to(m)
#output = st_folium(m, width=1000, height=500)



if 'valleySelect' not in st.session_state:
    st.session_state['valleySelect'] = ''

if 'm' not in st.session_state:
    st.session_state['m'] = m

index = options.index("SATELLITE")



with col2:

    basemap = st.selectbox("Select a basemap:", options, index)
    st.write("Select which valley your field is in:")

    st.session_state['valleySelect'] = st.selectbox('Select a valley', valleys)
    
    if st.session_state['valleySelect']=='Sacramento':
        model=sa_model
        #st.session_state['valleySelect']='Sacramento'
    else:
        model=sj_model
        #st.session_state['valleySelect']='San Joaquin'

    slider_val = st.slider("Select a step value", min_value=1, max_value=24,step=1,help='Select a step value')
    final_steps=get_steps(slider_val)

    
with col1:
    if st.session_state['valleySelect'] == 'Sacramento':
        #st.session_state['m'].set_center(-120.283654, 37.011918, 10)
        #st.session_state['m'].to_streamlit(height=500)
        predict(sa_model,final_steps)
        #st.write('submitted :)')
        #st.write("valley", st.session_state['valleySelect'])
        #st.write("slider", slider_val)     
    else:
        #st.session_state['m'].set_center(139.745438, 35.658581, 10)
        #st.session_state['m'].to_streamlit(height=500)
        predict(sj_model,final_steps)
        #st.write('submitted :)')
        #st.write("valley", st.session_state['valleySelect'])
        #st.write("slider", slider_val)
    #output = st_folium(m, width=700, height=500)
    st.session_state['m']=m
    #st.write(output)

    info=st.session_state['m']
    #st.write(info.st_draw_features("all_drawings"))



map_data = st_folium(
    m,
    center=st.session_state["center"],
    zoom=st.session_state["zoom"],
    key="new",
    width=1285,
    height=725,
    returned_objects=["all_drawings"],
    use_container_width=True
)

print('st.session_state is', st.session_state)
print('')
print('')
print('')




if 'coords' not in st.session_state:
    st.session_state['coords'] = st.session_state["854d756172cea1cf186e8d95e0274ad865d14305dc6131c0b525e67e2b1076a9"]["all_drawings"][0]["geometry"]["coordinates"]
    
print('coords:', st.session_state['coords'])
print('st.session_state[\'m\']:', st.session_state['m'])

st.write(st.session_state)  
#if output is not None:
st.write(m)

region=st.session_state['coords']
desc="field1"
fold="Datasets/SA/Cluster1"
collect(region,desc,fold)