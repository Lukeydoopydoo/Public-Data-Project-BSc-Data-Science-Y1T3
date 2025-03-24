# Public Data Project - Flood Warning Classification

# Executive Summary

As part of Module 5 of my BSc Data Science Apprenticeship, which focussed on applying public datasets to real-world data science problems, I developed a flood warning predictor for my local flood warning area. The aim of this project was to gain a deeper understanding of how the Environment Agency issues flood warnings, and to explore whether these warnings are based on specific thresholds in river level, flow, or rainfall. I was particularly interested in discovering whether warnings are automated, what levels are indicative of potential flooding, and which time-based features—such as lagged readings — can help predict flood events.

The project exclusively utilised data from the Environment Agency Hydrology API, accessed via the DEFRA Data Services Platform. I selected Flood Warning Area 761 as a prototype (and because it was one of the most local to me), allowing for focussed model development and experimentation.

The final model, a Random Forest classifier, demonstrated strong performance with a precision of 0.88, recall of 0.76, and an F1 score of 0.82. Final features included lagged river levels (1h and 2h), short-term and medium-term rolling averages, and exponentially weighted moving averages (EWMAs) over time windows ranging from 3 to 24 hours. Rainfall was incorporated as cumulative totals over the past 12 hours, 3 days, and 2 weeks to account for both short-term and long-term saturation effects. Flow-based features were also included, mirroring the lag, average, and EWMA structure used for river levels. 

In addition, a Decision Tree model was trained on the top features to extract interpretable thresholds, offering a clear set of rules for future monitoring and potential integration into operational decision-making.

Many avenues for further investigation were also identified throughout the creation of the final model and are discussed in the Next Steps section.

# API & Other Data Importation

The **Environment Agency Hydrology API** was used to identify relevant monitoring stations and retrieve time series data for **river level** and **river flow** measurements. This API provided historical data directly linked to active stations associated with the river that was identified in the chosen Flood Warning Area.

In addition to API-based data, other datasets were downloaded directly from the **DEFRA Data Services Platform**. These included the **Historic Flood Warning Data**, **Flood Warning Area Data**, and **Historic Rainfall Data** specific to the West Sedgemoor region.

API Import File Names: 

**[API IMPORT] (River Parrett) Hydrology Flow API Data (15 Minute Historic Just The Stations Of Interest)**

**[API IMPORT] (River Parrett) Hydrology Level API Data (15 Minute Historic Just The Stations Of Interest)**

**Example:** Identification of stations on the River Parrett which provide water level data. 
```
# API URL to get water level stations on River Parrett.
api_url = "http://environment.data.gov.uk/hydrology/id/stations.json"
params = {"observedProperty": "waterLevel", "riverName": "River Parrett"}

# Request data from the API using 'get'.
response = rq.get(api_url, params=params)
response.raise_for_status()  # Stop if there's an error and defines the error.
stations = response.json().get("items", []) #Converts the response into a Python dictionary and gets the "items" key.

# Extract relevant data into a list using list comprehension, checking it is a dictionary and is complete. 
station_data = [{"Station Name": station.get("label"),"GUID": station.get("stationGuid"),"Latitude": station.get("lat"),"Longitude": station.get("long"),"River Served": station.get("riverName", "N/A"),"Catchment": station.get("catchmentName", "N/A")} for station in stations if isinstance(station, dict) and station.get("notation")]

# Convert to DataFrame and save as CSV.
df_stations = pd.DataFrame(station_data)
```
**Example:** Using the identified stations to return water level data from those stations with the specific parameters identified. 
```
#Initialise a list.
readings = []

# Loop over unique station GUIDs
for guid in df_stations["GUID"].dropna().unique():
    #API URL.
    url = "http://environment.data.gov.uk/hydrology/data/readings.json"
    #Parameters for the request. 
    params = {"station": guid,"observedProperty": "waterLevel","periodName": "15min","mineq-date": "2019-12-19","max-date": "2024-11-25","_limit": 2000000}
    #Parameters for the request. 
    response = rq.get(url, params=params)
    response.raise_for_status()  # Stop if there's an error and defines the error.
    items = response.json().get("items", []) #Returns a Python dictionary of the items.
    # Append readings to the list, adding the GUID for reference.
    for item in items:
        item["GUID"] = guid
        readings.append(item)

# Create a DataFrame from the collected readings.
df_readings = pd.DataFrame(readings)
```

# Data Preprocessing 

As part of this project, each key data type — **Flow**, **Level**, **Rainfall**, and **Warning** — is handled in a dedicated preprocessing file. These files focus on the  exploration and preparation of the datasets, with the goal of cleaning the data and understanding how each can be transformed into useful features for the final machine learning model.

The preprocessing steps include cleaning, exploring the impact of data quality types, checking for missing data, exploring outliers and engineering time-based features such as lag values, rolling averages, and cumulative measures. 

Preprocessing Files Names: 

**[PREPROCESSING] Level Data**

**[PREPROCESSING] Flow Data**

**[PREPROCESSING] Warning Data**

**[PREPROCESSING] Rainfall Data**

Since preprocessing for water level, water flow, and rainfall data followed a similar approach, water flow data is presented here as a representative example.
The data frame was loaded and followed by initial exploration of the data quality dimensions: completeness, uniqueness, consistency, timeliness, validity and accuracy. 

![image](https://github.com/user-attachments/assets/78648f28-3fda-48be-8b04-04b670b127a6)

During exploratory data analysis, I examined the distribution of values and quality flags to evaluate data reliability.

![image](https://github.com/user-attachments/assets/643a5537-6bf6-4237-8d80-cbce53cdd69d)

![image](https://github.com/user-attachments/assets/62ecb9ea-895e-4929-a015-40985d6e7fe2)

![image](https://github.com/user-attachments/assets/7567dc4d-1364-414d-b786-27671b97a563)

![image](https://github.com/user-attachments/assets/343197e0-703b-486b-82e9-cad18c7a69b2)

In most cases, closer inspection of the data quality types supported retaining the data, as it appeared representative of the dataset overall. In these instances, the potential negative impact of outlier mitigation was judged to outweigh the benefits. Where data quality issues were more significant, time-series interpolation was applied to address them.

![image](https://github.com/user-attachments/assets/60245360-3e91-4728-8409-ec6bcb8f071b)

The 15-minute interval data was resampled to hourly and the water flow and level utilised the max reading to be representative over the hour whereas rainfall utilised the mean. 

```
warnings_df = warnings_df[warnings_df['id'] == "Flood_Warning_Areas.761"]

warnings_df
```

I kept only the columns relevant to later steps. 

![image](https://github.com/user-attachments/assets/039ae9b8-9e35-4d75-ac83-8e8a476081a6)

The approach to Flood Warning Data differed in that it involved plotting the geometry after merging the Historic Flood Warning Data with the Flood Warning Areas, in order to assess the effects of any imperfect matches. While this step could technically be removed—since the geometry was only used with GeoPandas to identify the nearest Flood Warning Areas to my postcode—it has been retained. This is because it doesn’t affect the single area selected and keeps the workflow flexible for future development, where the geometry may become necessary.

![image](https://github.com/user-attachments/assets/71abb440-89af-462b-9aef-961f54f1d762)

I filtered the dataset for only the Flood Warning Area chosen.

```# Convert the dateTime column to datetime.
df['dateTime'] = pd.to_datetime(df['dateTime'])

# Resample the data to hourly intervals and keeping the maximum flow value.
df_hourly = df.resample('H', on='dateTime').max().reset_index()
```
I removed any flood messages which did not denote either the start or end of a warning.

```
warnings_df = warnings_df[~warnings_df['Message Type'].str.contains('Update', case=False, na=False)]
warnings_df
```
I generated a full date range at hour frequency by making a new data frame with a date range and then merging the original data frame with this new data frame. 

```
warnings_df = warnings_df.copy()
full_range = pd.date_range(warnings_df['Approved'].min(), warnings_df['Approved'].max(), freq='H')
full_dates = pd.DataFrame({'Approved': full_range})
warnings_df_filled = full_dates.merge(warnings_df, on='Approved', how='left')
warnings_df_filled
```
I mapped the flood warnings to either 1 or 0 depending on whether a message was setting a warning or removing it. 

```
warnings_df_filled['daily_indicator'] = warnings_df_filled['Message Type'].map({'Flood Warning': 1, 'Remove Flood Warning': 0})
```
I then forward filled the new ‘daily_indicator’ column so that it was complete and kept only the columns needed for merging. 

```
warnings_df_filled = warnings_df_filled[['Approved', 'daily_indicator']]
warnings_df_filled
warnings_df_filled['daily_indicator'] = warnings_df_filled['daily_indicator'].ffill()
```
![image](https://github.com/user-attachments/assets/779b9122-45fa-4bfb-a551-de7bdf81dd1b)


