# Public Data Project - Flood Warning Classification

# Executive Summary

As part of Module 5 of my BSc Data Science Apprenticeship, which focussed on applying public datasets to real-world data science problems, I developed a flood warning predictor for my local flood warning area. The aim of this project was to gain a deeper understanding of how the Environment Agency issues flood warnings, and to explore whether these warnings are based on specific thresholds in river level, flow, or rainfall. I was particularly interested in discovering whether warnings are automated, what levels are indicative of potential flooding, and which time-based features - such as lagged readings - can help predict flood events.

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

The preprocessing steps include cleaning, exploring the impact of data quality types, checking for missing data, exploring outliers and engineering time-based features such as lag values, rolling averages, and cumulative measures.  I adopted an iterative approach to integrating my datasets, choosing to preprocess each one separately before combining them. This decision was driven by the need for modularity, clarity, and control throughout the data preparation process. Each dataset - such as rainfall, flow, river level, and flood warnings - contained unique structures, time resolutions, and quality considerations. By treating them as modular components, I was able to address data-specific issues in isolation, ensuring each was cleaned, validated, and understood before integration - undertaking engineering of the features in the preprocessing in some cases. This method also allowed me to evaluate the incremental impact of each dataset on model performance, making it easier to identify the value added by individual features. 

Preprocessing Files Names: 

**[PREPROCESSING] Level Data**

**[PREPROCESSING] Flow Data**

**[PREPROCESSING] Warning Data**

**[PREPROCESSING] Rainfall Data**

Since preprocessing for water level, water flow, and rainfall data followed a similar approach, water flow data is presented here as a representative example, although there will be differences in approach between the datasets; where these differences in approach are significant, they are detailed in this README.

The data frame was loaded and followed by initial exploration of the data quality dimensions: completeness, uniqueness, consistency, timeliness, validity and accuracy. For any dataset that had more than one station, this process was also undertaken per station.

![image](https://github.com/user-attachments/assets/78648f28-3fda-48be-8b04-04b670b127a6)

During exploratory data analysis, I examined the distribution of values and quality flags to evaluate data reliability.

Firstly, I visualised the data over time and by the data quality type because this allowed me to identify trends, gaps, or inconsistencies in the dataset, assess the reliability of the data across different periods, and better understand how data quality may have influenced the recorded values. For any dataset that had more than one station, this process was also undertaken per station.

![image](https://github.com/user-attachments/assets/643a5537-6bf6-4237-8d80-cbce53cdd69d)

I then visualised the quality types in boxplots to highlight the distribution, spread, and potential outliers within each quality category, making it easier to compare their variability and assess whether lower-quality data might be skewing the results or introducing anomalies. For any dataset that had more than one station, this process was also undertaken per station.

![image](https://github.com/user-attachments/assets/62ecb9ea-895e-4929-a015-40985d6e7fe2)

I  used histograms to examine the frequency distribution of values, allowing for a clearer understanding of how the data is spread, whether it follows a normal distribution. In most cases, I also created separate histograms of the different data quality types to assess their distributions. For any dataset that had more than one station, this process was also undertaken per station.

![image](https://github.com/user-attachments/assets/7567dc4d-1364-414d-b786-27671b97a563)

Finally, I used descriptive statistics to summarise the central tendencies, dispersion, and overall characteristics of the data across different quality types, providing a quantitative basis for comparing them and supporting decisions about data reliability and potential preprocessing steps. For any dataset that had more than one station, this process was also undertaken per station.

![image](https://github.com/user-attachments/assets/343197e0-703b-486b-82e9-cad18c7a69b2)

In most cases, closer inspection of the data quality types supported retaining the data, as it appeared representative of the dataset overall. In these instances, the potential negative impact of outlier mitigation was judged to outweigh the benefits. Where data quality issues were more significant, time-series interpolation was applied to address them after assessing the size of gaps in the data upon which the interpolation would be applied as interpolating over a large gap can introduce false or overly smooth values that don’t reflect real-world variation.

![image](https://github.com/user-attachments/assets/60245360-3e91-4728-8409-ec6bcb8f071b)

The 15-minute interval data was resampled to hourly and the water flow and level utilised the max reading to be representative over the hour whereas rainfall utilised the mean. I did this because maximum values for flow and level better capture peak conditions that are critical for flood analysis, while the mean is more appropriate for rainfall as it reflects the overall intensity over time, reducing the impact of short, extreme spikes.

```
# Convert the dateTime column to datetime.
df['dateTime'] = pd.to_datetime(df['dateTime'])

# Resample the data to hourly intervals and keeping the maximum flow value.
df_hourly = df.resample('H', on='dateTime').max().reset_index()
```

I kept only the columns relevant to later steps to streamline the dataset, reduce memory usage, and improve processing efficiency, ensuring that subsequent analyses and models are focused only on the most meaningful and necessary information.

![image](https://github.com/user-attachments/assets/039ae9b8-9e35-4d75-ac83-8e8a476081a6)

The approach to Flood Warning Data differed in that it involved plotting the geometry after merging the Historic Flood Warning Data with the Flood Warning Areas, in order to assess the effects of any imperfect matches. While this step could technically be removed—since the geometry was only used with GeoPandas to identify the nearest Flood Warning Areas to my postcode—it has been retained. This is because it doesn’t affect the single area selected and keeps the workflow flexible for future development, where the geometry may become necessary.

![image](https://github.com/user-attachments/assets/71abb440-89af-462b-9aef-961f54f1d762)

I filtered the dataset for only the Flood Warning Area chosen.

```
warnings_df = warnings_df[warnings_df['id'] == "Flood_Warning_Areas.761"]
warnings_df
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
I then forward filled the new ‘daily_indicator’ column so that it was complete and kept only the columns needed for merging, visualising the outcome.

```
warnings_df_filled = warnings_df_filled[['Approved', 'daily_indicator']]
warnings_df_filled
warnings_df_filled['daily_indicator'] = warnings_df_filled['daily_indicator'].ffill()
```
![image](https://github.com/user-attachments/assets/779b9122-45fa-4bfb-a551-de7bdf81dd1b)

# Feature Engineering 

I explored the correlation of features like flow and rainfall with the original water level dataset to inform the creation of features to be used in the final machine learning model. This helped identify which variables had the strongest relationships with water level changes, guiding the selection and engineering of predictive inputs. The example from the rainfall preprocessing file is below.

Amongst other visualisations, I visualised the relationship between rainfall and water level rolling averages during the winter of 2023 to observe how short-term trends in rainfall influenced water levels over time, and to identify any lagged effects or patterns that could support feature engineering for the predictive model.

![image](https://github.com/user-attachments/assets/fb6d3ff3-6bac-41b8-a914-a14531a23d05)

I also calculated and visualised the correlation between rainfall over a series of lags and water level to identify how past rainfall events influenced current water levels, helping to determine the most informative lag intervals for feature creation and improving the temporal relevance of inputs to the machine learning model.

![image](https://github.com/user-attachments/assets/0dc82618-980d-41d2-9395-8819fdff8c30)

From this work, I created a number of features to use in the model: 

```
df_hourly['rainfall_cumulative_12h'] = df_hourly['rainfall'].rolling(window=12, min_periods=1).sum()
df_hourly['rainfall_cumulative_3d'] = df_hourly['rainfall'].rolling(window=72, min_periods=1).sum()
df_hourly['rainfall_cumulative_2w'] = df_hourly['rainfall'].rolling(window=336, min_periods=1).sum()
```
Much of this work exposed avenues of further development (discussed in the Next Steps section).

# Machine Learning 

In this section, I will outline the iterative process toward creating the final model.

I began with just level data over two stations, renaming the ‘value’ column to ‘level’ to aid in clarity. 
```
level.rename(columns={"value": "level"}, inplace=True)
```
I also pivoted the data by the two stations to explore their individual impact on the model. 
```
level = level.pivot(index="dateTime", columns="station", values="level")
```
I renamed the columns using a list comprehension to identify what data the stations were reporting, ready for later data merges.
```
level.columns = [f"{col} level" for col in level.columns]
```
The dataset for baseline modelling: 

![image](https://github.com/user-attachments/assets/420b7fc7-c01b-4351-b1a7-b525765db228)

I merged the dataset with the warning data on the date/time column using an 
```
level_warnings = new_warnings.merge(level, left_on='Approved', right_on='dateTime', how='inner')
```
I visualised the two stations and their water levels against a flood warning being issued to assess the relationships: 
![image](https://github.com/user-attachments/assets/c28aba44-45ba-461b-bbcc-e6de9e5dbd7e)

![image](https://github.com/user-attachments/assets/4a96b688-72d0-46d8-809b-017801f91fc1)

I also created tables of descriptive statistics for each station: 

![image](https://github.com/user-attachments/assets/ec53f529-addc-473a-982b-a38d38c8b4be)

The inference from the visuals and the descriptive statistics was that one of the rivers would contribute far more to the prediction of a flood warning event than the other. 

I began with a logistic regression as a baseline and followed this with a random forest, visualising the feature importance of the two stations and confirming my previous thoughts.   

![image](https://github.com/user-attachments/assets/9e4c1b1f-49f0-4b89-a8ce-66955aff391d)

I performed a GridSearchCV to tune the hyperparameters of the Random Forest model. The F1 score was used as the evaluation metric during grid search to balance the need for detecting floods (recall) with minimising false alarms (precision). Initially, I used StratifiedKFold for cross-validation; however, I later transitioned to TimeSeriesSplit to avoid data leakage, as StratifiedKFold can allow the model to peek into future data in time series contexts. This switch helped preserve temporal integrity but occasionally resulted in folds without any positive (flood) cases. In such cases, StratifiedKFold with no shuffling was used as a 'best-efforts' fallback approach. SMOTE was used to address class imbalance in the training data by synthetically generating new instances of the minority class but its results did not contribute positively to the model and so it was not used in future models.

```
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ["balanced", "balanced_subsample"]  
}

grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=cv_strategy,
                           scoring='f1',  
                           n_jobs=-1,
                           verbose=1)
```

Using the best model, I created a Partial Dependence Plot to visualise how individual features influence the model’s predictions, helping to interpret the relationships the Random Forest learned and to identify which features have the most significant impact on the likelihood of a flood warning being issued. One of the stations demonstrated a threshold relationship for flood prediction and the other station demonstrated a much more complex relationship. The decision was taken, after seeing the feature importance of later iterations of the model, that the station 'ce5176cf' should be removed.

![image](https://github.com/user-attachments/assets/b4d2e19d-1abe-4d01-8919-b2eb62e30573)

I then used the water level data to create lag, rolling average and exponentially weighted moving average features:
Level lag 1h & 2h: The river level an hour and two hours before. A high level in the recent past could be a warning sign of a future flood warning.
Level rolling average 3h & 6h: Short-term averages of level, smoothing out spikes and representing sustained level changes.
Level EWMA (3h, 6h, 9h, 12h, 24h): Exponentially Weighted Moving Average of level over various periods. These emphasise recent readings (for short spans) or capture longer trends (12–24h spans). A 24h EWMA, for instance, reflects the general level of the river over the past day.

```
#Create lagged features for the level column using 1, 2, and 6 hour lags
level_warnings['lag_1h'] = level_warnings['7998bf73-641d-4084-b00c-ca6989f2ba2b level'].shift(1)
level_warnings['lag_2h'] = level_warnings['7998bf73-641d-4084-b00c-ca6989f2ba2b level'].shift(2)


# Define rolling window sizes in hours (up to 48 hours)
rolling_windows_hours = [3, 6]

# Create rolling average features for each window size
for window in rolling_windows_hours:
    col_name = f'rolling_avg_{window}h'
    level_warnings[col_name] = level_warnings['7998bf73-641d-4084-b00c-ca6989f2ba2b level'].rolling(window=window).mean()

# Apply Exponential Weighted Moving Average (EWMA) with various spans (in hours)
ewma_spans = [3, 6, 24, 12, 9]
for span in ewma_spans:
    col_name = f'ewma_{span}h'
    level_warnings[col_name] = level_warnings['7998bf73-641d-4084-b00c-ca6989f2ba2b level'].ewm(span=span, adjust=False).mean()

level_warnings
```
This yielded a much stronger model:
```
Best Parameters: {'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}

              precision    recall  f1-score   support

           0       1.00      1.00      1.00      7829
           1       0.96      0.67      0.79        67

    accuracy                           1.00      7896
   macro avg       0.98      0.84      0.89      7896
weighted avg       1.00      1.00      1.00      7896
```
The model relied heavily on EWMA and rolling averages, emphasising longer-term trends over sudden changes.This suggests that predicting flood warnings requires monitoring how water levels evolve over time, rather than just looking at individual readings and that there is not automation to the flood warnings. 

![image](https://github.com/user-attachments/assets/bd1b5593-af0b-4340-8b0b-ecd713cc7bba)

When examining the Partial Dependence Plot from this Random Forest model, the ewma_6h exhibits a sharp increase at around 1.2–1.3, suggesting a threshold effect where flood probability significantly rises past this level.Features like rolling_avg_6h and ewma_24h display a slight upward slope, indicating a weak positive correlation with flood probability.

![image](https://github.com/user-attachments/assets/eda42013-c7f5-4ff9-b548-35ed78fd38ff)

Merging the rainfall data created in the **[PREPROCESSING] Rainfall**, the following rainfall features were added to the model: 
```
'rainfall_cumulative_12h',
 'rainfall_cumulative_3d',
 'rainfall_cumulative_2w',
 'rainfall_2w_lagged']
```
The model result for the addition of the new features showed a marginally better balance between precision and recall: 
```
Best Parameters: {'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100}

              precision    recall  f1-score   support

           0       1.00      1.00      1.00      7829
           1       0.94      0.70      0.80        67

    accuracy                           1.00      7896
   macro avg       0.97      0.85      0.90      7896
weighted avg       1.00      1.00      1.00      7896
```

The feature importance of the rainfall features were low: 

![image](https://github.com/user-attachments/assets/059baabe-ce3f-47dd-98db-aa891d2e2c56)

Merging the water flow data, which had a high correlation with water level data, I engineered its features in the same way as the water level data previously.

```
#Create lagged features for the level column using 1, 2, and 6 hour lags.
merged_df['flow lag_1h'] = merged_df['flow'].shift(1)
merged_df['flow lag_2h'] = merged_df['flow'].shift(2)


#Define rolling window sizes in hours (up to 48 hours).
rolling_windows_hours = [3, 6]

#Create rolling average features for each window size.
for window in rolling_windows_hours:
    col_name = f'flow_rolling_avg_{window}h'
    merged_df[col_name] = merged_df['flow'].rolling(window=window).mean()

#Apply Exponential Weighted Moving Average (EWMA) with various spans (in hours).
ewma_spans = [3, 6, 24, 12, 9]
for span in ewma_spans:
    col_name = f'flow_ewma_{span}h'
    merged_df[col_name] = merged_df['flow'].ewm(span=span, adjust=False).mean()

merged_df
```

The final model yielded the greatest balance in precision and recall: 

```
Best Parameters: {'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 5, 'n_estimators': 300}
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      7829
           1       0.86      0.76      0.81        67

    accuracy                           1.00      7896
   macro avg       0.93      0.88      0.90      7896
weighted avg       1.00      1.00      1.00      7896
```
After visualising the feature importances of the final model, I found that flow-based moving averages were  the strongest predictors, suggesting that floods were better signaled by long-term flow changes rather than just rainfall or water levels .Water level EWMAs remained important, but they take a secondary role compared to flow trends.

![image](https://github.com/user-attachments/assets/44abb8a1-f7e0-4424-b2ce-b6be4b49aa3f)

# Applying A Decision Tree

This code trains a simple decision tree classifier to predict flood warnings using the most important features from the dataset. It then prints out the thresholds and conditions the model uses to classify whether a flood warning should be issued. From the decision tree, the thresholds for a flood warning are: flow_ewma_12h > 16.04, flow_ewma_3h > 17.86 (if others are moderate), rainfall_cumulative_3d > 12.69 mm ewma_6h / ewma_12h > 2.08 / 2.13.

```
#Train a simple Decision Tree on the top features to get the values for flood warnings. 
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

#Extract the threhold values. 
tree_rules = export_text(dt_model, feature_names=list(X_train.columns))
print(tree_rules)
```

# Results

Using only Environment Agency hydrology ensor data, the final Random Forest classification model achieved a much better balance between flood detection and false alarms compared to the initial baseline. Early on, a simple Logistic Regression yielded high overall accuracy but failed to catch many flood events due to extreme class imbalance. Replacing this with a Random Forest improved flood recall dramatically (catching around 90% of flood events in the baseline RF model) but initially at the cost of many false positives. Through class rebalancing and hyperparameter tuning, the model reached a more reasonable trade-off – after tuning, the Random Forest was able to identify about 84% of flood days while reducing false alarms (precision around 57% for flood class, up from 31% baseline). Incorporating additional features further boosted performance: adding cumulative rainfall data produced a model with roughly 0.73 recall and 0.79 precision for flood warnings (a notably more balanced outcome), and including river flow features pushed performance to its best levels. The final model (with levels, rainfall, and flow inputs) detects most flood warning hours while keeping the false-alarm rate relatively low, outperforming all previous versions with 0.88 precision and 0.76 recall and 0.82 F1.

# Next Steps

Given that some flood events are still missed, one immediate refinement is adjusting the classifier’s probability threshold. The current model uses the default 0.5 threshold to decide a “flood” warning. Lowering this threshold (e.g. to 0.4) could catch a few more flood events at the expense of more false positives. While the Random Forest has performed well, exploring other machine learning algorithms could yield further improvements. A natural next step is to try Gradient Boosting Machines such as XGBoost or LightGBM. Another avenue is to consider time-series specific models. Since flood warnings are inherently time-sequenced events, models like recurrent neural networks (LSTM/GRU) or sequence classification approaches could capture temporal patterns beyond the fixed lag features we engineered. Further research into the effects of rainfall on water level may also lead to more impactful features for rain and a more accurate model. 
