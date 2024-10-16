# Are we in a Doom Loop, San Francisco?
<table style="width:100%"><tr><center><td width="100%">
  <img src="images/sf_crime_locations_map.png" border="0"/>
  <br><em>Figure 1: Crime Incident Distribution by location (2018 - Present)</em>
</td></center></tr></table>

<p align='right'>
Capstone Project<br>
UC Berkeley ML/AI Professional Certification coursework<br>
Fazeel Mufti
</p>
  
**Resources**

* [`ExploratoryDataAnalysis.ipynb`](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ExploratoryDataAnalysis.ipynb)
  * Jupyter notebook containing the initial exploratory data analysis (EDA) of the dataset to develop a domain understanding
  * Includes data retrieval and cleanup required before we can apply DS/ML techniques
* [`ModelExploration.ipynb`](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ModelExploration.ipynb)
  * Jupyter notebook detailing the Data Models that were explored
  * Includes tuning various classification models and analysis of the wining `XGBClassifier` model
* [`ModelVisualizations.ipynb`](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ModelVisualizations.ipynb)
  * Visualizations developed to support the project story line
* `src` directory: Utility python code for the project
* `data` directory: You can download this file to play with the data yourself or get the latest from SFGov links below

## Executive summary

I am interested in forecasting and classification problems and use of data for finding solutions to social issues. As a San Francisco resident, I have looked skeptically at recent reporting on the **_San Francisco Doom Loop_**! There has been a regular narrative that the city's downtown area is in a downward spiral due to a combination of pandemic-related effects, declining foot traffic, rising homelessness, drug use, and businesses closing. The term gained traction as offices remained empty following COVID-19, which reduced the city's tax revenues, led to closures of key stores, and increased concerns about safety.

I came across this daily feed from the SF Police Dept. (SFPD) from January 2018 thru the present. I used this data of triaged crime incidents as a way to test the **_San Francisco Doom Loop hypothesis_**. My goals are to:

> 1. Correctly classify the crime category based on the historical data as categorized by SFPD officers
> 1. Learn how machine learning and Data Sciences can be applied to Social Engineering issues

This work is part of my Capstone Project for the UC Berkeley Artificial Inetlligence and Machine Learning Professional Certification.

### Tl;DR

#### This is a hard problem!

#### The more things change, the more they remain the same!

<table style="width:100%"><tr>
  <td width="100%"><em>Figure TO_DO: Total of 894,585 incidents from 2018-01-01 00:00:00 to 2024-10-02 21:45:00</em><img src="images/incidents_per_day.png" border="0"/></td>
</tr></table>

#### It's the stories we tell!

* Build Narratives that Support Ground Reality
* Refute False Narratives

This is a very high-level synopsis of my findings, I invite you to read the rest of this Summary Report and then dive deeper into the associated notebooks - there is a lot of interesting data, especially for anyone who's walked the streets of San Francisco!

## The Data - SFPD Daily Crime Incidents Reports

As part of the [City of San Francisco's Open DataSF Project](https://datasf.org/opendata/), the San Francisco Police Deparment (SFPD) provides a daily feed of Police Crime Incidents to the public for accountability purposes. This data has been made available since 2003 and provides a snapshot of crimes reported and investigated by the SFPD. Incoming reports are triaged and categorized based on the incident details and resolution. As such, this is a great resource on the makeup of crime in the City, and gives us an opportunity to apply Data Science (DS) and Machine Learning (ML) to analyze trends over a significant period of time.

The dataset is available as two CSV files with historical data from 2003-2018 and 2018-Present:

1. [San Franciso Police Department Incident Reports: Historical 2003 to May 2018](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry/about_data)
1. [Police Department Incident Reports: 2018 to Present](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783/about_data)

For this project, we will use the 2018-Present data as we are focused on the Doom Loop around the Covid Pandemic period:

* DataType: Time Series data for SFPD incidents reported by:
  * Police officers
  * Citizens using SFPD website and Twitter/X
* Published daily and ranges from Jan 1, 2003 up to the current time - as of Oct 2, 2024, my last snapshot, there are 894,585 records to analyze!
* The datafeed contains one incident per row, with some incidents spanning multiple rows as the police officer investigates the situation at the scene of the crime. The data does not include any subsequent investigation after the initial crime _recording_
* There are 35 feature columns containing different attributes about the incidents containing:
  * Categorization and sub-categorization of incidents, as well as resolution codes
  * GIS (latitude, longitude) and neighborhood data that can be cross-indexed with other SF City datasets
* Contains data for _incoming incidents_, not their final resolution, so this is not a way to gauge crime resolution, but onlhy a barometer of the overall incidence rates
  
#### Methodology
1. Exploratory Data Analysis: we first analyze the data to understand it and prepare it for the Data Modeling phase. The work summarized here is detailed in the [`ExploratoryDataAnalysis.ipynb`](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ExploratoryDataAnalysis.ipynb) Jupyter notebook
2. Model Exploration: We start with a few ML models suitable for multi-class classification problems, identify good candidates for optimization and evaluate them on our evaluation criteria - details in the [`ModelExploration.ipynb`](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ModelExploration.ipynb) notebook
3. Interpretation of Best Model: Finally we apply learnings from the trained model to our Doom Loop scenario - see [`ModelVisualizations.ipynb`](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ModelVisualizations.ipynb)

### Exploratory Data Analysis

#### Inspect Features

Since this data is used for both incident recording as well as internal data house-keeping, during the first pass, we used the following strategy to reduce the initial set of columns:

* Columns that are mostly empty and not related to our classification problem
  * `esncag_-_boundary_file`
  * `central_market/tenderloin_boundary_polygon_-_updated`
  * `civic_center_harm_reduction_project_boundary`
  * `hsoc_zones_as_of_2018-06-05`
  * `invest_in_neighborhoods_(iin)_areas`
* Any administrative columns that are not related to predicting the crime category from its related features:
  * `report_type_code`
  * `report_type_description`
  * `filed_online`
* Columns that identify street or address information that we will not be using as we are focused on predictions based on LatLong, neighborhood or Police District and Precinct
  * `intersection`
  * `cnn`
  * `Point: Redundant since we have latitude, longitude`
* Columns that have to do with City governance and not related to crime prediction
  * `supervisor_district`
  * `supervisor_district_2012`
  * `crrent_supervisor_districts`

**Temporal Features**

Our data is indexed by DateTimeStamp, but this is not very useful to work with. We converted the time of each incidence to it's component values to facilitate analysis and for consideration by our ML models:

  * `hour`
  * `minute`
  * `day`
  * `month`
  * `year`
  * `Day of Week`

In addition, time has semantic meaning beyond it's absolute value and concepts like weekends, holidays need to be represented in our predictions. So we introduced new synthetic features to address this issue:

* `weekend`
* `season`: Winter, Spring, Summer, Fall
* `US Holidays`
* `Time of Day`: Morning, Afternoon, Evening, Night

**Geo-Based Features**

According to [LatLong.net](https://www.latlong.net/place/san-francisco-bay-area-ca-usa-32614.html) the San Francisco County is bounded by the following rectangle:

* Latitude Range:
  * Northern limit: 37.8330° N
  * Southern limit: 37.7031° N
* Longitude Range:
  * Western limit: -122.52279° W
  * Eastern limit: -122.3515° W

We verified that all data was within these bounds - the SFPD has actually done a good job as no exceptions were discovered.

**Incidence-Specific Features: Target Variable - Category**

We then looked at features that are important for classifying the incidents because there were mulitple rows with the same incident ID, and we found the explanation as follows (from the [DataSF Dataset Explainer](https://sfdigitalservices.gitbook.io/dataset-explainers/sfpd-incident-report-2018-to-present#multiple-incident-codes)):

>Incident reports can have one or more associated Incident Codes. For example, an officer may have a warrant for an
>arrest and while making the arrest, discovers narcotics in the individual’s possession. The officer would record
>two Incident Codes: (1) for the warrant and (2) for the discovery of narcotics.
>
>When multiple Incident Codes exist, the Incident ID, Incident Number and CAD Numbers remain the same and the
>Row ID field can be used as a unique identifier for each row of data. An example is provided below.

Since we are interested in predicting the `incident_category` based on time and location, we'll retain  all three rows, each with a different category, but remove the columns that merge them into a single incident. This gives us more training data with a whetted outcome, i.e. target variable.

So we removed the following columns:

  * incident_datetime / report_datetime
  * incident_id / incident_code / row_id / incident_number / cad_number
  * incident_subcategory
  * incident_description

and only retained `incident_category` as our target variable:

<table style="width:100%"><tr>
  <td width="100%"><em>Figure TO_DO: Category, our target variable, has 49 possible classes</em><img src="images/categories_histogram.png" border="0"/></td>
</tr></table>

Looking at the `category` distribution, we noticed the following categories:

* `Human Trafficking (A), Commercial Sex Acts` (10 occurences) and `Human Trafficking, Commercial Sex Acts` (3 occurences): Since they are so few, we'll comnine them into a single `Human Trafficking (Combined)` class
* `Motor Vehicle Theft` (4,900 occurences) and `Motor Vehicle Theft?` (8 occurences) are similar enough that I converted them into the dominant class
* `Weapons Offense` (619 occurennces) `Weapons Offence` (4 occurences) look like an entry error, so they were converted to the dominant class

This left us with 45 possible Cateogries that a crime can be classified in San Francisco, and these will be the prediction outputs of our ML models.

**District-Specific Features**

We also looked at Police District and Neighborhood features and cleaned them up so we could train our on these two specific dimenions. Here is a heatmap of the Crime Categories distributed across the repobsible Police Districts based on the <latitude, longituded> of the incidence location.

<table style="width:100%"><tr>
  <td width="100%"><em>Figure TO_DO: Police District Heatmap of all Incidents</em><img src="images/incidents_per_district_heatmap.png" border="0"/></td>
</tr></table>

Overall, we found numerous data entry issues that were fixed or cleaned as much as possible. The remaining data is due to human error and is difficult to "fix" easily. After this cleanup, here is the final list of features that will be used to train our models:

```
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 829328 entries, 2023-03-11 14:00:00 to 2023-03-21 17:42:00
Data columns (total 18 columns):
 #   Column           Non-Null Count   Dtype  
---  ------           --------------   -----  
 0   date             829328 non-null  object 
 1   time             829328 non-null  object 
 2   year             829328 non-null  int64  
 3   dow              829328 non-null  object 
 4   category         829328 non-null  object 
 5   resolution       829328 non-null  object 
 6   police_district  829328 non-null  object 
 7   neighborhood     829328 non-null  object 
 8   latitude         829328 non-null  float64
 9   longitude        829328 non-null  float64
 10  hour             829328 non-null  int64  
 11  minute           829328 non-null  int64  
 12  day              829328 non-null  int64  
 13  month            829328 non-null  int64  
 14  weekend          829328 non-null  int64  
 15  season           829328 non-null  object 
 16  holiday          829328 non-null  bool   
 17  tod              829328 non-null  object 
dtypes: bool(1), float64(2), int64(6), object(9)
```

## Model Development


#### Results

<table style="width:100%"><tr>
  <td width="100%"><em>Figure TO_DO: Results Tally</em><img src="images/table_results_tally.png" border="0"/></td>
</tr></table>

<table style="width:100%"><tr>
  <td width="100%"><em>Figure TO_DO: Candidate Models for Tuning</em><img src="images/table_models_tuned.png" border="0"/></td>
</tr></table>

<table style="width:100%"><tr>
  <td width="100%"><em>Figure TO_DO: Hyperparameter Tuning Results</em><img src="images/table_models_CV.png" border="0"/></td>
</tr></table>

## Evaluation & Interpretation

## Next steps
