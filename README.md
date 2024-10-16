# Are we in a Doom Loop, San Francisco?
<table style="width:100%"><tr><center><td width="100%">
  <img src="images/sf_crime_locations_map.png" border="0"/>
</td></center></tr></table>

<p align='right'>
Capstone Project<br>
UC Berkeley ML/AI Professional Certification coursework<br>
Fazeel Mufti
</p>
  
**Jupyter Notebooks**

These Jupyter notebooks contain the detailed analysis summarized in this Executive Summary:

* [`ExploratoryDataAnalysis.ipynb`](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ExploratoryDataAnalysis.ipynb)
  * Jupyter notebook containing the initial exploratory data analysis (EDA) of the dataset to develop a domain understanding
  * Includes data retrieval and cleanup required before we can apply DS/ML techniques
* [`ModelExploration.ipynb`](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ModelExploration.ipynb)
  * Jupyter notebook detailing the Data Models that were explored
  * Includes tuning various classification models and analysis of the wining `XGBClassifier`
* [`ModelVisualizations.ipynb`](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ModelVisualizations.ipynb)
  * Visualizations developed to support the project story line
* `src` directory: Utility python code for the project
* `data` directory: You can download this file to play with the data yourself or get the latest from SFGov links below

#### Executive summary

I am interested in forecasting and classification problems and use of data for finding solutions to social issues. I came across this daily feed from the SF Police Dept. (SFPD) a few months ago when there was a lot of reporting about San Francisco being in a **_Doom Loop_**. I plan to use this data of triaged crime incidents as a way to train several ML models to classify the crime category based on the historical data being captured by the SFPD, and show the accuracy of my predictions based on the classification by the Police Officers. Our goals are to: 

1. Classify the crime category based on the historical data categorized by SFPD officers
1. Learn how machine learning can help predict crime classification and aid police to protect our communities better

This will be used as the capstone project for my UC Berkeley Artificial Inetlligence and Machine Learning Professional Certification.

By focusing on crime prediction and hotspot identification, law enforcement can enhance their ability to prevent crime, use resources more efficiently, and build stronger relationships with the communities they serve, ultimately leading to a safer and more secure environment for everyone. The data-driven decision-making approach can benefit the following areas:

* Proactive Policing and Crime Prevention
* Efficient Use of Resources
* Data-Driven Decision Making
* Enhancing Public Safety and Trust
* Cross-Departmental Collaboration
* Emergency Preparedness
  
#### Data Sources

As part of the [City of San Francisco's Open DataSF Project](https://datasf.org/opendata/), the San Francisco Police Deparment (SFPD) provides a daily feed of Police Crime Incidents to the public for accountability purposes. This data has been made available since 2018 and provides a snapshot of crimes reported and investigated by the SFPD. Incoming reports are triaged and categorized based on the incident details and resolution. As such, this is a great resource on the makeup of crime in the City, and gives us an opportunity to apply Data Science (DS) and Machine Learning (ML) to analyze trends over a significant period of time.

The dataset is available as two CSV files with historical data from 2003-2018 and 2018-Present:

1. [San Franciso Police Department Incident Reports: Historical 2003 to May 2018](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry/about_data)
1. [Police Department Incident Reports: 2018 to Present](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783/about_data)

* DataType: Time Series data for SFPD incidents reported by:
  * Police officers
  * Citizens using SFPD website and Twitter/X
* Published daily and ranges from Jan 1, 2003 up to the current time. Due to limited compute resources, I will focus on the incidents from 2018 till today
* Includes categorization and sub-categorization of incidents, as well as resolution codes
* Include GIS (LatLong) and neighborhood data that can be cross-indexed with other SF City datasets
  
#### Methodology
1. Exploratory Data Analysis
2. Model Exploration
3. Discussion of Best Model

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

#### Next steps

#### Outline of project


##### Contact and Further Information
