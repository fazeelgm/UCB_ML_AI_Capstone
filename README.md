### Project Title

**Author**: Fazeel Mufti

#### Executive summary

As part of the City of San Francisco's DataSF Project, the San Francisco Police Deparment provides a daily feed of Police Incidents to the public for accountability purposes. This data has been made available since 2018 and provides a snapshot of crimes reported and investigated by the SFPD. Incoming reports are triaged and categorized based on the incident details and resolution. As such, this is a great resource on the makeup of crime in the City, and gives us an opportunity to apply Data Science (DS) and Machine Learning (ML) to analyze trends over a significant period of time.

I am interested in forecasting and classification problems and use of data for social issues. I came across this dataset daily feed from the SF Police Dept. (SFPD) a few months ago when there was a lot of reporting about San Francisco being in a doom-loop, and this dataset was referred to show that things were actually not that bad from a crime lens. I plan to use the triaged crime incidents as a way to train several ML models to predict the likelihood of a crime category occuring in a specific area at a specific time and show the accuracy of my predictions based on the classification by the Police Officers. This will be used as a term project for my UC Berkeley Artificial Inetlligence and Machine Learning Professional Certification.

#### Rationale

By focusing on crime prediction and hotspot identification, law enforcement can enhance their ability to prevent crime, use resources more efficiently, and build stronger relationships with the communities they serve, ultimately leading to a safer and more secure environment for everyone. The data-driven decision-making approach can benefit the following areas:

* Proactive Policing and Crime Prevention
* Efficient Use of Resources
* Data-Driven Decision Making
* Enhancing Public Safety and Trust
* Cross-Departmental Collaboration
* Emergency Preparedness
  
#### Research Question
1. Can we predict the likelihood of a crime occuring in a specific area at a specific time?
1. Can we identify Hotspots by crime categories?

#### Data Sources

The dataset is available as two CSV files with historical data from 2003-2018 and 2018-Present:

1. [San Franciso Police Department Incident Reports: Historical 2003 to May 2018](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry/about_data)
1. [Police Department Incident Reports: 2018 to Present](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783/about_data)

* DataType: Time Series data for SFPD incidents reported by:
  * Police officers
  * Citizens using SFPD website and Twitter/X
* Published daily and ranges from Jan 1, 2003 up to the current time
* Includes categorization and sub-categorization of incidents, as well as resolution codes
* Include GIS (LatLong) and neighborhood data that can be cross-indexed with other SF City datasets
  
#### Methodology
1. Exploratory Data Analysis
2. Model Exploration
3. Discussion of Best Model

#### Results
What did your research find?

#### Next steps

#### Outline of project

* [notebooks/ExploratoryDataAnalysis.ipynb](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ExploratoryDataAnalysis.ipynb)
  * Jupyter notebook containing the initial exploratory data analysis (EDA) of the dataset to develop a domain understanding
  * Includes data retrieval and cleanup required before we can apply DS/ML techniques
* [notebooks/ModelExploration.ipynb](https://github.com/fazeelgm/UCB_ML_AI_Capstone/blob/main/notebooks/ModelExploration.ipynb)
  * Jupyter notebook detailing the Data Models that were explored

##### Contact and Further Information
