# Designing and implementing a Complex Visualisation Engine (CoVE)

## Project Background

The Synthetic Population Estimation and Scenario Projection Model (SPENSER) is a high resolution, comprehensive synthetic population 
estimation and projection model that employs dynamic microsimulation techniques to estimate characteristics and model change in 
population. SPENSER provides a set of tools that allow the user to customise scenario parameters to generate a scenario projection 
output and visualise these outputs.

The aim of this project is to create a framework for an innovative visualisation engine for SPENSER which combines interactive maps and 
projections plots in a web-based dashboard. This visualisation engine will communicate effectively the projection models’ findings to 
support the evidence-based decision making as well as produce more explainable simulation results.

## Data and Methods

Three different types of data were needed: geospatial, population projections and lookup data. The geospatial data was used for 
visualising the microsimulation results geographically. The lookup dataset was utilised for linking geographical boundaries of Local 
Authorities Districts (LAD) with the population projections data. Both datasets were obtained from the Office for National Statistics 
(ONS).

The population projections data were generated for LAD in England for the period 2016 to 2021, using the ukpopulation, humanleague and 
demographyMicrosim packages. The ukpopulation package was employed to create population projection variants. The population projections 
by demographic factors such as age, sex and ethnicity were produced by the demographyMicrosim and humanleague packages. 

Pre-processing and structural transformation were carried on the generated data allowing the tool to focus only on visualisation tasks 
which enhanced the performance and utility of the engine. The visualisation tool was created in a modular way to update it easily.

## Results

An easy to use interactive web dashboard-like visualisation tool was developed for visualising the dynamic microsimulation and 
projection model results. The elements are:
*	A choropleth map of the total population projection;
*	The population age and sex structure, using a population pyramid, disaggregated by ethnicity; 
*	A population projection variants plot. 

The plots are updated by clicking a geographical region on the map. The population projection can also be filtered by year and 
ethnicity (See Figure 1).


Figure 1:  Visualisation tool view for the selected filters: Tower Hamlets Borough of London, 2019 year and white (WHO) ethnicity. 
Inset of the zoomed out choropleth map.

## Value of the Research

This work has added interactive visualisation capability of the microsimulation outputs from SPENSER. This tool allows for an 
assessment in real-time of current population modelling results to check the usefulness of such outputs in the decision-making and to 
be able to interpret the results.

In addition, it provides an easy-to-use interface with data filtering controls and navigable time slider that is flexible and adaptable 
to different spatial scales of the results. This tool also provides effective information delivery to researchers and policy makers who 
are interested in microsimulation and modelling of populations.

Further details of tool can be found at https://github.com/aniuska/Visualisation-spenser
