# eviction-data-quality
Effort to improve the legal outreach activities of Non Profit organization __ to reduce eviction in Chicago. We use Machine Learning to build a model that predicts the tracts in Chicago that will be among the top 10% tracts with the highest eviction rate. With this prediction, we hope to improve the efficiency of the work made by ____, because they will be able to destinate their resources to the tracts mots it need. To do this, we use data from the Chicago Evictions Data Portal from the Lawyers’ Committee for Better Housing, and we complement it with data from the ACS and the CHicago Open Data Portal.


## Requirements
- Python 3.6
- numpy 1.15.4
- pandas 0.24.2
- mapclassify 2.0.1
- geopandas 0.4.0+67.g08ad2bf
- sodapy 1.5.2
- census 0.8.13
- us 1.0.0 
- shapely 1.6.4.post2
- scipy 1.2.1
- seaborn 0.9.0
- sklearn 0.20.3 
- aequitas 0.38.0

## Repository organization

- scripts\
Contains the scripts used to download and process the data, to run the machine learning pipeline, and to produce the descriptive statistics. It also contains Jupyter notebooks where the scripts are used.
	- mlpipeline.py: Contains the pipeline to split the train and test data, to fit the classifiers, to evaluate the models, and perform the biass and fairness analysis using Aequitas.

	-helper.py: Is a help[er script used by mlpipeline to run the classifiers.

	-download.py: Contains the functions to download the source data from the Evictions Lawyers, from the ACS and the Chicago Open Data Portal and save it in csv files. It also has the functions to load that data and to merge it all into a DataFrame aggregated by tract and year.

	-describe.py: Contains the functions to produce visualizations of the data.

	- run_classification.ipynb: Jupyter notebook where the data is loaded, the classifiers are built and the resultd are saved in results.csv

	- model_analysis.ipynb: Kupyuter notebook where we analyze the classification results, we select the best classifier and we fit the best model to the whole dataset.

	- graphs.ipynb: Jupyter notebook where we use the desvriptives.py script to produce visualizations.

- Inputs\
Contains the eviction data from the eviction lawyers in csv format and the sh script that calls download.py to download the rest of the data. When get_data.sh is run, the original inputs are created inside the ch_opdat and acs folders, and the secondary inputs that are aggregated by tract and year are created directly in the folder as csv files.

- results\
Contains the csv with the results of the classifiers.

- figures\
Contains figures created by the graph.ipynb notebook.

- Reducing_evictions_Chicago.pdf
Contains report of the project

## Running
- To download the data, clone the repository and run --sh get_files.sh-- inside the inputs repository.
- To run prediction of the Census tracts that in 2019 will be abong the highest 10%, run --run_classification.py--

## Contributors
Camilo Arias
Chi Nguyen
Angelica Valdiviezo

