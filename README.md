# eviction-data-quality
Predict tracts that will be in the highest 10% in terms of eviction rate in Chicago to inform legal outreach activities by Chica.


## Requirements
numpy 1.15.4
pandas 0.24.2
mapclassify 2.0.1
geopandas 0.4.0+67.g08ad2bf
sodapy 1.5.2
census 0.8.13
us 1.0.0 
shapely 1.6.4.post2
scipy 1.2.1
seaborn 0.9.0
sklearn 0.20.3
aequitas 0.38.0

##Repo organization

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



This research uses data from The Eviction Lab at Princeton University, a project directed by Matthew Desmond and designed by Ashley Gromis, Lavar Edmonds, James Hendrickson, Katie Krywokulski, Lillian Leung, and Adam Porton. The Eviction Lab is funded by the JPB, Gates, and Ford Foundations as well as the Chan Zuckerberg Initiative. More information is found at evictionlab.org.

