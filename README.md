# Code to reproduce Yomix paper figures
## Run main.py on the file of your choice to evaluate cosg, yomix's function to compute signature and scanpy(t-test, wilcoxon)

In a python virtual environment, do:

    python3 main.py [path_to_your_file]

where your file is an anndata object (h5ad format) with 'labels' in obs field

The results will be stored in the 'result' folder as csv files : one for the perfomances of the classifiers on the signatures from the different methods, one for the time it took to compute those methods (with the suffix _runtime), one for the features semected by the different methods for the different task (with suffix _top_features).

Now you can run the scripts to plot the figures by passing the file recently produced as argument


To get the main figure of the SVM performances comparison across labels:

	python3 performance_main_figure.py 

To get runtime comparison:

	python3 runtime_figure_main.py
	
For the supplementary figures comparing KNN, SVM and RF:

python3 classifier_comparison.py

For the volcano plots:

python3 volcano_plots.py


## List of contributors

Nicolas Perrin-Gilbert

Joshua Waterfall

Pierre Fumeron

Nisma Amjad

Jason Z. Kim

Erkan Narmanli

Christopher R. Myers

James P. Sethna

Jérôme Contant

Thomas Fuks

Julien Vibert

Silvia Tulli
