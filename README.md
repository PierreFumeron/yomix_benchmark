# Code to reproduce Yomix paper figures
## Run main.py on the file of your choice to evaluate cosg, yomix's function to compute signature and scanpy(t-test, wilcoxon)

In a python virtual environment, do:

    python3 main.py [path_to_your_file]

where your file is an anndata object (h5ad format) with 'labels' in obs field

The results will be stored in the 'result' folder as csv files : one for the perfomances of the classifiers on the signatures from the different methods, one for the time it took to compute those methods (with the suffix _runtime), one for the features semected by the different methods for the different task (with suffix _top_features).


To get the main figure of the SVM performances comparison across labels:

	python3 performance_main_figure.py 

To get runtime comparison, use runtime_figure_main.py it takes every _runtime.csv file in the result folder except if you provide --manuscript_style creates the exact manuscript figure(requires citeseq.h5ad, pbmc_log.h5ad, sarc_ba.h5ad, proteomics_nonan.h5ad, recount_log_normalized_hvg)

	python3 runtime_figure_main.py --output [output_filename]
or
	python3 runtime_figure_main.py --output [output_filename] --manuscript_style
	
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
