## Distributed Spectral Cluster with PySpark

Run `python main.py` at the terminal to apply clustering to both testing datasets int `./data`.  This will output a scatter plot into `./figures/spectral_cluster_test.png` which will show a scatter plot of the test datasets after clustering.

To regenerate the test dataset run `python generate_test_data.py` from the commandline.

The code for the clustering algorithm lives in `spectral.py`.