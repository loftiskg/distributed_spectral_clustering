from pyspark.sql import SparkSession
from spectral import SpectralClustering
from pyspark.sql.types import StructField, StructType, FloatType
from pyspark.ml.feature import VectorAssembler
import seaborn as sns
import matplotlib.pyplot as plt

ss = SparkSession.builder.getOrCreate()

# read in data
schema = StructType(
    [StructField("x", FloatType(), True), StructField("y", FloatType(), True)]
)
globular_data = ss.read.csv("./data/globular_test_data.csv", schema)
moons_data = ss.read.csv("./data/moons_test_data.csv", schema)

# process data for clustering
va = VectorAssembler(inputCols=["x", "y"], outputCol="features")
globular_data_processed = va.transform(globular_data)
moons_data_processed = va.transform(moons_data)

# initialize clustering object
cluster_k3 = SpectralClustering(k=3, distance="euclidean")
cluster_k2 = SpectralClustering(k=2, distance="euclidean")

df_globular = (
    cluster_k3.cluster(globular_data_processed, ss)
    .select(["x", "y", "predictions"])
    .toPandas()
)

df_moons = (
    cluster_k2.cluster(moons_data_processed, ss)
    .select(["x", "y", "predictions"])
    .toPandas()
)

# plot data
plt.figure(figsize=(10, 5), dpi=150)
plt.subplot(121)
sns.scatterplot(x="x", y="y", hue="predictions", data=df_globular)
plt.subplot(122)
sns.scatterplot(x="x", y="y", hue="predictions", data=df_moons)
plt.savefig("./figures/spectral_cluster_test.png")
print("Figure saved in ./figures/spectral_cluster_test.png")
