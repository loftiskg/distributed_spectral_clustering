from pyspark.sql.functions import udf
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.sql.window import *
import pyspark.sql.types as t
import numpy as np
from pyspark.ml.linalg import VectorUDT
from pyspark.mllib.linalg.distributed import RowMatrix


class SpectralClustering:
    def __init__(
        self,
        k=2,
        k_nearest=7,
        num_eigenvectors=20,
        featureCol="features",
        predictionCol="predictions",
        distance="cosine",
    ):
        self.k = k
        self.k_nearest = k_nearest
        self.num_eigenvectors = num_eigenvectors
        self.featureCol = featureCol
        self.predictionCol = predictionCol

        if distance not in ["cosine", "euclidean"]:
            raise ValueError('distance must either be "cosine" or "euclidean"')
        distances = {
            "cosine": cosine_similarity_udf,
            "euclidean": euclidean_distance_udf,
        }
        self.distance = distances[distance]

    def cluster(self, df, session, repartition_num=8):
        n = df.count()
        # index rows
        df_index = df.select(
            (
                row_number().over(Window.partitionBy(lit(0)).orderBy(self.featureCol))
                - 1
            ).alias("id"),
            "*",
        )
        df_features = df_index.select("id", self.featureCol)

        # prep for joining
        df_features = df_features.repartitionByRange(repartition_num, "id")

        left_df = df_features.select(
            df_features["id"].alias("left_id"),
            df_features[self.featureCol].alias("left_features"),
        )
        right_df = df_features.select(
            df_features["id"].alias("right_id"),
            df_features[self.featureCol].alias("right_features"),
        )

        # join on self where left_id does not equal right_id
        joined_df = left_df.join(right_df, left_df["left_id"] != right_df["right_id"])

        # comupte cosine similarity between vectors
        joined_df = joined_df.select(
            "left_id",
            "right_id",
            self.distance(
                array(joined_df["left_features"], joined_df["right_features"])
            ).alias("norm"),
        )
        ranked = joined_df.select(
            "left_id",
            "right_id",
            rank().over(Window.partitionBy("left_id").orderBy("norm")).alias("rank"),
        )
        knn = ranked.where(ranked["rank"] <= 5)
        knn_grouped = knn.groupBy("left_id").agg(f.collect_list("right_id").alias("nn"))

        # generate laplacian
        laplacian = knn_grouped.select(
            knn_grouped["left_id"].alias("id"),
            toVector_udf(
                laplacian_vector_udf(
                    knn_grouped["left_id"],
                    knn_grouped["nn"],
                    lit(n),
                    lit(self.k_nearest),
                )
            ).alias("lap_vector"),
        )

        laplacian_matrix = RowMatrix(
            laplacian.select("lap_vector").rdd.map(lambda x: x[0].toArray()), -1, -1
        )

        svd = laplacian_matrix.computeSVD(
            k=laplacian_matrix.numRows()  # self.num_eigenvectors
        )
        eigenvectors = [
            (idx, Vectors.dense([float(item) for item in row]))
            for idx, row in enumerate(svd.V.toArray()[:, -self.k :].tolist())
        ]

        eigen_df = session.createDataFrame(eigenvectors, ["id", self.featureCol])
        model = KMeans(
            featuresCol=self.featureCol, predictionCol=self.predictionCol, k=self.k
        ).fit(eigen_df)
        predictions = model.transform(eigen_df).join(df_index, on="id")
        return predictions


def cosine_similarity(arr):
    arr[0], arr[1] = np.array(arr[0]), np.array(arr[1])
    return float(
        arr[0].dot(arr[1]) / ((arr[0].dot(arr[0]) ** 0.5) * (arr[1].dot(arr[1]) ** 0.5))
    )


def euclidean_distance(arr):
    arr[0], arr[1] = np.array(arr[0]), np.array(arr[1])
    return float((arr[0].dot(arr[1])) ** 2)


def laplacian_vector(row_id, arr, size, k):
    lap_vec = np.zeros(size, dtype=int)
    lap_vec[np.array(arr)] = 1
    lap_vec[row_id] = -k
    return list([int(item) for item in lap_vec])


def toVector(array):
    return Vectors.dense([float(item) for item in array])


toVector_udf = udf(toVector, VectorUDT())
cosine_similarity_udf = udf(cosine_similarity, t.DoubleType())
euclidean_distance_udf = udf(euclidean_distance, t.DoubleType())

laplacian_vector_udf = udf(laplacian_vector, t.ArrayType(t.IntegerType()))
