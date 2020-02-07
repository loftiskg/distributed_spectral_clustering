import numpy as np


def generate_globular_test_clusters(n=100):
    centroids = np.array([[2.0, 4.0], [4, 2], [6, 4]])
    test_data_small = np.concatenate([centroids for i in range(n)], axis=0)
    test_data_small = test_data_small + np.random.rand(*test_data_small.shape)
    return test_data_small


def generate_moon_test_clusters(n=100):
    data = np.array(
        [[i, np.sin(i) + np.random.rand() / 2] for i in np.linspace(0, np.pi, num=n)]
    )
    data2 = np.array(
        [
            [i, np.sin(i) + np.random.rand() / 2]
            for i in np.linspace(np.pi, 2 * np.pi, num=n)
        ]
    )
    data2[:, 0] = data2[:, 0] - (np.pi / 2)
    return np.concatenate([data, data2])


if __name__ == "__main__":
    test_data_globular = generate_globular_test_clusters(n=100)
    test_data_moons = generate_moon_test_clusters(n=100)

    np.savetxt(
        "./data/globular_test_data.csv", test_data_globular, delimiter=",", fmt="%.5f"
    )
    np.savetxt("./data/moons_test_data.csv", test_data_moons, delimiter=",", fmt="%.5f")

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,5))
    # plt.subplot(121)
    # plt.title('Globular Test Data')
    # plt.scatter(test_data_globular[:,0],test_data_globular[:,1])
    # plt.subplot(122)
    # plt.title('Moon Test Data')
    # plt.scatter(test_data_moons[:,0],test_data_moons[:,1])
    # plt.show()
