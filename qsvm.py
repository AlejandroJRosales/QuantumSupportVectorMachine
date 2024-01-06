import numpy as np
# plotting libs
import matplotlib.pyplot as plt
# dataset libs
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.datasets import ad_hoc_data
# qml libs
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
# SVC lib
from sklearn.svm import SVC


class QSVM:
    def __init__(self):
        # init quantum kernel
        quantum_kernel()

    def fit(test=False):
        self.fit_kernel()
        if test:
            self.test_fit()

    def quantum_kernel(self):
        # 2-qubit ZZ feature mapping
        self.adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
        self.sampler = Sampler()
        self.fidelity = ComputeUncompute(sampler=sampler)
        self.adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)

    def fit_kernel(self):
        self.adhoc_svc = SVC(kernel=self.adhoc_kernel.evaluate)
        self.adhoc_svc.fit(train_features, train_labels)

    def test_fit(self):
        adhoc_score_callable_function = self.adhoc_svc.score(test_features, test_labels)
        print(f"Callable kernel classification test score: {adhoc_score_callable_function}")


class Data:
    def __init__(self):
        # initialize seed for reproducibility
        algorithm_globals.random_seed = 12345

        # initalize training and testing data
        self.adhoc_dimension = 2
        self.train_features, self.train_labels, self.test_features, self.test_labels, self.adhoc_total = ad_hoc_data(
            training_size=20,
            test_size=5,
            n=self.adhoc_dimension,
            gap=0.3,
            plot_data=False,
            one_hot=False,
            include_sample_total=True,
        )

    @staticmethod
    def plot_features(ax, features, labels, class_label, marker, face, edge, label):
        # A train plot
        ax.scatter(
            # x coordinate of labels where class is class_label
            features[np.where(labels[:] == class_label), 0],
            # y coordinate of labels where class is class_label
            features[np.where(labels[:] == class_label), 1],
            marker=marker,
            facecolors=face,
            edgecolors=edge,
            label=label,
        )


    def plot_dataset(self):
        plt.figure(figsize=(5, 5))
        plt.ylim(0, 2 * np.pi)
        plt.xlim(0, 2 * np.pi)
        plt.imshow(
            np.asmatrix(self.adhoc_total).T,
            interpolation="nearest",
            origin="lower",
            cmap="RdBu",
            extent=[0, 2 * np.pi, 0, 2 * np.pi],
        )

        # A train plot
        self.plot_features(plt, self.train_features, self.train_labels, 0, "s", "w", "b", "A train")
        # B train plot
        self.plot_features(plt, self.train_features, self.train_labels, 1, "o", "w", "r", "B train")
        # A test plot
        self.plot_features(plt, self.test_features, self.test_labels, 0, "s", "b", "w", "A test")
        # B test plot
        self.plot_features(plt, self.test_features, self.test_labels, 1, "o", "r", "w", "B test")

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.title("Ad hoc dataset")

        plt.show()


data = Data()
data.plot_dataset()

qsvm = QSVM()
qsvm.fit(test=True)