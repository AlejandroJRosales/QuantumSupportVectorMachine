# Quantum Support Vector Machine
SVM with quantum enhanced feature spaces based on the research paper, Supervised Learning with Quantum Enhanced Feature Spaces 

Read the amazing original work of the researchers on the paper, Supervised Learning with Quantum Enhanced Feature Spaces. Which you can find here: https://arxiv.org/pdf/1804.11326.pdf

You can read a summary and analysis of the math, circuitry, and quantum mechanics behind the support vector machine with quantum-enhanced feature space on my website here: https://www.context-switching.com/tcs/quantummachinelearning/#quantum-support-vector-machine

## Installation

### Running IBM Quantum Experience Locally

**Install and Set Up Qiskit**

```
pip install qiskit
```

```
pip install qiskit-ibm-runtime
```

**Qiskit Quantum Algorithms Installation**

```
pip install qiskit-algorithms
```

**Qiskit Machine Learning Installation**

```
pip install qiskit-machine-learning
```

### Non-Qiskit Libraries

**SKLearn Machine Learning Library**

```
pip install -U scikit-learn
```

## Examples

```python
# generate data
data = Data()
data.plot_dataset()

# init quantum kernel
qsvm = QSVM(data)
# fit kernel and test fit
qsvm.fit(test=True)
```

<img src="/examples/img/adhoc_dataset-test1.png" alt="Adhoc Dataset Test 1" width="400"/>
```
Callable kernel classification test score: 1.0
```
