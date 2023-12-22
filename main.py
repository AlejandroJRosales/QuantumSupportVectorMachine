# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, execute, BasicAer
import quantum_feature_space as qfs
import qsvm


# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options

qc = QuantumCircuit(3, 3)
rand_state = qi.random_statevector(2)
telp_state = qt(rand_state, draw=True)
print(f"Statevector:\n{telp_state}")
