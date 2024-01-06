import numpy as np
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name


def quantum_variational_class():
  feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
  optimizer = SPSA(maxiter=40, c0=4.0, skip_calibration=True)
  var_form = TwoLocal(feature_dim, ['ry', 'rz'], 'cz', reps=3)
  vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, datapoints[0])
  
  backend = BasicAer.get_backend('qasm_simulator')
  quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)
  
  result = vqc.run(quantum_instance)
  return result
