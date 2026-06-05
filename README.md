# dpdPy

**dpdPy** is a Python package for the numerical implementation and optimization of **Digital Predistortion (DPD)** models aimed at compensating nonlinear effects in **Analog Radio-over-Fiber (A-RoF)** communication systems and wireless systems employing **Power Amplifiers (PAs)**.

The package provides a framework for the development, evaluation, and optimization of DPD algorithms, supporting multiple model architectures, including:

* **Memory Polynomial (MP)** models;
* **Neural network-based** architectures, such as **ARVTDNN** and **ETDNN**;
* **Kolmogorov-Arnold Network (KAN)**-based architectures, such as **ETDKAN**.

In addition to DPD implementations, dpdPy includes modules for:

* Power amplifier (PA) modeling;
* Analog Radio-over-Fiber (A-RoF) system modeling;
* Transmission performance evaluation;
* Computational complexity analysis.

The package provides tools for calculating key communication system metrics, including:

* **Adjacent Channel Leakage Ratio (ACLR)**;
* **Peak-to-Average Power Ratio (PAPR)**;
* Computational complexity metrics for DPD models.

Developed entirely in **Python**, dpdPy integrates seamlessly with widely adopted scientific computing and deep learning libraries, including:

* NumPy
* SciPy
* PyTorch

It also supports integration with the **OptiCommPy** (https://github.com/edsonportosilva/OptiCommPy) optical communication system simulation framework, enabling end-to-end evaluation of communication systems under realistic conditions.

Through its integration with **Optuna**, dpdPy enables automated hyperparameter optimization for all supported DPD models using **multi-objective optimization** strategies. Users can simultaneously optimize transmission performance metrics and computational complexity, facilitating the selection of model configurations that best satisfy the requirements of different deployment scenarios.

By bringing together modeling, simulation, evaluation, and optimization capabilities within a single environment, dpdPy provides researchers and telecommunications engineers with a complete platform for the design, analysis, testing, and optimization of Digital Predistortion systems. This comprehensive toolset enables the exploration of different DPD architectures and operating conditions, supporting both academic research and practical system development.

<img class="center" src="https://github.com/daniel7fontes/DPD_for_ARoF/blob/main/dpd/capa_repo.png" width="800">
