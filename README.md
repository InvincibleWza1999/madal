This is the code and data for paper "Dual-view Microservice Anomaly Detection via Modality Adaptive Consistency Learning"

## Project Structure
```
.
├── dir_tree.md
├── gaia
│   ├── data.zip
│   ├── log_template.csv
│   ├── network
│   │   ├── cross_modal.py
│   │   ├── mymodel.py
│   │   ├── __pycache__
│   │   └── se_net.py
│   ├── run.bash
│   ├── train.py
│   └── utils
│       ├── log_embedding.py
│       ├── loss.py
│       ├── __pycache__
│       └── SavedDataset.py
├── nezha
│   ├── data.zip
│   ├── log_template.csv
│   ├── network
│   │   ├── cross_modal.py
│   │   ├── mymodel.py
│   │   ├── __pycache__
│   │   └── se_net.py
│   ├── run.bash
│   ├── train.py
│   └── utils
│       ├── log_embedding.py
│       ├── loss.py
│       ├── __pycache__
│       └── SavedDataset.py
└── tree.txt
```

## Dataset
1. GAIA：GAIA dataset records metrics, traces, and logs of the MicroSS simulation system in July 2021, which consists of ten microservices and some middleware such as Redis, MySQL, and Zookeeper.
2. Nezha: Nezha dataset is from the paper "Nezha: Interpretable fine-grained root causes analysis for microservices on multi-modal observability data", ESEC/FSE 2023, which consists of 9 interdependent services deployed on a Kubernetes cluster with 12 virtual machines.

## Training Procedure
1. Unzip the data.zip file
2. Run the run.bash



