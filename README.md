# ArchPower
ArchPower: Dataset for Architecture-Level Power Modeling of Modern CPU Design

Power is the primary design objective of large-scale integrated circuits (ICs), especially for complex modern processors (i.e., CPUs). Accurate CPU power evaluation requires designers to go through the whole time-consuming IC implementation process, easily taking months. At the early design stage (e.g., architecture-level), classical power models are notoriously inaccurate. Recently, ML-based architecture-level power models have been proposed to boost accuracy, but the data availability is a severe challenge. Currently, there is no open-source dataset for this important ML application. A typical dataset generation process involves correct CPU design implementation and repetitive execution of power simulation flows, requiring significant design expertise, engineering effort, and execution time. Even private in-house datasets often fail to reflect realistic CPU design scenarios. 

In this work, we propose ArchPower, the first open-source dataset for architecture-level processor power modeling. We go through complex and realistic design flows to collect the CPU architectural information as features and the ground-truth simulated power as labels. Our dataset includes 200 CPU data samples, collected from 25 different CPU configurations when executing 8 different workloads. There are more than 100 architectural features in each data sample, including both hardware and event parameters. The label of each sample provides fine-grained power information, including the total design power and the power for each of the 11 components. Each power value is further decomposed into four fine-grained power groups: combinational logic power, sequential logic power, memory power, and clock power.

Huggingface link: https://huggingface.co/datasets/zqj23333/ArchPower (However, we recommend using the dataset in numpy format on github)

## Quick Start
To evaluate four ML-based architecture-level power models on ArchPower dataset, you can run the instructions below.
```
cd src
python McPAT-Calib.py
python McPAT-Calib-Component.py
python McPAT-Calib-CompGroup.py
python PANDA.py
```

