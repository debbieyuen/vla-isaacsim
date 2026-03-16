# Empathic Robotic VLAs

## Requirements
  * [Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)

## Setup 

Download Isaac Sim 5.1.0. We installed via Workstation. You can use `pip install isaacsim[compatibility-check]` to install a minimal setup for the Compatibility Checker app instead of installing the full version.

Next, run the `isaacsim isaacsim.exp.compatibility_check` command.

From your folder, start Isaac Sim
```bash

.\isaac-sim.selector.bat 

```

Run experimentatl scripts using: 
```bash

./python.sh scripts/test_sim.py

```