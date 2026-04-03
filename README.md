# Empathic Robotic VLAs

Joint types are by DOF-type grouping. Not by body-part grouping. Which means that it is better for learning symmetry and matches the simulator DOF structure. 

## Requirements
  * [Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)
  * [Newton Physics for Isaac Lab](https://developer.nvidia.com/newton-physics)

## Setup 

Download Isaac Sim 5.1.0. We installed via Workstation. You can use `pip install isaacsim[compatibility-check]` to install a minimal setup for the Compatibility Checker app instead of installing the full version.

Next, run the `isaacsim isaacsim.exp.compatibility_check` command.

From your folder, start Isaac Sim
```bash
.\isaac-sim.selector.bat 
```

Run experimental scripts using: 
```bash
./python.sh scripts/test_sim.py
```

Debbie run headless: 
```bash
.\python.bat C:\Users\debbi\Documents\VLA\vla-isaacsim\scripts\franka_robot.py
```

A successful script should open Isaac Sim.

Install the Hugging Facetools
```bash
pip install datasets huggingface_hub
```

Install NumPy
```
pip install numpy
```

To get more data, run the simulator file `h1_standalone.py`. Give it some time to walk around then close the simulator when you are finished. Then in the scripts folder, run `check_data.py` to see inspect the NumPy data outputs. 
```
python check_data.py
```

## Resources
  * [How to add my own usd into isaac sim?](https://forums.developer.nvidia.com/t/how-to-add-my-own-usd-into-isaac-sim/251205)
  * [Isaac Sim - How to import USD assets into a scene?](https://forums.developer.nvidia.com/t/isaac-sim-how-to-import-usd-assets-into-a-scene/201253/4)