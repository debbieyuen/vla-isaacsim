# Empathic Robotic VLAs

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

## Resources
  * [How to add my own usd into isaac sim?](https://forums.developer.nvidia.com/t/how-to-add-my-own-usd-into-isaac-sim/251205)
  * [Isaac Sim - How to import USD assets into a scene?](https://forums.developer.nvidia.com/t/isaac-sim-how-to-import-usd-assets-into-a-scene/201253/4)