# **AcroRL: Learning Aggressive Quadrotor Inversion using Bidirectional Thrust**
[![Paper](https://img.shields.io/badge/paper-AcroRL-blueviolet.svg)]()
[![YouTube](https://img.shields.io/badge/video-RISLabYouTube-red.svg)](https://youtu.be/7pPTKY5KKtU)
[![Website](https://img.shields.io/badge/website-RISLabWebsite-green.svg)]()

<table>
<tr>
<td><img src="gifs/nti.gif"></td>
<td><img src="gifs/itn.gif"></td>
</tr>
</table>

---

## **Key Components**
1. **Reference Modulation Policy $(\boldsymbol{\pi})$**  
   - We train a learned reference modulation policy for aggressive inversion maneuvers leveraging bidirectional thrust.
3. **Hopf Fibration-Based Geometric Control (HFCA)**  
   - We implement and open source an implementation of the Hopf-Fibration Based Geometric Controller with custom inversion logic.
2. **Thrust Model $(T(\Omega))$**  
   - We devise and implement a steady-state and stochastic transient thrust model for asymmetric propellers.
3. **Optimal Control Allocation (OCA)**  
   - We use projected gradient descent (PGD) to perform optimal control allocation under actuator constraints at each dynamics step.

---

## **Installation**

Install mamba [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

Installation tested on Ubuntu 20.04 and 22.04. If you are using CUDA != 12, edit `environment.yaml` to reflect that, updating:`jax[cuda=XX]`, where XX is your cuda version. 

```bash
git clone git@github.com:rislab/acrorl.git
cd acrorl

mamba env create -f environment.yaml
mamba activate flightning

pip install --use-pep517 -e .
```

---

## **Usage**

For all scripts, run `python3 scripts/{script} --help` for a full list of options.

### **Training**

```bash 
python3 scripts/train.py --inversion_type nti --save_plots --save_data
```

### **Evaluate Reference Controller with Different Trajectories**

Parallel nominal-to-inverted step command transition example usage:
```bash
python3 scripts/eval_control.py --inversion_method step_oca --inversion_type nti --save_plots --num_drones 20 --randomize_reset  --show_hud 
```

### **Evaluate Reference Controller with Learned Reference Modulation**

Single nominal-to-inverted transition example usage:
```bash
python3 scripts/eval_learned_control.py --policy_name nti_final --inversion_type nti  --save_plots 
```

### **Export Learned Reference Modulation Policy to ONNX**

Example exporting usage:
```bash
python3 export.py --policy-path /policies/my_policy --output ../policies/my_policy.onnx --verify
```
---

## **Citation**
```
@misc{acrorl,
  author={},
  title={}, 
  year={2026},
  url={}
}
```
Please also cite the original developers of the `flightning` simulator.
```
@misc{flightning,
    title={Learning Quadrotor Control From Visual Features Using Differentiable Simulation}, 
    author={Johannes Heeg and Yunlong Song and Davide Scaramuzza},
    year={2025},
    booktitle={IEEE International Conference on Robotics and Automation, 2025}
    url={https://arxiv.org/abs/2410.15979}, 
}
```

Forked from [rpg_flightning](https://github.com/uzh-rpg/rpg_flightning).

Adapted animation module from [pyplot3d](https://github.com/kanishkegb/pyplot-3d).