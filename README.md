# **AcroRL: Learning Aggressive Quadrotor Inversion using Bidirectional Thrust**
[![Paper](https://img.shields.io/badge/paper-AcroRL-blueviolet.svg)](https://arxiv.org/abs/2605.24301)
[![YouTube](https://img.shields.io/badge/video-YouTube-red.svg)](https://youtu.be/7pPTKY5KKtU)
[![Website](https://img.shields.io/badge/webpage-RISLabWebsite-green.svg)](https://rislab.org/projects/2026-05-14-acro-rl.html)

AcroRL is a simulation framework for learning aggressive quadrotor inversion maneuvers using bidirectional thrust and geometric control.

<table>
<tr>
<td><img src="gifs/nti.gif"></td>
<td><img src="gifs/itn.gif"></td>
</tr>
</table>

---

## **Key Components**

1. **Reference Modulation Policy $(\boldsymbol{\pi})$**  
   - Learned reference modulation policy for aggressive inversion maneuvers leveraging bidirectional thrust.
2. **Hopf Fibration-Based Geometric Control (HFCA)**  
   - Open-source implementation of a Hopf-fibration-based geometric controller with custom inversion logic.
3. **Thrust Model $(T(\Omega))$**  
   - Steady-state and stochastic transient thrust model for asymmetric propellers.
4. **Optimal Control Allocation (OCA)**  
   - Projected gradient descent (PGD)-based optimal control allocation under actuator constraints.
---

## **Installation**

Install mamba [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

Installation tested on Ubuntu 20.04 and 22.04.

If using a CUDA version other than 12, update the
`jax[cuda=XX]` dependency in `environment.yaml`
to match your local CUDA installation.

```bash
git clone git@github.com:rislab/acrorl.git
cd acrorl

mamba env create -f environment.yaml
mamba activate acrorl

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
python3 scripts/eval_control.py --inversion_method step_oca --inversion_type nti --save_plots --num_drones 20 --randomize_reset --show_hud
```

### **Evaluate Reference Controller with Learned Reference Modulation**

Single nominal-to-inverted transition example usage:

```bash
python3 scripts/eval_learned_control.py --policy_name nti_final --inversion_type nti --save_plots
```

### **Export Learned Reference Modulation Policy to ONNX**

Example export usage:

```bash
python3 export.py --policy-path /policies/my_policy --output ../policies/my_policy.onnx --verify
```

---

## **Citation**

```bibtex
@misc{rodriguez2026acrorl,
      title={AcroRL: Learning Aggressive Quadrotor Inversion using Bidirectional Thrust}, 
      author={Gabriel Rodriguez and Henri Sayag and Abhishek Rathod and John Stecklein and Siddharth Saha and Christopher Barngrover and Wennie Tabib},
      year={2026},
      eprint={2605.24301},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.24301}, 
}
```

If you use this repository in academic work, please also cite the original
developers of the `flightning` simulator:

```bibtex
@inproceedings{flightning,
    title={Learning Quadrotor Control From Visual Features Using Differentiable Simulation},
    author={Johannes Heeg and Yunlong Song and Davide Scaramuzza},
    year={2025},
    booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
    url={https://arxiv.org/abs/2410.15979},
}
```

Forked from [rpg_flightning](https://github.com/uzh-rpg/rpg_flightning).

Adapted animation module from [pyplot3d](https://github.com/kanishkegb/pyplot-3d).