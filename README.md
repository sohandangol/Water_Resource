# Water Resource

This repository contains the fundamental concepts of water resources including hydrology, hydraulics, and computational techniques. Interactive visualizations and simulations are used to explain complex equations with simple intuitions and logic.

## Repository layout

```text
WaterResource/
├── src/                       # Current, maintained applications
│   ├── solver.py              # Kinematic and diffusive wave visualizer
│   └── Kinematic_wave_equation_solver.py
├── TrialsPyCodes/             # Preserved experiments and earlier prototypes
├── requirements.txt           # Python dependencies
├── .devcontainer/             # Development-container configuration
└── .vscode/                   # Editor settings
```

`src/` is the canonical location for new maintained Python applications. Keep experiments, alternative implementations, and unfinished investigations in `TrialsPyCodes/` until they are ready to be promoted.

## Run the Streamlit visualizer

From the repository root:

```powershell
python -m pip install -r requirements.txt
streamlit run src/solver.py
```

The second application can be started with:

```powershell
streamlit run src/Kinematic_wave_equation_solver.py
```
