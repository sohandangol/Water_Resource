# Water Resource

Interactive visualizations and simulations to explain complex water resource concepts, hydrology, and computational hydraulics with simple intuitions.

## Repository Layout

```text
WaterResource/
├── src/                       # Main applications and logic
│   ├── coreApp.py             # Integrated Hydrodynamics Modeling Suite
│   ├── core/                  # Core modules (GIS, routing, hydrology)
│   ├── solver.py              # Kinematic & diffusive wave visualizer
│   └── Kinematic_wave_equation_solver.py
└── requirements.txt           # Python dependencies
```

## Running the Applications

First, install the required dependencies:
```powershell
pip install -r requirements.txt
```

Then, you can launch any of the tools using Streamlit:

* **Hydrodynamics Suite (New):** `streamlit run src/coreApp.py`
* **Wave Visualizer:** `streamlit run src/solver.py`
* **Kinematic Solver:** `streamlit run src/Kinematic_wave_equation_solver.py`

*(The original wave equation solver is also hosted online at [sohandangolwave.streamlit.app](https://sohandangolwave.streamlit.app/) while the code I am currently building ie coreApp can be viewed at [on-testing-app.streamlit.app](https://on-testing-app.streamlit.app))*
