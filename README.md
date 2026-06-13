# AI-Powered Airfoil Design Suite
An interactive tool that uses a Deep Learning surrogate model to predict aerodynamic performance and perform inverse design optimization.

## Features
- **Instant Inference:** Predicts $C_L$ and $C_D$ using a trained neural network.
- **Inverse Optimization:** Automatically suggests NACA parameters ($m, p, t$) to meet a user-defined Target $C_L$.
- **Validation:** Integrates with XFOIL via AeroSandbox to visualize real-world geometry and performance polars.

## Requirements
- Python 3.x
- [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/) (You must download this separately)

## Setup
1. Install dependencies:
   `pip install -r requirements.txt`
2. Update the `XFOIL_PATH` variable in `Airfoil_Design_App.py` to point to your `xfoil.exe`.
3. Run the application:
   `python Airfoil_Design_App.py`