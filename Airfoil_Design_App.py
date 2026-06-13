import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import aerosandbox as asb
import os

# --- 1. Neural Network Architecture ---
class AeroNetMulti(nn.Module):
    def __init__(self):
        super(AeroNetMulti, self).__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

# --- 2. GUI Application Class ---
class AirfoilDesignerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Airfoil Design Suite")
        
        # Configuration
        self.XFOIL_PATH = r"C:\Users\mohta\Desktop\python\XFOIL\xfoil.exe"
        self.model_path = "aero_model_multi.pth"
        
        # 1. Verification: Check if files exist before building GUI
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"File '{self.model_path}' not found. Please place it in the script folder.")
            root.destroy()
            return

        # 2. Load Model/Scalers
        try:
            self.model = AeroNetMulti()
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            self.scaler_X = joblib.load("scaler_X_multi.pkl")
            self.scaler_y = joblib.load("scaler_y_multi.pkl")
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to load model: {e}")
            root.destroy()
            return
        
        self.last_res = None
        
        # 3. Build UI
        tk.Label(root, text="Enter Target CL:").grid(row=0, column=0, padx=10, pady=10)
        self.entry_cl = tk.Entry(root)
        self.entry_cl.grid(row=0, column=1, padx=10, pady=10)
        
        tk.Button(root, text="Run Optimization", command=self.run_optimization).grid(row=1, column=0, columnspan=2, pady=5)
        tk.Button(root, text="Visualize Geometry & Polar", command=self.visualize_result).grid(row=2, column=0, columnspan=2, pady=5)
        
        self.res_label = tk.Label(root, text="Waiting for input...", font=("Consolas", 10))
        self.res_label.grid(row=3, column=0, columnspan=2, pady=10)

    def objective(self, p):
        # Use detached numpy array for Scipy compatibility
        input_scaled = self.scaler_X.transform([p])
        with torch.no_grad():
            pred = self.model(torch.tensor(input_scaled, dtype=torch.float32)).detach().numpy()
        
        target = float(self.entry_cl.get())
        real_cl = self.scaler_y.inverse_transform(pred)[0, 0]
        return abs(real_cl - target)

    def run_optimization(self):
        try:
            self.last_res = minimize(self.objective, [0.03, 0.4, 0.1, 5], 
                                     bounds=[(0, 0.09), (0.1, 0.9), (0.05, 0.2), (-5, 15)], 
                                     method='SLSQP')
            
            m, p, t, a = self.last_res.x
            self.res_label.config(text=f"Optimal Design:\nCamber(m): {m:.3f}\nPos(p): {p:.3f}\nThick(t): {t:.3f}\nAlpha: {a:.2f}°")
        except Exception as e:
            messagebox.showerror("Optimization Error", str(e))

    def visualize_result(self):
        if self.last_res is None: 
            messagebox.showwarning("Warning", "Run optimization first!")
            return
            
        m, p, t = self.last_res.x[0:3]
        name = f"naca{int(round(m*100))}{int(round(p*10))}{int(round(t*100)):02d}"
        airfoil = asb.Airfoil(name)
        
        # Plot Geometry
        plt.figure(figsize=(8, 3))
        airfoil.draw()
        plt.title(f"Geometry: {name}")
        
        # Plot Polar
        try:
            analysis = asb.XFoil(airfoil=airfoil, Re=1e6, xfoil_command=self.XFOIL_PATH)
            polar = analysis.alpha(alpha=np.linspace(-5, 15, 20))
            
            plt.figure(figsize=(8, 6))
            plt.plot(polar['CD'], polar['CL'], 'bo-', label='Efficiency Polar')
            plt.xlabel("Coefficient of Drag (CD)")
            plt.ylabel("Coefficient of Lift (CL)")
            plt.title(f"Aerodynamic Polar: {name}")
            plt.grid(True)
            plt.legend()
            plt.show()
        except Exception as e:
            messagebox.showerror("XFOIL Error", f"Check XFOIL path or input:\n{e}")

# --- 3. Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AirfoilDesignerApp(root)
    root.mainloop()