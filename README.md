# 🧠 Social Network Ads – Machine Learning Project

This project applies machine learning models to the *Social_Network_Ads dataset* to predict whether a user will purchase a product based on their *Age* and *Estimated Salary*.  
The workflow includes preprocessing, model training, hyperparameter tuning, evaluation, and insights.

---

## 📂 Project Structure
- Project.ipynb – Jupyter notebook with step-by-step analysis and visualizations.
- run_project.py – Main pipeline script (data loading, preprocessing, training, evaluation, saving artifacts).
- requirements.txt – Python dependencies.
- Social_Network_Ads.csv – Input dataset (features: Age, EstimatedSalary; target: Purchased).
- Generated artifacts:
  - model_comparison_metrics.csv – Performance metrics for all trained models.
  - prompt_scenario_predictions.csv – Predictions for predefined “what-if” scenarios.
  - controlled_predictions.csv – Predictions for controlled age/salary combinations.
  - project_summary.json – Summary of best model and performance.
  - conclusions.txt – Final conclusions and correlation findings.

---

## ⚙ Installation & Setup
1. *Clone or download* this project.
2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   # Activate:
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
▶ How to Run
Run the full pipeline from terminal:

bash
Copy code
python run_project.py
Artifacts will be saved in the same folder.

To explore the workflow interactively, open the Jupyter notebook:

bash
Copy code
jupyter notebook Project.ipynb
📊 Results
Best Model: KNN (n_neighbors=7)

Accuracy: 0.92

Precision/Recall/F1: 0.89 each

ROC-AUC: 0.94

Key Insight: Salary has a stronger positive relationship with purchase decisions than age
.

🔮 Example Predictions
Scenario-based predictions (from prompt_scenario_predictions.csv):

Age 30, Salary 87,000 → Likely to purchase

Age 40, No Salary → Salary imputed with median, moderate probability

Age 22, Salary 600,000 → High probability of purchase

Age 60, Salary 100,000,000 → Extremely high probability

Controlled experiments (from controlled_predictions.csv) further validate the influence of salary at fixed ages.
