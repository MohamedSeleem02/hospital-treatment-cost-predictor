import os
import sys
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


def resource_path(filename):
    """Return the path to a bundled file, including PyInstaller one-file mode."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)


MODEL_PATH = resource_path("model_file.joblib")


try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    GENDER_OPTIONS = bundle["gender_encoder"].classes_.tolist()
    DEPARTMENT_OPTIONS = bundle["onehot_encoder"].categories_[0].tolist()
    DIAGNOSIS_OPTIONS = bundle["onehot_encoder"].categories_[1].tolist()
    SEVERITY_OPTIONS = bundle["severity_encoder"].categories_[0].tolist()
    INSURANCE_OPTIONS = bundle["onehot_encoder"].categories_[2].tolist()
    OUTCOME_OPTIONS = bundle["onehot_encoder"].categories_[3].tolist()

    AGE_MIN, AGE_MAX = map(int, bundle["domain_bounds"]["Age"])
    LOS_MIN, LOS_MAX = map(int, bundle["domain_bounds"]["Length_of_Stay_Days"])
    WAIT_MIN, WAIT_MAX = map(int, bundle["domain_bounds"]["Wait_Time_Minutes"])

except Exception as exc:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Startup Error",
        f"Could not load the model.\n\n{exc}\n\n"
        "Make sure model_file.joblib is next to the application."
    )
    root.destroy()
    raise


def predict_cost(values):
    input_data = pd.DataFrame([values])

    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]
    nominal_cols = bundle["nominal_cols"]

    X_input = input_data.copy()

    X_input[numeric_cols] = bundle["numeric_imputer"].transform(
        X_input[numeric_cols]
    )
    X_input[categorical_cols] = bundle["categorical_imputer"].transform(
        X_input[categorical_cols]
    )

    X_input[["Severity_Level"]] = bundle["severity_encoder"].transform(
        X_input[["Severity_Level"]]
    )

    X_input["Gender"] = bundle["gender_encoder"].transform(
        X_input["Gender"]
    )

    onehot = pd.DataFrame(
        bundle["onehot_encoder"].transform(X_input[nominal_cols]),
        columns=bundle["onehot_encoder"].get_feature_names_out(nominal_cols),
        index=X_input.index,
    )

    X_input = pd.concat(
        [X_input.drop(columns=nominal_cols), onehot],
        axis=1,
    )

    X_input[numeric_cols] = bundle["scaler"].transform(
        X_input[numeric_cols]
    )

    X_input = X_input[bundle["feature_columns"]]

    prediction = model.predict(X_input)

    if bundle["uses_target_scaling"]:
        prediction = bundle["target_scaler"].inverse_transform(
            prediction.reshape(-1, 1)
        ).ravel()

    return float(prediction[0])


class HospitalCostApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hospital Treatment Cost Predictor")
        self.root.geometry("760x780")
        self.root.minsize(680, 700)
        self.root.configure(bg="#0f1117")

        self._setup_style()
        self._build_ui()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "App.TFrame",
            background="#0f1117",
        )
        style.configure(
            "Card.TFrame",
            background="#171a23",
        )
        style.configure(
            "Title.TLabel",
            background="#0f1117",
            foreground="#ffffff",
            font=("Segoe UI", 25, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#0f1117",
            foreground="#aeb4c0",
            font=("Segoe UI", 11),
        )
        style.configure(
            "Field.TLabel",
            background="#171a23",
            foreground="#e8eaf0",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Value.TLabel",
            background="#171a23",
            foreground="#ffffff",
            font=("Segoe UI", 10),
        )
        style.configure(
            "TCombobox",
            fieldbackground="#242834",
            background="#242834",
            foreground="#ffffff",
            arrowcolor="#ffffff",
            padding=8,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#242834")],
            foreground=[("readonly", "#ffffff")],
        )
        style.configure(
            "TSpinbox",
            fieldbackground="#242834",
            background="#242834",
            foreground="#ffffff",
            arrowcolor="#ffffff",
            padding=8,
        )
        style.configure(
            "Predict.TButton",
            font=("Segoe UI", 12, "bold"),
            padding=12,
            background="#4f8cff",
            foreground="#ffffff",
        )
        style.map(
            "Predict.TButton",
            background=[("active", "#6a9dff")],
        )

    def _field_label(self, parent, text, row, column):
        label = ttk.Label(parent, text=text, style="Field.TLabel")
        label.grid(row=row, column=column, sticky="w", padx=12, pady=(10, 5))

    def _build_ui(self):
        main = ttk.Frame(self.root, style="App.TFrame", padding=28)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="🏥  Hospital Treatment Cost",
            style="Title.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            main,
            text="Enter patient information to estimate the treatment cost.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(5, 20))

        card = ttk.Frame(main, style="Card.TFrame", padding=18)
        card.pack(fill="both", expand=True)

        for col in range(2):
            card.columnconfigure(col, weight=1)

        self.age = tk.IntVar(value=30)
        self.gender = tk.StringVar(value=GENDER_OPTIONS[0])
        self.department = tk.StringVar(value=DEPARTMENT_OPTIONS[0])
        self.diagnosis = tk.StringVar(value=DIAGNOSIS_OPTIONS[0])
        self.severity = tk.StringVar(value=SEVERITY_OPTIONS[0])
        self.length_of_stay = tk.IntVar(value=3)
        self.wait_time = tk.IntVar(value=30)
        self.insurance = tk.StringVar(value=INSURANCE_OPTIONS[0])
        self.readmission = tk.IntVar(value=0)
        self.outcome = tk.StringVar(value=OUTCOME_OPTIONS[0])

        fields = [
            ("Age", "spin", self.age, AGE_MIN, AGE_MAX),
            ("Gender", "combo", self.gender, GENDER_OPTIONS),
            ("Department", "combo", self.department, DEPARTMENT_OPTIONS),
            ("Diagnosis", "combo", self.diagnosis, DIAGNOSIS_OPTIONS),
            ("Severity Level", "combo", self.severity, SEVERITY_OPTIONS),
            ("Length of Stay (Days)", "spin", self.length_of_stay, LOS_MIN, LOS_MAX),
            ("Wait Time (Minutes)", "spin", self.wait_time, WAIT_MIN, WAIT_MAX),
            ("Insurance Type", "combo", self.insurance, INSURANCE_OPTIONS),
            ("Readmission", "combo", self.readmission, ["No", "Yes"]),
            ("Outcome", "combo", self.outcome, OUTCOME_OPTIONS),
        ]

        self.readmission_combo = None

        for i, field in enumerate(fields):
            row = i // 2
            col = i % 2
            name, kind, variable, *options = field

            self._field_label(card, name, row * 2, col)

            if name == "Readmission":
                self.readmission_combo = ttk.Combobox(
                    card,
                    textvariable=tk.StringVar(value="No"),
                    values=["No", "Yes"],
                    state="readonly",
                )
                self.readmission_display_var = self.readmission_combo.cget("textvariable")
                self.readmission_combo.grid(
                    row=row * 2 + 1,
                    column=col,
                    sticky="ew",
                    padx=12,
                    pady=(0, 10),
                )
            elif kind == "combo":
                widget = ttk.Combobox(
                    card,
                    textvariable=variable,
                    values=options[0],
                    state="readonly",
                )
                widget.grid(
                    row=row * 2 + 1,
                    column=col,
                    sticky="ew",
                    padx=12,
                    pady=(0, 10),
                )
            else:
                widget = ttk.Spinbox(
                    card,
                    textvariable=variable,
                    from_=options[0],
                    to=options[1],
                    increment=1,
                )
                widget.grid(
                    row=row * 2 + 1,
                    column=col,
                    sticky="ew",
                    padx=12,
                    pady=(0, 10),
                )

        ttk.Separator(main).pack(fill="x", pady=18)

        self.predict_button = ttk.Button(
            main,
            text="Predict Treatment Cost",
            style="Predict.TButton",
            command=self._predict,
        )
        self.predict_button.pack(fill="x")

        self.result_frame = tk.Frame(
            main,
            bg="#171a23",
            highlightthickness=1,
            highlightbackground="#303646",
        )
        self.result_frame.pack(fill="x", pady=(18, 0))

        tk.Label(
            self.result_frame,
            text="PREDICTED TREATMENT COST",
            bg="#171a23",
            fg="#aeb4c0",
            font=("Segoe UI", 10, "bold"),
        ).pack(pady=(16, 3))

        self.result_label = tk.Label(
            self.result_frame,
            text="—",
            bg="#171a23",
            fg="#62e6a8",
            font=("Segoe UI", 28, "bold"),
        )
        self.result_label.pack(pady=(0, 16))

    def _predict(self):
        try:
            readmission_value = 1 if self.readmission_combo.get() == "Yes" else 0

            values = {
                "Age": int(self.age.get()),
                "Gender": self.gender.get(),
                "Department": self.department.get(),
                "Diagnosis": self.diagnosis.get(),
                "Severity_Level": self.severity.get(),
                "Length_of_Stay_Days": int(self.length_of_stay.get()),
                "Wait_Time_Minutes": int(self.wait_time.get()),
                "Insurance_Type": self.insurance.get(),
                "Readmission_Flag": readmission_value,
                "Outcome": self.outcome.get(),
            }

            prediction = predict_cost(values)

            self.result_label.config(text=f"${prediction:,.2f}")

        except Exception as exc:
            messagebox.showerror(
                "Prediction Error",
                f"Could not calculate the prediction.\n\n{exc}",
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = HospitalCostApp(root)
    root.mainloop()
