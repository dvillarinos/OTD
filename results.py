import pandas as pd

def make_tables(df):
    df = df.copy()
    df["objective_value"] = pd.to_numeric(df["objective_value"], errors="coerce")
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")

    for instance, df_inst in df.groupby("instance"):

        print("\n" + "=" * 80)
        print(f"INSTANCIA: {instance}")
        print("=" * 80)

        # Buscar baseline
        baseline_row = df_inst[df_inst["method"] == "Actual"]

        if baseline_row.empty:
            print("⚠️  No existe baseline 'Actual' para esta instancia.")
            continue

        baseline = baseline_row.iloc[0]
        base_obj = baseline["objective_value"]
        base_time = baseline["time_s"]

        rows = []

        for _, r in df_inst.iterrows():
            rows.append({
                "Method": r["method"],
                "Settings": r["settings"],
                "Obj": r["objective_value"],
                "Time (s)": r["time_s"],
                "Δ Obj": r["objective_value"] - base_obj,
                "Δ Obj (%)": 100 * (r["objective_value"] - base_obj) / base_obj if base_obj else None,
                "Time / Actual": r["time_s"] / base_time if base_time else None
            })

        table = pd.DataFrame(rows)

        # Ordenar por mejor objetivo
        table = table.sort_values("Obj", ascending=False)

        # Redondear para mejor lectura
        table["Time (s)"] = table["Time (s)"].round(3)
        table["Δ Obj (%)"] = table["Δ Obj (%)"].round(2)
        table["Time / Actual"] = table["Time / Actual"].round(2)

        # Mostrar tabla
        print(table.to_string(index=False))
