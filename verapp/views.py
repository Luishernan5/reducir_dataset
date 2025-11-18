from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import arff
from sklearn.model_selection import train_test_split


def fig_to_base64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def load_kdd_dataset_from_fileobj(file_obj):
    # Leer archivo como texto (no bytes)
    text = file_obj.read().decode('utf-8', errors='ignore')

    # Cargar ARFF
    arff_data = arff.loads(text)

    # Convertir a DataFrame
    df = pd.DataFrame(arff_data['data'],
                      columns=[attr[0] for attr in arff_data['attributes']])
    return df


def upload_file(request):
    graphs = []
    titles = []
    rows = None
    columns = None
    df_html = None

    if request.method == "POST" and request.FILES.get("file"):
        uploaded = request.FILES["file"]

        # Cargar el dataset ARFF
        df = load_kdd_dataset_from_fileobj(uploaded)

        columns = df.columns.tolist()
        rows = len(df)

        # Mostrar primeras 20 filas
        df_html = df.head(20).to_html(
            classes="table table-striped table-bordered",
            index=False
        )

        # Separación train/val/test
        train_set, temp_set = train_test_split(df, test_size=0.4, random_state=42)
        val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

        col = "protocol_type"
        if col in df.columns:
            for data, name in [
                (df, "Dataset completo"),
                (train_set, "Train"),
                (val_set, "Validation"),
                (test_set, "Test"),
            ]:
                plt.figure(figsize=(6, 4))
                data[col].value_counts().plot(kind="bar")
                plt.title(f"Distribución de {col} en {name}")
                graphs.append(fig_to_base64())
                titles.append(f"{col} — {name}")

    return render(request, "upload.html", {
        "graphs": zip(graphs, titles),
        "columns": columns,
        "rows": rows,
        "df_html": df_html,
    })