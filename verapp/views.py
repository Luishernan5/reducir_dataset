import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from django.shortcuts import render
from sklearn.model_selection import train_test_split
import arff   # liac-arff


def fig_to_base64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def load_kdd_dataset_from_fileobj(file_obj):
    raw = file_obj.read()
    text = raw.decode('utf-8', errors='ignore')

    parsed = arff.loads(text)
    attributes = [a[0] for a in parsed['attributes']]
    data = parsed['data']

    df = pd.DataFrame(data, columns=attributes)
    return df


def upload_file(request):
    graphs = []
    graph_titles = []
    columns = []
    rows = 0

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded = request.FILES['file']
        df = load_kdd_dataset_from_fileobj(uploaded)

        columns = df.columns.tolist()
        rows = len(df)

        # dividir el dataset
        train_set, temp_set = train_test_split(df, test_size=0.4, random_state=42)
        val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

        col = "protocol_type"
        if col in df.columns:
            for dataset, name in [
                (df, "df"),
                (train_set, "train_set"),
                (val_set, "val_set"),
                (test_set, "test_set")
            ]:
                try:
                    plt.figure(figsize=(6, 4))
                    dataset[col].value_counts().plot(kind='bar')
                    plt.title(f"Distribución de {col} en {name}")
                    graphs.append(fig_to_base64())
                    graph_titles.append(f"Distribución de {col} ({name})")

                except Exception:
                    graphs.append(None)
                    graph_titles.append(f"Distribución de {col} ({name})")

    context = {
        "graphs": zip(graphs, graph_titles),
        "columns": columns,
        "rows": rows,
    }
    return render(request, "upload.html", context)