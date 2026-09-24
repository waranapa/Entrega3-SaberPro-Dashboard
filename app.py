from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


print("=" * 68)
print("MODELADOR SABER PRO 98 PROFESSIONAL · VERSION 4")
print("Selector de variables activo | Puerto: 8051")
print("=" * 68)

# ============================================================
# RUTAS Y CARGA DE DATOS
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

ARCHIVOS = {
    "datos": DATA / "datos_dashboard.csv",
    "metricas": DATA / "metricas_modelos.csv",
    "coeficientes": DATA / "coeficientes_modelos.csv",
    "vif": DATA / "vif.csv",
    "correlaciones": DATA / "correlaciones.csv",
    "predicciones": DATA / "predicciones_dashboard.csv",
}

faltantes = [p.name for p in ARCHIVOS.values() if not p.exists()]
if faltantes:
    raise FileNotFoundError(
        "Faltan archivos del notebook en la carpeta data/: "
        + ", ".join(faltantes)
        + ". Copie allí los CSV generados en salidas_dashboard."
    )

DATOS = pd.read_csv(ARCHIVOS["datos"])
METRICAS = pd.read_csv(ARCHIVOS["metricas"])
COEF = pd.read_csv(ARCHIVOS["coeficientes"])
VIF = pd.read_csv(ARCHIVOS["vif"])
PRED = pd.read_csv(ARCHIVOS["predicciones"])
CORR = pd.read_csv(ARCHIVOS["correlaciones"], index_col=0)

# ============================================================
# ETIQUETAS Y PALETA WINDOWS 98
# ============================================================
ETIQUETAS = {
    "lectura": "Lectura Crítica",
    "ciudadanas": "Competencias Ciudadanas",
    "ingles": "Inglés",
    "comunicacion": "Comunicación Escrita",
    "razonamiento": "Razonamiento Cuantitativo",
}
PREDICTORES = ["lectura", "ciudadanas", "ingles", "comunicacion"]

C = {
    "azul": "#0A246A",
    "azul2": "#3A6EA5",
    "azul3": "#7FA0C8",
    "cafe": "#7A5C3E",
    "cafe2": "#A7845F",
    "gris": "#808080",
    "gris_oscuro": "#404040",
    "gris_medio": "#B7B7B7",
    "gris_claro": "#D4D0C8",
    "crema": "#E6E2D8",
    "blanco": "#FFFFFF",
    "rojo": "#9D3333",
}

app = Dash(__name__, title="Modelador Saber Pro 98")
server = app.server

# ============================================================
# FUNCIONES DE ESTILO
# ============================================================
def aplicar_estilo_win98(fig: go.Figure, titulo: str | None = None) -> go.Figure:
    """Aplica un estilo visual tipo Windows 98 a una figura Plotly."""
    fig.update_layout(
        paper_bgcolor=C["gris_claro"],
        plot_bgcolor=C["crema"],
        font=dict(
            family="Tahoma, Verdana, Arial, sans-serif",
            size=12,
            color="#1F1F1F",
        ),
        title=dict(
            text=titulo if titulo is not None else fig.layout.title.text,
            font=dict(
                family="Tahoma, Verdana, Arial, sans-serif",
                size=16,
                color=C["azul"],
            ),
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=55, r=25, t=62, b=55),
        legend=dict(
            bgcolor=C["gris_claro"],
            bordercolor=C["gris"],
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor=C["blanco"],
            bordercolor=C["gris_oscuro"],
            font=dict(
                family="Tahoma, Verdana, Arial, sans-serif",
                color="#111111",
            ),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#B9B6AF",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=C["gris_oscuro"],
        linewidth=1,
        ticks="outside",
        tickcolor=C["gris_oscuro"],
        title_font=dict(color=C["azul"]),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#B9B6AF",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=C["gris_oscuro"],
        linewidth=1,
        ticks="outside",
        tickcolor=C["gris_oscuro"],
        title_font=dict(color=C["azul"]),
    )

    return fig


def grafica_vacia(titulo="Sin datos para mostrar"):
    fig = go.Figure()
    fig.add_annotation(
        text=titulo,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=15, color=C["gris_oscuro"]),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return aplicar_estilo_win98(fig, titulo)


def tarjeta(titulo, identificador, nota):
    return html.Div(
        [
            html.Div(titulo, className="kpi-titlebar"),
            html.Div(
                [
                    html.H2(id=identificador),
                    html.P(nota, className="kpi-note"),
                ],
                className="kpi-body",
            ),
        ],
        className="kpi-card",
    )



def figura_estructura_modelo(variables, nombre_modelo):
    """Diagrama simple de la especificación seleccionada."""
    fig = go.Figure()

    if not variables:
        return grafica_vacia("Seleccione variables para el modelo")

    posiciones = np.linspace(0.82, 0.18, len(variables))

    # Resultado
    fig.add_shape(
        type="rect",
        x0=0.66, x1=0.96,
        y0=0.38, y1=0.62,
        fillcolor="#F2F2F2",
        line=dict(color=C["azul"], width=2),
    )
    fig.add_annotation(
        x=0.81, y=0.50,
        text="<b>Razonamiento<br>Cuantitativo</b>",
        showarrow=False,
        font=dict(size=13, color="#000000"),
        align="center",
    )

    for variable, ypos in zip(variables, posiciones):
        etiqueta = ETIQUETAS.get(variable, variable)

        fig.add_shape(
            type="rect",
            x0=0.04, x1=0.34,
            y0=ypos - 0.075, y1=ypos + 0.075,
            fillcolor="#E8E8E8",
            line=dict(color="#606060", width=1.5),
        )
        fig.add_annotation(
            x=0.19, y=ypos,
            text=f"<b>{etiqueta}</b>",
            showarrow=False,
            font=dict(size=11, color="#000000"),
            align="center",
        )

        fig.add_annotation(
            x=0.66, y=0.50,
            ax=0.34, ay=ypos,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=C["azul"],
            text="",
        )

    fig.update_layout(
        title=dict(
            text=f"Estructura: {nombre_modelo}",
            x=0.5,
            xanchor="center",
            font=dict(
                family="Tahoma, Verdana, Arial, sans-serif",
                size=15,
                color="#000000",
            ),
        ),
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(range=[0, 1], visible=False),
        paper_bgcolor="#F2F2F2",
        plot_bgcolor="#F2F2F2",
        margin=dict(l=20, r=20, t=55, b=20),
        height=max(360, 95 * len(variables)),
    )
    return fig


def construir_modelo_exploratorio(tipo, alpha):
    """Devuelve el estimador solicitado para el explorador."""
    alpha = float(alpha)

    if tipo == "lineal":
        return LinearRegression()

    if tipo == "ridge":
        return Pipeline([
            ("escala", StandardScaler()),
            ("modelo", Ridge(alpha=alpha)),
        ])

    if tipo == "lasso":
        return Pipeline([
            ("escala", StandardScaler()),
            ("modelo", Lasso(alpha=alpha, max_iter=20000)),
        ])

    return Pipeline([
        ("grado2", PolynomialFeatures(degree=2, include_bias=False)),
        ("escala", StandardScaler()),
        ("modelo", Ridge(alpha=alpha)),
    ])


# ============================================================
# LAYOUT
# ============================================================
app.layout = html.Div(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("▣", className="app-icon"),
                                html.Span(
                                    "Modelador Saber Pro 98 Professional · v4",
                                    className="app-caption",
                                ),
                            ],
                            className="titlebar-left",
                        ),
                        html.Div(
                            [
                                html.Span("_", className="caption-button"),
                                html.Span("□", className="caption-button"),
                                html.Span("×", className="caption-button"),
                            ],
                            className="caption-buttons",
                        ),
                    ],
                    className="program-titlebar",
                ),
                html.Div(
                    [
                        html.Span("Archivo"),
                        html.Span("Modelo"),
                        html.Span("Ver"),
                        html.Span("Resultados"),
                        html.Span("Ayuda"),
                    ],
                    className="program-menu",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "Saber Pro 2024 · Razonamiento Cuantitativo",
                                    className="workspace-title",
                                ),
                                html.Div(
                                    "Evaluación, comparación y especificación de modelos predictivos",
                                    className="workspace-subtitle",
                                ),
                            ]
                        ),
                        html.Div("PROFESSIONAL v4 · PORT 8051", className="status-badge"),
                    ],
                    className="workspace-header",
                ),
            ],
            className="hero",
        ),
        html.Div(
            [
                html.Aside(
                    [
                        html.Div("MODELO", className="window-titlebar model-main-title"),
                        html.Div(
                            [
                                html.Div(
                                    "1. Seleccione el tipo de modelo",
                                    className="sidebar-step",
                                ),
                                dcc.Dropdown(
                                    id="selector-modelo",
                                    options=[
                                        {"label": "Regresión lineal", "value": "lineal"},
                                        {"label": "Ridge", "value": "ridge"},
                                        {"label": "Lasso", "value": "lasso"},
                                        {"label": "Ridge + grado 2", "value": "ridge2"},
                                    ],
                                    value="lineal",
                                    clearable=False,
                                ),

                                html.Div(
                                    "2. Marque las variables que participan",
                                    className="sidebar-step",
                                ),
                                dcc.Checklist(
                                    id="variables-modelo",
                                    options=[
                                        {"label": ETIQUETAS[v], "value": v}
                                        for v in PREDICTORES
                                    ],
                                    value=PREDICTORES,
                                    className="model-checklist prominent-checklist",
                                ),

                                html.Div(
                                    "3. Regularización α",
                                    className="sidebar-step",
                                ),
                                dcc.Dropdown(
                                    id="alpha-modelo",
                                    options=[
                                        {"label": "0.001", "value": 0.001},
                                        {"label": "0.01", "value": 0.01},
                                        {"label": "0.1", "value": 0.1},
                                        {"label": "1", "value": 1.0},
                                        {"label": "10", "value": 10.0},
                                        {"label": "100", "value": 100.0},
                                    ],
                                    value=1.0,
                                    clearable=False,
                                ),

                                html.Div(
                                    "Cambie estas opciones y abra la pestaña MODELO "
                                    "para ver el diagrama y las métricas de la especificación.",
                                    className="model-instruction",
                                ),

                                html.Div("EXPLORACIÓN GRÁFICA", className="sidebar-section-title"),

                                html.Label("Competencia para explorar"),
                                dcc.Dropdown(
                                    options=[
                                        {"label": ETIQUETAS[v], "value": v}
                                        for v in PREDICTORES
                                    ],
                                    value="lectura",
                                    clearable=False,
                                    id="variable",
                                ),

                                html.Label("Rango del puntaje"),
                                dcc.RangeSlider(
                                    min=0,
                                    max=300,
                                    step=5,
                                    value=[0, 300],
                                    marks={0: "0", 100: "100", 200: "200", 300: "300"},
                                    id="rango",
                                ),

                                html.Label("Tamaño de muestra gráfica"),
                                dcc.Slider(
                                    min=1000,
                                    max=min(10000, max(1000, len(DATOS))),
                                    step=1000,
                                    value=min(5000, len(DATOS)),
                                    marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    id="muestra",
                                ),

                                html.Button(
                                    "Restablecer filtros gráficos",
                                    id="reset",
                                    n_clicks=0,
                                    className="win-button",
                                ),
                            ],
                            className="sidebar-body",
                        ),
                    ],
                    className="sidebar win-window",
                ),
                html.Main(
                    [
                        html.Div(
                            [
                                tarjeta("Registros visualizados", "kpi-n", "según los filtros"),
                                tarjeta("Promedio RC", "kpi-rc", "Razonamiento Cuantitativo"),
                                tarjeta("R² modelo final", "kpi-r2", "conjunto de prueba"),
                                tarjeta("RMSE modelo final", "kpi-rmse", "error típico aproximado"),
                            ],
                            className="kpi-grid",
                        ),
                        dcc.Tabs(
                            [
                                dcc.Tab(
                                    label="Panorama",
                                    children=[
                                        html.Div(
                                            [dcc.Graph(id="scatter"), dcc.Graph(id="histograma")],
                                            className="grid-2",
                                        ),
                                        html.Div(
                                            [dcc.Graph(id="correlaciones")],
                                            className="panel",
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Lectura del panorama", className="window-titlebar"),
                                                html.P(
                                                    "La relación entre las competencias es positiva, pero ninguna variable por sí sola explica completamente el desempeño en Razonamiento Cuantitativo."
                                                ),
                                            ],
                                            className="narrative win-window",
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Modelo",
                                    children=[
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div("Diagrama del modelo", className="window-titlebar"),
                                                        dcc.Graph(id="modelo-diagrama"),
                                                    ],
                                                    className="model-window win-window",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div("Evaluación de la especificación", className="window-titlebar"),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Div("R² prueba", className="mini-label"),
                                                                        html.Div(id="modelo-r2", className="mini-value"),
                                                                    ],
                                                                    className="mini-metric",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Div("RMSE", className="mini-label"),
                                                                        html.Div(id="modelo-rmse", className="mini-value"),
                                                                    ],
                                                                    className="mini-metric",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Div("MAE", className="mini-label"),
                                                                        html.Div(id="modelo-mae", className="mini-value"),
                                                                    ],
                                                                    className="mini-metric",
                                                                ),
                                                            ],
                                                            className="model-metrics",
                                                        ),
                                                        html.Div("Representación del modelo", className="model-subheading"),
                                                        html.Pre(id="modelo-ecuacion", className="model-equation"),
                                                        html.P(id="modelo-texto", className="model-text"),
                                                    ],
                                                    className="model-summary win-window",
                                                ),
                                            ],
                                            className="model-layout",
                                        ),
                                    ],
                                ),

                                dcc.Tab(
                                    label="Comparación de modelos",
                                    children=[
                                        html.Div(
                                            [dcc.Graph(id="r2-modelos"), dcc.Graph(id="rmse-modelos")],
                                            className="grid-2",
                                        ),
                                        html.Div(
                                            [
                                                html.Div("¿Qué cambió al mejorar el modelo?", className="window-titlebar"),
                                                html.P(id="texto-modelos"),
                                            ],
                                            className="narrative win-window",
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Tabla comparativa", className="window-titlebar"),
                                                dash_table.DataTable(
                                                    id="tabla-modelos",
                                                    data=METRICAS.round(4).to_dict("records"),
                                                    columns=[
                                                        {"name": c, "id": c}
                                                        for c in METRICAS.columns
                                                    ],
                                                    page_size=10,
                                                    style_table={"overflowX": "auto"},
                                                    style_cell={
                                                        "fontFamily": "Tahoma, Verdana, Arial, sans-serif",
                                                        "fontSize": 12,
                                                        "padding": "7px",
                                                        "backgroundColor": C["crema"],
                                                        "color": "#111111",
                                                        "border": "1px solid #808080",
                                                    },
                                                    style_header={
                                                        "backgroundColor": C["azul"],
                                                        "color": "white",
                                                        "fontWeight": "bold",
                                                        "border": "1px solid #404040",
                                                    },
                                                    style_data_conditional=[
                                                        {
                                                            "if": {"row_index": "odd"},
                                                            "backgroundColor": "#EFECE5",
                                                        }
                                                    ],
                                                ),
                                            ],
                                            className="table-window win-window",
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Variables y estabilidad",
                                    children=[
                                        html.Div(
                                            [dcc.Graph(id="coeficientes"), dcc.Graph(id="vif")],
                                            className="grid-2",
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Interpretación", className="window-titlebar"),
                                                html.P(
                                                    "Los VIF observados son bajos a moderados. La regularización Ridge y Lasso, aplicada sobre las mismas cuatro variables, no produjo una mejora práctica frente a la regresión múltiple."
                                                ),
                                            ],
                                            className="narrative win-window",
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Diagnóstico",
                                    children=[
                                        html.Div(
                                            [dcc.Graph(id="real-pred"), dcc.Graph(id="residuos")],
                                            className="grid-2",
                                        ),
                                        html.Div(
                                            [dcc.Graph(id="hist-residuos")],
                                            className="panel",
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Qué muestran los errores", className="window-titlebar"),
                                                html.P(
                                                    "El modelo reproduce la tendencia general, pero conserva una dispersión importante alrededor de los valores observados. Por ello, sus predicciones deben interpretarse como aproximaciones y no como resultados exactos para cada estudiante."
                                                ),
                                            ],
                                            className="narrative win-window",
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Conclusiones",
                                    children=[
                                        html.Div(
                                            [
                                                html.Div("Conclusiones principales", className="window-titlebar"),
                                                html.Div(
                                                    [
                                                        html.P("1. El modelo múltiple mejora de forma clara frente al modelo que utiliza únicamente Lectura Crítica."),
                                                        html.P("2. No se observaron señales relevantes de sobreajuste: entrenamiento, prueba y validación cruzada mostraron resultados muy similares."),
                                                        html.P("3. Ridge y Lasso no ofrecieron una mejora práctica al trabajar solamente con las cuatro competencias originales."),
                                                        html.P("4. Las características de segundo grado produjeron el mejor desempeño predictivo, aunque la mejora adicional fue pequeña."),
                                                        html.P("5. La mayor ganancia se obtuvo al pasar del modelo simple al modelo múltiple; después de ese punto aparecen rendimientos decrecientes."),
                                                        html.P("6. Una parte importante del desempeño en Razonamiento Cuantitativo continúa sin ser explicada por las variables incluidas, por lo que futuros análisis deberían considerar información adicional pertinente."),
                                                    ],
                                                    className="conclusions-body",
                                                ),
                                            ],
                                            className="conclusions win-window",
                                        )
                                    ],
                                ),
                            ],
                            className="classic-tabs",
                        ),
                        html.Footer(
                            "Fuente: resultados Saber Pro 2024 y archivos exportados por el notebook de la Actividad 3."
                        ),
                    ],
                    className="content",
                ),
            ],
            className="shell",
        ),
    ],
    className="app",
)

# ============================================================
# CALLBACK DE RESTABLECIMIENTO
# ============================================================
@app.callback(
    Output("variable", "value"),
    Output("rango", "value"),
    Output("muestra", "value"),
    Input("reset", "n_clicks"),
    prevent_initial_call=True,
)
def restablecer(_):
    return "lectura", [0, 300], min(5000, len(DATOS))


# ============================================================
# CALLBACK PRINCIPAL
# ============================================================
@app.callback(
    Output("kpi-n", "children"),
    Output("kpi-rc", "children"),
    Output("kpi-r2", "children"),
    Output("kpi-rmse", "children"),
    Output("scatter", "figure"),
    Output("histograma", "figure"),
    Output("correlaciones", "figure"),
    Output("r2-modelos", "figure"),
    Output("rmse-modelos", "figure"),
    Output("texto-modelos", "children"),
    Output("coeficientes", "figure"),
    Output("vif", "figure"),
    Output("real-pred", "figure"),
    Output("residuos", "figure"),
    Output("hist-residuos", "figure"),
    Input("variable", "value"),
    Input("rango", "value"),
    Input("muestra", "value"),
)
def actualizar(variable, rango, muestra):
    d = DATOS[(DATOS[variable] >= rango[0]) & (DATOS[variable] <= rango[1])].copy()

    if d.empty:
        v = grafica_vacia()
        return (
            "0",
            "--",
            "--",
            "--",
            v,
            v,
            v,
            v,
            v,
            "Sin datos con los filtros actuales.",
            v,
            v,
            v,
            v,
            v,
        )

    m = d.sample(n=min(int(muestra), len(d)), random_state=42)

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------
    mejor = METRICAS.sort_values(
        ["R2_test", "RMSE_test"],
        ascending=[False, True],
    ).iloc[0]

    kpi_n = f"{len(d):,}"
    kpi_rc = f"{d['razonamiento'].mean():.1f}"
    kpi_r2 = f"{mejor['R2_test']:.4f}"
    kpi_rmse = f"{mejor['RMSE_test']:.2f}"

    # --------------------------------------------------------
    # PANORAMA
    # --------------------------------------------------------
    fig_scatter = px.scatter(
        m,
        x=variable,
        y="razonamiento",
        opacity=0.28,
        trendline="ols",
        title=f"{ETIQUETAS[variable]} y Razonamiento Cuantitativo",
        labels={
            variable: ETIQUETAS[variable],
            "razonamiento": "Razonamiento Cuantitativo",
        },
    )
    fig_scatter.update_traces(
        marker={
            "color": C["azul2"],
            "size": 6,
            "line": {"width": 0.3, "color": C["azul"]},
        }
    )
    # Plotly agrega la recta de tendencia como otra traza.
    if len(fig_scatter.data) > 1:
        fig_scatter.data[-1].line.color = C["cafe"]
        fig_scatter.data[-1].line.width = 2.5
    fig_scatter = aplicar_estilo_win98(fig_scatter)

    fig_hist = px.histogram(
        d,
        x="razonamiento",
        nbins=35,
        title="Distribución de Razonamiento Cuantitativo",
        labels={"razonamiento": "Puntaje"},
    )
    fig_hist.update_traces(
        marker_color=C["cafe2"],
        marker_line_color=C["cafe"],
        marker_line_width=0.7,
    )
    fig_hist.update_layout(yaxis_title="Frecuencia")
    fig_hist = aplicar_estilo_win98(fig_hist)

    escala_corr = [
        [0.00, "#E6E2D8"],
        [0.25, "#B6C7DD"],
        [0.50, "#7FA0C8"],
        [0.75, "#466D9C"],
        [1.00, "#0A246A"],
    ]

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=CORR.values,
            x=[ETIQUETAS.get(c, c) for c in CORR.columns],
            y=[ETIQUETAS.get(i, i) for i in CORR.index],
            zmin=0,
            zmax=1,
            colorscale=escala_corr,
            text=np.round(CORR.values, 3),
            texttemplate="%{text}",
            colorbar=dict(
                title="r",
                bgcolor=C["gris_claro"],
                outlinecolor=C["gris"],
                outlinewidth=1,
            ),
        )
    )
    fig_corr = aplicar_estilo_win98(
        fig_corr,
        "Correlaciones entre competencias",
    )

    # --------------------------------------------------------
    # COMPARACIÓN DE MODELOS
    # --------------------------------------------------------
    orden = METRICAS.sort_values("R2_test", ascending=True)
    fig_r2 = px.bar(
        orden,
        x="R2_test",
        y="Modelo",
        orientation="h",
        title="R² en el conjunto de prueba",
        text=orden["R2_test"].map(lambda x: f"{x:.4f}"),
    )
    fig_r2.update_traces(
        marker_color=C["azul2"],
        marker_line_color=C["azul"],
        marker_line_width=1,
        textposition="outside",
    )
    fig_r2.update_layout(xaxis_title="R²", yaxis_title="")
    fig_r2 = aplicar_estilo_win98(fig_r2)

    orden_rmse = METRICAS.sort_values("RMSE_test", ascending=False)
    fig_rmse = px.bar(
        orden_rmse,
        x="RMSE_test",
        y="Modelo",
        orientation="h",
        title="RMSE en el conjunto de prueba",
        text=orden_rmse["RMSE_test"].map(lambda x: f"{x:.2f}"),
    )
    fig_rmse.update_traces(
        marker_color=C["cafe2"],
        marker_line_color=C["cafe"],
        marker_line_width=1,
        textposition="outside",
    )
    fig_rmse.update_layout(xaxis_title="RMSE", yaxis_title="")
    fig_rmse = aplicar_estilo_win98(fig_rmse)

    base = METRICAS[METRICAS["Modelo"] == "Regresión múltiple"].iloc[0]
    delta_r2 = mejor["R2_test"] - base["R2_test"]
    delta_rmse = base["RMSE_test"] - mejor["RMSE_test"]

    texto = (
        f"El mejor desempeño correspondió a {mejor['Modelo']}. "
        f"Frente a la regresión múltiple, el R² aumentó {delta_r2:.4f} "
        f"y el RMSE disminuyó {delta_rmse:.4f} puntos. "
        "La mejora adicional es pequeña, por lo que la mayor complejidad "
        "debe justificarse por su utilidad predictiva y no solo por ocupar "
        "el primer lugar en la tabla."
    )

    # --------------------------------------------------------
    # VARIABLES Y ESTABILIDAD
    # --------------------------------------------------------
    coef_plot = COEF.copy()
    coef_plot["Variable"] = coef_plot["Variable"].map(
        lambda x: ETIQUETAS.get(x, x)
    )

    fig_coef = px.bar(
        coef_plot,
        x="Coeficiente",
        y="Variable",
        color="Modelo",
        barmode="group",
        orientation="h",
        title="Coeficientes de los modelos",
        color_discrete_sequence=[C["azul"], C["azul3"], C["cafe2"]],
    )
    fig_coef.update_traces(
        marker_line_color=C["gris_oscuro"],
        marker_line_width=0.5,
    )
    fig_coef.update_layout(yaxis_title="")
    fig_coef = aplicar_estilo_win98(fig_coef)

    vif_plot = VIF.copy()
    vif_plot["Variable"] = vif_plot["Variable"].map(
        lambda x: ETIQUETAS.get(x, x)
    )
    vif_plot = vif_plot.sort_values("VIF")

    fig_vif = px.bar(
        vif_plot,
        x="VIF",
        y="Variable",
        orientation="h",
        text=vif_plot["VIF"].map(lambda x: f"{x:.2f}"),
        title="Factor de inflación de la varianza (VIF)",
    )
    fig_vif.update_traces(
        marker_color=C["cafe2"],
        marker_line_color=C["cafe"],
        marker_line_width=1,
        textposition="outside",
    )
    fig_vif.update_layout(yaxis_title="")
    fig_vif = aplicar_estilo_win98(fig_vif)

    # --------------------------------------------------------
    # DIAGNÓSTICO
    # --------------------------------------------------------
    pm = PRED.sample(n=min(int(muestra), len(PRED)), random_state=42)

    fig_real = px.scatter(
        pm,
        x="RC_real",
        y="RC_predicho",
        opacity=0.28,
        title="Valores reales frente a predichos",
        labels={"RC_real": "RC real", "RC_predicho": "RC predicho"},
    )
    lo = min(pm["RC_real"].min(), pm["RC_predicho"].min())
    hi = max(pm["RC_real"].max(), pm["RC_predicho"].max())
    fig_real.add_shape(
        type="line",
        x0=lo,
        y0=lo,
        x1=hi,
        y1=hi,
        line={"dash": "dash", "color": C["cafe"], "width": 2},
    )
    fig_real.update_traces(
        marker={
            "color": C["azul2"],
            "size": 6,
            "line": {"width": 0.25, "color": C["azul"]},
        }
    )
    fig_real = aplicar_estilo_win98(fig_real)

    fig_res = px.scatter(
        pm,
        x="RC_predicho",
        y="residuo",
        opacity=0.28,
        title="Residuos frente a valores predichos",
        labels={"RC_predicho": "RC predicho", "residuo": "Residuo"},
    )
    fig_res.add_hline(
        y=0,
        line_dash="dash",
        line_color=C["cafe"],
        line_width=2,
    )
    fig_res.update_traces(
        marker={
            "color": C["azul3"],
            "size": 6,
            "line": {"width": 0.25, "color": C["azul"]},
        }
    )
    fig_res = aplicar_estilo_win98(fig_res)

    fig_hist_res = px.histogram(
        PRED,
        x="residuo",
        nbins=35,
        title="Distribución de residuos",
        labels={"residuo": "Residuo"},
    )
    fig_hist_res.update_traces(
        marker_color=C["cafe2"],
        marker_line_color=C["cafe"],
        marker_line_width=0.7,
    )
    fig_hist_res.update_layout(yaxis_title="Frecuencia")
    fig_hist_res = aplicar_estilo_win98(fig_hist_res)

    return (
        kpi_n,
        kpi_rc,
        kpi_r2,
        kpi_rmse,
        fig_scatter,
        fig_hist,
        fig_corr,
        fig_r2,
        fig_rmse,
        texto,
        fig_coef,
        fig_vif,
        fig_real,
        fig_res,
        fig_hist_res,
    )


# ============================================================
# CALLBACK DEL MODELADOR INTERACTIVO
# ============================================================
@app.callback(
    Output("modelo-diagrama", "figure"),
    Output("modelo-r2", "children"),
    Output("modelo-rmse", "children"),
    Output("modelo-mae", "children"),
    Output("modelo-ecuacion", "children"),
    Output("modelo-texto", "children"),
    Input("selector-modelo", "value"),
    Input("variables-modelo", "value"),
    Input("alpha-modelo", "value"),
)
def actualizar_modelador(tipo, variables, alpha):
    variables = variables or []

    nombres = {
        "lineal": "Regresión lineal",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "ridge2": "Ridge + características de segundo grado",
    }
    nombre = nombres.get(tipo, tipo)

    if not variables:
        return (
            grafica_vacia("Seleccione al menos una variable"),
            "--",
            "--",
            "--",
            "Sin modelo: seleccione al menos una variable predictora.",
            "No es posible evaluar un modelo sin variables de entrada.",
        )

    datos_exp = DATOS[variables + ["razonamiento"]].dropna().copy()

    X_exp = datos_exp[variables]
    y_exp = datos_exp["razonamiento"]

    X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
        X_exp,
        y_exp,
        test_size=0.20,
        random_state=42,
    )

    modelo = construir_modelo_exploratorio(tipo, alpha)
    modelo.fit(X_train_exp, y_train_exp)
    pred_exp = modelo.predict(X_test_exp)

    r2 = r2_score(y_test_exp, pred_exp)
    rmse = np.sqrt(mean_squared_error(y_test_exp, pred_exp))
    mae = mean_absolute_error(y_test_exp, pred_exp)

    fig = figura_estructura_modelo(variables, nombre)

    if tipo == "lineal":
        ecuacion = f"RC = {modelo.intercept_:.2f}"
        for var, coef in zip(variables, modelo.coef_):
            signo = "+" if coef >= 0 else "-"
            ecuacion += (
                f" {signo} {abs(coef):.3f} × "
                f"{ETIQUETAS.get(var, var)}"
            )

    elif tipo in ("ridge", "lasso"):
        estimador = modelo.named_steps["modelo"]
        partes = []
        for var, coef in zip(variables, estimador.coef_):
            partes.append(
                f"{coef:.3f} × z({ETIQUETAS.get(var, var)})"
            )
        ecuacion = (
            "Coeficientes sobre variables estandarizadas:\n"
            + " + ".join(partes)
        )

    else:
        poly = modelo.named_steps["grado2"]
        nombres_terminos = poly.get_feature_names_out(variables)
        coeficientes = modelo.named_steps["modelo"].coef_
        indices = np.argsort(np.abs(coeficientes))[::-1][:8]

        lineas = [
            f"{coeficientes[i]: .3f} × {nombres_terminos[i]}"
            for i in indices
        ]
        ecuacion = (
            "Términos con mayor peso absoluto:\n"
            + "\n".join(lineas)
        )

    texto = (
        f"Especificación evaluada: {nombre}, con {len(variables)} variable(s). "
        f"En la partición 80/20 de la muestra del dashboard obtuvo R²={r2:.4f}, "
        f"RMSE={rmse:.2f} y MAE={mae:.2f}. Use esta herramienta para comparar "
        "qué ocurre al incluir o retirar competencias; la selección final debe "
        "considerar desempeño, simplicidad e interpretación."
    )

    return (
        fig,
        f"{r2:.4f}",
        f"{rmse:.2f}",
        f"{mae:.2f}",
        ecuacion,
        texto,
    )




if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8051)
