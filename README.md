# Dashboard Saber Pro 2024

Dashboard interactivo de la Actividad 3 del proyecto de Ciencia de Datos. Resume la evolución del modelo para estimar el puntaje de Razonamiento Cuantitativo a partir de otras competencias genéricas de Saber Pro 2024.

## Archivos necesarios

Copie desde la carpeta `salidas_dashboard` generada por el notebook los siguientes archivos dentro de `data/`:

- `datos_dashboard.csv`
- `metricas_modelos.csv`
- `coeficientes_modelos.csv`
- `vif.csv`
- `correlaciones.csv`
- `predicciones_dashboard.csv`

## Ejecutar localmente

Abra una terminal en esta carpeta y ejecute:

```bash
pip install -r requirements.txt
python app.py
```

Después abra:

`http://127.0.0.1:8050`

## Estructura del dashboard

El tablero contiene cinco secciones:

1. **Panorama:** relación de las competencias con Razonamiento Cuantitativo.
2. **Comparación de modelos:** R² y RMSE del modelo simple, múltiple, Ridge, Lasso y Ridge con características de segundo grado.
3. **Variables y estabilidad:** coeficientes y VIF.
4. **Diagnóstico:** valores reales frente a predichos y análisis de residuos.
5. **Conclusiones:** síntesis narrativa de los principales hallazgos.

## GitHub y Binder

1. Cree un repositorio en GitHub.
2. Suba esta carpeta completa, incluyendo `data/` y `binder/start`.
3. Cambie `USUARIO` y `REPOSITORIO` en `ENLACES.txt`.
4. Abra el enlace de Binder. El archivo `binder/start` iniciará el dashboard en el puerto 8050 y `jupyter-server-proxy` permitirá visualizarlo.

## Nota metodológica

Los archivos de visualización pueden contener una muestra de hasta 30.000 registros para mantener el dashboard ligero. Los modelos fueron entrenados y evaluados sobre todos los registros completos empleados en el notebook.
