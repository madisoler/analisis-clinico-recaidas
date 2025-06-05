# Proyecto de Análisis de Factores de Riesgo y Recaída en Pacientes

Este proyecto tiene como objetivo explorar, modelar e interpretar factores clínicos, demográficos y patológicos asociados al riesgo y a la recurrencia en pacientes. Utiliza técnicas de análisis estadístico y machine learning (regresión logística, árboles de decisión y random forest) aplicadas a un dataset clínico real.

---

## 📦 Requisitos

Para ejecutar correctamente este proyecto, se recomienda tener instaladas las siguientes bibliotecas de Python:

```bash
pandas
numpy
scikit-learn
statsmodels
matplotlib
seaborn
patsy
```

Se puede instalar todo utilizando:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Estructura del Código

- `Codigo_Proyecto_1_annotated.py`: Código principal con todas las etapas del análisis explicadas paso a paso.
  - Limpieza y exploración de datos
  - Codificación de variables categóricas
  - Evaluación de colinealidad (correlación y VIF)
  - Ajuste de modelos de regresión logística multinomial
  - Ajuste de modelos de regresión logística binaria para predicción de recaída
  - Implementación de Árbol de Decisión y Random Forest
  - Cálculo e interpretación de métricas de evaluación

---

## ▶️ Cómo Ejecutar

1. Asegurarse de tener el dataset en formato `.csv` o cargar los datos que usó el notebook.
2. Ejecutar el script `.py` desde tu entorno local:

```bash
python Codigo_Proyecto_1_annotated.py
```

3. Los resultados aparecerán en la consola e incluirán métricas como precisión, matriz de confusión y reportes de clasificación.

---

## 👩‍⚕️ Créditos

Proyecto desarrollado por Dra. Marcela de Dios Soler como parte del proceso de formación en análisis de datos clínicos y medicina de precisión.  
Mentoría técnica y guía: Dra. Gisela Pattarone

---
