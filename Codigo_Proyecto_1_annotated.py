#!/usr/bin/env python
# coding: utf-8

from google.colab import drive
drive.mount ('/content/drive')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leer la data

data = pd.read_csv("/content/drive/MyDrive/Copia de Thyroid_Diff.csv")


# ### Bajo Riesgo

grupo_bajo_riesgo = data[data['Risk'] == 'Bajo']
grupo_no_recaidos_bajo_riesgo = data[(data['Risk'] == 'Bajo') & (data['Recurred'] == "No")]
grupo_bajo_riesgo_recaido = data[(data['Risk'] == 'Bajo') & (data['Recurred'] == "Yes")]


# ### Riesgo Intermedio

# In[ ]:


grupo_riesgo_intermedio = data[data['Risk'] == 'Intermedio']
grupo_no_recaidos_riesgo_intermedio = data[(data['Risk'] == 'Intermedio') & (data['Recurred'] == "No")]


# ### Alto Riesgo

# In[ ]:


grupo_alto_riesgo = data[data['Risk'] == 'Alto']
grupo_recaidos_alto_riesgo = data[(data['Risk'] == 'Alto') & (data['Recurred'] == "Yes")]


# ### Grupos por Recurrencia

# In[ ]:


grupo_con_recurrencia = data[data['Recurred'] == "Yes"]


# In[ ]:


grupo_sin_recurrencia = data[data['Recurred'] == "No"]


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Recurred', y='Age', data=data, palette='Set2')
plt.title('Comparación de la Edad por Grupo c/s recurrencia')
plt.xlabel('Nivel de Riesgo')
plt.ylabel('Edad')
plt.show()


# ### Subgrupos combinados

# #### Bajo riesgo con recurrencia:

# In[ ]:


grupo_recurrencia_bajo_riesgo = data[(data['Risk'] == 'Bajo') & (data['Recurred'] == "Yes")]


# #### Alto riesgo sin recurrencia:

# In[ ]:


grupo_alto_riesgo_sin_recurrencia = data[(data['Risk'] == 'Alto') & (data['Recurred'] == "No")]


# #### Riesgo Intermedio con y sin recurrencia

# In[ ]:


grupo_riesgo_intermedio_con_recurrencia = data[(data['Risk'] == 'Intermedio') & (data['Recurred'] =="Yes")]
grupo_riesgo_intermedio_sin_recurrencia = data[(data['Risk'] == 'Intermedio') & (data['Recurred'] == "No")]


# ### Visualizaciones para comparar grupos

# 1. Comparación de Edad por Grupo de Riesgo (Boxplot combinado)

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='Risk', y='Age', data=data, palette='Set2')
plt.title('Comparación de la Edad por Grupo de Riesgo')
plt.xlabel('Nivel de Riesgo')
plt.ylabel('Edad')
plt.show()


# 2. Distribución de Recurrencia en cada Grupo de Riesgo (Gráfico de Barras Apiladas)

# In[ ]:


plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Risk', hue='Recurred', palette='coolwarm')
plt.title('Distribución de Recurrencia por Grupo de Riesgo')
plt.xlabel('Nivel de Riesgo')
plt.ylabel('Número de Pacientes')
plt.legend(title='Recurrencia', labels=['No', 'Sí'])
plt.show()


# 3. Frecuencia de Recurrencia por Riesgo (Pie Chart por Grupo)

# In[ ]:


# Importar matplotlib para los gráficos
import matplotlib.pyplot as plt

# Transformar los valores de las columnas 'Risk' y 'Recurred'
data['Risk'] = data['Risk'].replace({'Low': 'Bajo', 'Intermediate': 'Intermedio', 'High': 'Alto'})
data['Recurred'] = data['Recurred'].replace({'No': 0, 'Yes': 1})


# Generar nuevamente el gráfico de Pie Charts por grupo de riesgo
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for idx, risk_level in enumerate(['Bajo', 'Intermedio', 'Alto']):
    grupo = data[data['Risk'] == risk_level]
    recurred_counts = grupo['Recurred'].value_counts().reindex([0, 1], fill_value=0)

    # Etiquetas basadas en los valores
    labels = ['No', 'Sí']

    # Verificar si hay datos para graficar
    if recurred_counts.sum() > 0:
        axs[idx].pie(recurred_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
        axs[idx].set_title(f'Recurrencia - {risk_level} Riesgo')
    else:
        axs[idx].text(0.5, 0.5, 'Sin datos', horizontalalignment='center', verticalalignment='center', fontsize=12)
        axs[idx].set_title(f'Recurrencia - {risk_level} Riesgo')

plt.suptitle('Distribución de Recurrencia por Nivel de Riesgo')
plt.show()



# In[ ]:


data['Adenopathy'] = data['Adenopathy'].replace({'No': 'No', 'Bilateral': 'Si', 'Extensive': 'Si', "Left": "Si", "Posterior": "Si", "Right": "Si"  })


# In[ ]:


data['N'] = data['N'].replace({"N0": 0, "N1a": 1, "N1b": 1})


# In[ ]:


# Generar gráfico de Pie Charts
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

data['N'] = data['N'].replace({'No': 0, 'Yes': 1})
data['Adenopathy'] = data['Adenopathy'].replace({'No': 'No', 'Bilateral': 'Si', 'Extensive': 'Si', "Left": "Si", "Posterior": "Si", "Right": "Si"  })

colors=['#66b3ff', '#ff9999', '#cce6ff']
labels_map = {"N0":0, "Yes":1}

for idx, adenopathy_level in enumerate(['No', "Si"]):
    grupo = data[data["Adenopathy"] == adenopathy_level]
    N_counts = grupo['N'].value_counts().reindex([0, 1], fill_value=0)

    # Verificar si hay datos para graficar
    if N_counts.sum() > 0:
        axs[idx].pie(N_counts, labels=labels_map, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
        axs[idx].set_title(f'{adenopathy_level} Adenopatias')
    else:
        axs[idx].text(0.5, 0.5, 'Sin datos', horizontalalignment='center', verticalalignment='center', fontsize=12)
        axs[idx].set_title(f'{adenopathy_level} Adenopatias')


plt.suptitle('Distribucion de Estadio N segun Adenopatia')
plt.tight_layout()
plt.show()


# 4. Comparación Respuesta al Tratamiento vs Recurrencia (Gráfico de Barras Apiladas)

# In[ ]:


plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Response', hue='Recurred', palette='pastel')
plt.title('Comparación entre Respuesta al Tratamiento y Recurrencia')
plt.xlabel('Tipo de Respuesta al Tratamiento')
plt.ylabel('Número de Pacientes')
plt.legend(title='Recurrencia', labels=['No', 'Sí'])
plt.xticks(rotation=45)
plt.show()


# 5. Relación entre Respuesta al Tratamiento y Género (Gráfico de Barras)

# In[ ]:


plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Response', hue='Gender', palette='Set1')
plt.title('Relación entre Respuesta al Tratamiento y Género')
plt.xlabel('Tipo de Respuesta')
plt.ylabel('Número de Pacientes')
plt.legend(title='Género')
plt.xticks(rotation=45)
plt.show()


# 6. Relación entre Estadio y Recurrencia (Gráfico de Barras Apiladas)

# In[ ]:


plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Stage', hue='Recurred', palette='muted')
plt.title('Relación entre Estadio y Recurrencia')
plt.xlabel('Estadio')
plt.ylabel('Número de Pacientes')
plt.legend(title='Recurrencia', labels=['No', 'Sí'])
plt.show()


# 7. Análisis Específicos por Grupo de Riesgo (Subplots Combinados)

# In[ ]:


# Importar seaborn para los gráficos
import seaborn as sns

# Generar los gráficos combinados para los tres grupos de riesgo
fig, axs = plt.subplots(3, 2, figsize=(16, 18))

# Bajo riesgo
sns.boxplot(x='Gender', y='Age', data=data[data['Risk'] == 'Bajo'], ax=axs[0, 0])
axs[0, 0].set_title('Bajo Riesgo - Distribución de Edad por Género')

sns.countplot(data=data[data['Risk'] == 'Bajo'], x='Recurred', hue='Response', ax=axs[0, 1])
axs[0, 1].set_title('Bajo Riesgo - Respuesta al Tratamiento y Recurrencia')

# Riesgo intermedio
sns.boxplot(x='Gender', y='Age', data=data[data['Risk'] == 'Intermedio'], ax=axs[1, 0])
axs[1, 0].set_title('Riesgo Intermedio - Distribución de Edad por Género')

sns.countplot(data=data[data['Risk'] == 'Intermedio'], x='Recurred', hue='Response', ax=axs[1, 1])
axs[1, 1].set_title('Riesgo Intermedio - Respuesta al Tratamiento y Recurrencia')

# Alto riesgo
sns.boxplot(x='Gender', y='Age', data=data[data['Risk'] == 'Alto'], ax=axs[2, 0])
axs[2, 0].set_title('Alto Riesgo - Distribución de Edad por Género')

sns.countplot(data=data[data['Risk'] == 'Alto'], x='Recurred', hue='Response', ax=axs[2, 1])
axs[2, 1].set_title('Alto Riesgo - Respuesta al Tratamiento y Recurrencia')

# Ajustar diseño
plt.tight_layout()
plt.show()


# In[ ]:


# EDAD PROMEDIO GRUPOS SIN Y CON RECURRENCIA


# In[ ]:


df = data.copy("/content/drive/MyDrive/Copia de Thyroid_Diff.csv")


# In[ ]:


# Verificar las columnas y tipos de datos
print(df.dtypes)

# Verificar valores únicos en la columna 'Recurred'
print(df['Recurred'].unique())

# Convertir la columna 'Recurred' a valores binarios si es necesario
df['Recurred'] = df['Recurred'].replace({'Yes': 1, 'No': 0})

# Asegurar que 'Recurred' sea numérico y eliminar valores nulos
df['Recurred'] = pd.to_numeric(df['Recurred'], errors='coerce')
df = df.dropna(subset=['Recurred'])

# Convertir la columna 'Age' a numérico y eliminar valores no válidos
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df = df.dropna(subset=['Age'])

# Verificar los valores mínimos y máximos en 'Age' para asegurar que se pueden categorizar
print("Edad mínima:", df['Age'].min(), "Edad máxima:", df['Age'].max())

# Definir grupos de edad correctamente
bins = [0, 40, 55, 100]
labels = ['<40 años', '40-54 años', '>55 años']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False, ordered=False)

# Verificar la cantidad de datos en cada grupo
print(df['Age Group'].value_counts(dropna=False))

# Verificar que hay datos en cada grupo antes de calcular la tasa de recurrencia
if df['Age Group'].isnull().sum() > 0:
    print("Advertencia: Hay valores nulos en 'Age Group'.")

# Calcular la tasa de recurrencia por grupo de edad
recurrence_rates_final = df.groupby('Age Group', observed=False)['Recurred'].mean() * 100

# Verificar cuántos valores hay en cada grupo de edad
print(df.groupby('Age Group')['Recurred'].count())

# Graficar la distribución con los datos corregidos
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.barplot(x=recurrence_rates_final.index, y=recurrence_rates_final.values, palette='coolwarm')
plt.ylabel('Tasa de Recurrencia (%)')
plt.xlabel('Grupo de Edad')
plt.title('Tasa de Recurrencia por Grupo de Edad')
plt.ylim(0, 100)
plt.show()

# Mostrar la tasa de recurrencia en cada grupo de edad
print(recurrence_rates_final)


# ### Estadistica aplicada

# ##### ANOVA/Kruskal-Wallis (Edad): Bajo riesgo, Riesgo intermedio y Alto riesgo

# In[ ]:


data.head()


# In[ ]:


# Fijas: risk, TNM, Stage, Response, Age, Gender


# In[ ]:


# recurrencia: Age > 55 y < 55; creamos una nueva variable baja_probabilidad y alta_probabilidad, Gender, Smoking, Thyroid Function, Focality


# In[ ]:


data['Risk'].unique()


# In[ ]:


# Shapiro-Wilk (Verificación de Normalidad)
from scipy.stats import shapiro

# Filtrar edades por grupo de riesgo
bajo = data[data['Risk'] == 'Low']['Age']
intermedio = data[data['Risk'] == 'Intermediate']['Age']
alto = data[data['Risk'] == 'High']['Age']


# In[ ]:


#Verificar normalidad en cada grupo
print("Shapiro-Wilk Test:")
print("Bajo Riesgo:", shapiro(bajo))
print("Riesgo Intermedio:", shapiro(intermedio))
print("Alto Riesgo:", shapiro(alto))


# In[ ]:


# Los tres grupos presentan distribución no normal (p < 0.05), excepto Alto Riesgo, que muestra una distribución aproximadamente normal (p = 0.077).


# In[ ]:


# ANOVA
from scipy.stats import f_oneway

# ANOVA para comparar la edad entre los grupos de riesgo
f_stat, p_value = f_oneway(bajo, intermedio, alto)

print("\nResultado ANOVA:")
print("F =", f_stat, ", p =", p_value)


# In[ ]:


# El ANOVA muestra diferencias significativas en la edad entre los grupos de riesgo (F = 21.05, p < 0.001). El estadístico F mide la relación entre la variabilidad entre los grupos y la variabilidad dentro de los grupos.
# Un F alto (como 21.05) indica que la variación entre los grupos es mucho mayor que la variación interna, lo que sugiere diferencias significativas entre ellos.


# In[ ]:


# Kruskal-Wallis
from scipy.stats import kruskal


# In[ ]:


# Kruskal-Wallis para comparar grupos sin normalidad
h_stat, p_value = kruskal(bajo, intermedio, alto)

print("\nResultado Kruskal-Wallis:")
print("H =", h_stat, ", p =", p_value)


# In[ ]:


# El test de Kruskal-Wallis muestra diferencias significativas entre los grupos (H = 28.04, p < 0.001), confirmando que al menos uno difiere en su distribución.
# El estadístico H en el test de Kruskal-Wallis representa la suma de rangos entre grupos ajustada por el tamaño de muestra.
# Mide si hay diferencias significativas en la distribución de las medianas entre los grupos. Un H alto (como 28.04) indica que las diferencias en los rangos entre los grupos son grandes, lo que sugiere que no provienen de la misma población.


# ##### Chi-Cuadrado (Para variables categóricas)

# In[ ]:


# Crear tabla de contingencia entre Riesgo y Recurrencia
import pandas as pd
from scipy.stats import chi2_contingency

# Crear tabla de contingencia entre Riesgo y Recurrencia
tabla = pd.crosstab(data['Risk'], data['Recurred'])


# In[ ]:


# Prueba de Chi-cuadrado
# Prueba de Chi-cuadrado
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# La prueba de Chi-cuadrado muestra una asociación altamente significativa entre el Riesgo y la Recurrencia (Chi2 = 208.83, p < 0.001). Esto indica que la recurrencia varía según el nivel de riesgo.
# El estadístico Chi2 mide la diferencia entre los valores observados y los esperados en una tabla de contingencia. Un Chi2 alto (como 208.83) indica que hay mucha discrepancia entre lo que se esperaría si las variables fueran independientes y lo que realmente se observa.
# Esto sugiere que las variables están asociadas (en este caso, Riesgo y Recurrencia).


# ### Analizar otras estadisticas:
# 
# 1. Edad entre Pacientes con y sin Recurrencia
# 

# In[ ]:


data['Recurred']


# In[ ]:


data['Recurred'].unique()


# In[ ]:


# Shapiro-Wilk (Verificación de Normalidad)
from scipy.stats import shapiro

# Filtrar edades por grupo de c/s recurrencia
con_recurrencia = data[data['Recurred'] == "Yes"]['Age'].dropna()
sin_recurrencia = data[data['Recurred'] == "No"]['Age'].dropna()


# In[ ]:


# Verificar la cantidad de datos en cada grupo antes de aplicar Shapiro-Wilk
len_con_recurrencia = len(con_recurrencia)
len_sin_recurrencia = len(sin_recurrencia)


# In[ ]:


len_con_recurrencia


# In[ ]:


len_sin_recurrencia


# In[ ]:


print("Shapiro-Wilk Test:")
print("con_recurrencia:", shapiro(con_recurrencia))
print("sin_recurrencia:", shapiro(sin_recurrencia))


# In[ ]:


# Ambos grupos presentan distribuciones no normales (p < 0.05), tanto con como sin recurrencia, según el test de Shapiro-Wilk.


# In[ ]:


# ANOVA para comparar la edad entre los grupos con y sin recaida
f_stat, p_value = f_oneway(con_recurrencia, sin_recurrencia)

print("\nResultado ANOVA:")
print("F =", f_stat, ", p =", p_value)


# In[ ]:


# El ANOVA muestra diferencias significativas en la edad entre los grupos con y sin recaída (F = 27.37, p < 0.001).


# In[ ]:


# Kruskall-Wallis  para comparar grupos sin normalidad
h_stat, p_value = kruskal(con_recurrencia, sin_recurrencia)

print("\nResultado Kruskal-Wallis:")
print("H =", h_stat, ", p =", p_value)


# In[ ]:


# El test de Kruskal-Wallis muestra diferencias significativas en la edad entre los grupos con y sin recurrencia (H = 17.10, p < 0.001).


# ### Analizar otras estadisticas:
# 
# 2. Respuesta al Tratamiento según el Nivel de Riesgo

# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Risk'], data['Response'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Response'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Stage'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# VER CON GISELA EL VALOR DEL CHI (97) EN ESTADIOS Y RECURRENCIA INDICA O NO ASOCIACION????

# In[ ]:


def crear_grupo_estadio_II (data):
    filtro = (
        (data['Stage'] == 'II')
      )
    grupo_estadio_II = data[filtro]
    return grupo_estadio_II


# In[ ]:


grupo_estadio_II=crear_grupo_estadio_II  (data)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['grupo_estadio_II'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:





# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['T'])
print("Tabla de Contingencia:\n", tabla)


# T1a y b= NO; T2 el 87% no recayo ver si hay otras variables que influyeron en al 13% que si recayo ; T3a: 43% recayo; T3b y T4 recayeron

# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:





# In[ ]:





# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['N'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['M'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Pathology'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Smoking'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Hx Smoking'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Hx Radiothreapy'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Gender'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Focality'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Adenopathy'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Thyroid Function'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# Creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Physical Examination'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


data['Response'] = data['Response'].replace({'Excellent': 'Si', 'Biochemical Incomplete': 'No', 'Indeterminate': 'No', 'Structural Incomplete': 'No'})
tabla = pd.crosstab(data['Recurred'], data['Response'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# Chi-cuadrado code
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


data['Response'] = data['Response'].replace({'Excellent': 'Si', 'Biochemical Incomplete': 'No', 'Indeterminate': 'Si', 'Structural Incomplete': 'No'})
tabla = pd.crosstab(data['Recurred'], data['Response'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# ### Analizar otras estadisticas:
# 
# 3. Respuesta al Tratamiento según Recurrencia

# In[ ]:


# creamos tabla de contingencia
tabla = pd.crosstab(data['Recurred'], data['Response'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# aplicamos Chi-square
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# La prueba de Chi-cuadrado muestra una asociación altamente significativa entre la recurrencian y la Respuesta al tratamiento (Chi2 = 309.47, p < 0.001).
# Esto indica que la recurrencia varía según la respuesta al TTo.


# In[ ]:


tabla = pd.crosstab(data['Recurred'], data['Gender'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


tabla = pd.crosstab(data['Recurred'], data['Stage'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


# creamos tabla de contingencia
data['Thyroid Function'] = data['Thyroid Function'].replace({'Euthyroid': 'No', 'Clinical Hyperthyroidism': 'Si', 'Subclinical Hypothyroidism': 'Si', 'Clinical Hypothyroidism': "Si", 'Subclinical Hyperthyroidism': "Si"})
tabla = pd.crosstab(data['Recurred'], data['Thyroid Function'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


# aplicamos Chi-square
chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# 

# In[ ]:


tabla = pd.crosstab(data['Recurred'], data['Stage'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# In[ ]:


data['Stage'] = data['Stage'].replace({'I': 'Bajo', 'II': 'Bajo', 'III': 'Bajo', 'IVA': 'Alto', 'IVB': 'Alto'})
tabla = pd.crosstab(data['Recurred'], data['Stage'])
print("Tabla de Contingencia:\n", tabla)


# In[ ]:


chi2, p_value, _, _ = chi2_contingency(tabla)

print("\nResultado Chi-Cuadrado:")
print("Chi2 =", chi2, ", p =", p_value)


# # **CREACION DE NUEVA VARIABLE**

# In[ ]:


# Librerías básicas
import pandas as pd
import numpy as np


# In[ ]:


# 1. Copiamos el dataframe base
df = data.copy("/content/drive/MyDrive/Copia de Thyroid_Diff.csv")


# In[ ]:


# 2. Creamos la nueva variable 'probabilidad_recurrencia' según la edad
# Regla: si Age >= 55 → 'alta_probabilidad', si Age < 55 → 'baja_probabilidad'
df['probabilidad_recurrencia'] = np.where(df['Age'] >= 55, 'alta_probabilidad', 'baja_probabilidad')


# In[ ]:


# 3. Podemos refinar la variable combinando con otros factores:
# Ejemplo: si es fumador (Smoking == 'Yes') y Age > 55 → 'muy_alta_probabilidad'

df['probabilidad_recurrencia'] = np.where(
    (df['Age'] >= 55) & (df['Smoking'] == 'Yes'),
    'muy_alta_probabilidad',
    df['probabilidad_recurrencia']  # Mantiene el valor anterior
)


# In[ ]:


# 4. Considerar 'Gender': si es masculino y tiene alta edad → más riesgo
df['probabilidad_recurrencia'] = np.where(
    (df['Age'] >= 55) & (df['Gender'] == 'Male'),
    'muy_alta_probabilidad',
    df['probabilidad_recurrencia']
)


# In[ ]:


# 5. Incorporamos la 'Thyroid Function' como otro criterio
# Ejemplo: si la función tiroidea es 'Abnormal' y tiene alta edad → riesgo mayor

df['probabilidad_recurrencia'] = np.where(
    (df['Age'] >= 55) & (df['Thyroid Function'] == 'Abnormal'),
    'muy_alta_probabilidad',
    df['probabilidad_recurrencia']
)


# In[ ]:


# 6. Incorporamos la 'Focality' (campo 'Focality' o 'Focal' según el dataset)
# Si tiene 'Multifocality' → mayor probabilidad de recurrencia

df['probabilidad_recurrencia'] = np.where(
    df['Focality'] == 'Multifocal',
    'muy_alta_probabilidad',
    df['probabilidad_recurrencia']
)


# In[ ]:


# 7. Visualizamos la distribución de la nueva variable
print(df['probabilidad_recurrencia'].value_counts())


# In[ ]:


# 8. Opcional: transformar en variable numérica si quiere incluirla en modelos de ML
df['probabilidad_recurrencia_bin'] = df['probabilidad_recurrencia'].map({
    'baja_probabilidad': 0,
    'alta_probabilidad': 1,
    'muy_alta_probabilidad': 2 #poder asignarle los nombres y variables que mejor se adapten
})


# In[ ]:


# Chequeo del dataframe actualizado
df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#27 DE FEBRERO REGRESION LOGISTICA MULTINOMINAL:


# In[ ]:


# Importar librerías necesarias
import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy  # Permite usar fórmulas como en R


# In[ ]:


#  Verificar valores nulos y eliminarlos o imputarlos
print("Valores nulos en el dataset:\n", data.isnull().sum())
data = data.dropna()  # Opcional: También se puede usar data.fillna(metodo)


# In[ ]:


# 1ro NO USAR
# Y=Risk


# In[ ]:


# Convertir variable de respuesta 'Risk' en categórica
data['Risk'] = data['Risk'].astype('category')


# In[ ]:


# Verificar balance de clases en la variable dependiente
print("Distribución de clases en 'Risk':\n", data['Risk'].value_counts())


# In[ ]:


# Definir la fórmula 'Risk ~ .' y convertir con patsy
formula = "Risk ~ Age + Stage + Recurred + Response"  # Se eliminan posibles variables colineales
y, X = patsy.dmatrices(formula, data, return_type="dataframe")



# In[ ]:


# si dos o más variables están muy correlacionadas (colinealidad alta, típicamente >0.7 u 0.8), puede:
# Distorsionar la interpretación de los coeficientes del modelo.
# Generar inestabilidad en las estimaciones.
# Afectar la precisión del modelo (problemas de multicolinealidad).


# In[ ]:


# Verificar colinealidad antes de entrenar el modelo
corr_matrix = X.corr().abs()
print("\nMatriz de correlación:\n", corr_matrix)



# In[ ]:


#Recurred[T.Yes] y Response[T.Structural Incomplete] tienen una correlación alta (0.86). Estas dos variables explican casi lo mismo.
# Podríamos elegir eliminar una, combinarla o analizarla mejor antes de incluirlas juntas.

# Otras correlaciones fuertes:
#Recurred[T.Yes] y Response[T.Excellent]: 0.67
#Response[T.Excellent] y Response[T.Structural Incomplete]: 0.60


# In[ ]:


# Ajustar el modelo de Regresión Logística Multinomial
modelo = sm.MNLogit(y, X)
resultado = modelo.fit()



# In[ ]:





# In[ ]:


#  Mostrar resultados
print(resultado.summary())



# In[ ]:


# Extraer coeficientes y p-values
print("\nCoeficientes del Modelo:\n", resultado.params)
print("\nP-values:\n", resultado.pvalues)


# In[ ]:


# Calcular Odds Ratios
odds_ratios = np.exp(resultado.params)
print("\nOdds Ratios:\n", odds_ratios)


# In[ ]:


# El modelo explota cuando calcula exponenciales (exp(X)) y luego falla al dividir valores.
# Esto genera NaNs en los coeficientes, errores estándar, z-scores y p-valores.


# In[ ]:


# Causas:
#  Multicolinealidad extrema correlaciones altas entre Recurred y Response. Esto afecta el cálculo de los coeficientes y puede llevar a inestabilidades numéricas.
# Escalado de variables Si la variable Age tiene una escala muy diferente a las dummies (Stage, Response), puede contribuir a problemas numéricos.


# In[ ]:


# soluciones: Revisar multicolinealidad con VIF
# Importar el módulo de VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Crear un DataFrame para almacenar los VIFs
vif_data = pd.DataFrame()

# Nombre de las variables predictoras en X
vif_data["feature"] = X.columns

# Calcular el VIF para cada predictor
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Mostrar la tabla con los VIFs
print(vif_data)


# In[ ]:


# Las variables Recurred[T.Yes] y Response[T.Excellent]/[Structural Incomplete] están fuertemente correlacionadas y generan multicolinealidad que podría estar rompiendo tu modelo MNLogit.


# In[ ]:


# agregar regularizacion para soportar que haya colinearidad
modelo = sm.MNLogit(y, X)
resultado = modelo.fit_regularized()
print(resultado.summary())


# ### Descripción de cada ítem:
# 
# | **Elemento**              | **Significado**                                                                 |
# |---------------------------|---------------------------------------------------------------------------------|
# | **Dep. Variable: y**      | La variable dependiente del modelo (`Risk`).                                   |
# | **No. Observations: 383** | Cantidad total de observaciones usadas (número de pacientes o casos).          |
# | **Model: MNLogit**        | Tipo de modelo ajustado: Regresión Logística Multinomial.                      |
# | **Df Residuals: 363**     | Grados de libertad residuales (n = 383 - 20 parámetros estimados).             |
# | **Method: MLE**           | Método de estimación: Máxima Verosimilitud (Maximum Likelihood Estimation).    |
# | **Df Model: 18**          | Número de parámetros independientes estimados en el modelo.                    |
# | **Pseudo R-squ.: 0.5139** | Indica el ajuste del modelo (≈ 51% de la variabilidad explicada).              |
# | **Log-Likelihood: -156.34** | Valor de la verosimilitud final; más alto (menos negativo) = mejor ajuste.   |
# | **converged: True**       | El algoritmo de optimización encontró una solución estable.                    |
# | **LL-Null: -321.60**      | Log-Likelihood del modelo nulo (sin predictores, solo intercepto).             |
# | **LLR p-value: 2.446e-59**| P-valor de la prueba de razón de verosimilitud → el modelo es significativo.   |
# 
# ---
# 

# In[ ]:


# Tenemos un Pseudo R-squared = 0.5139, lo cual sugiere que el modelo explica aproximadamente el 51% de la variabilidad de Risk


# In[ ]:


# Interpretación del modelo MNLogit para y = Risk[Intermediate]
# --------------------------------------------------------------------

# Intercept
# -> Coef: 15.88
# -> Interpretación: Muy alto e inestable. Probable separación perfecta o mal ajuste.

# Stage[T.II]
# -> Coef: -2.14
# -> Interpretación: Estar en Stage II reduce la probabilidad (log-odds) de pertenecer al grupo Intermediate
#    respecto al grupo de referencia (High Risk).
# -> P-valor: 0.016 => Es estadísticamente significativo.
# -> Confiable, aunque el tamaño del efecto es grande.

# Stage[T.III]
# -> Coef: -4.75
# -> Interpretación: Estar en Stage III reduce aún más la probabilidad de estar en Intermediate vs High.
# -> P-valor: 0.004 => También es estadísticamente significativo.
# -> Efecto fuerte y relevante.

# Stage[T.IVA] y Stage[T.IVB]
# -> Coefs: -78.11 y -90.24
# -> Interpretación: Coeficientes extremadamente negativos, con errores estándar enormes.
#    Esto sugiere que hay problemas serios en la estimación (pocos datos o separación perfecta).
# -> P-valor: 1.000 => No significativo. Mala estimación.

# Recurred[T.Yes]
# -> Coef: -15.92
# -> Interpretación: Coeficiente inestable con p-valor 0.998.
# -> Problemas similares a los anteriores. Inconfiable.

# Response[T.Excellent]
# -> Coef: 50.61
# -> std err: nan
# -> Interpretación: Falló la estimación. No se puede interpretar porque no hay datos o hay separación perfecta.

# Response[T.Indeterminate]
# -> Coef: 23.91
# -> std err: 1.21e+05
# -> Interpretación: Coeficiente enorme pero con error descomunal.
# -> P-valor: 1.000 => No significativo. No sirve.

# Response[T.Structural Incomplete]
# -> Coef: 0.01
# -> Interpretación: Efecto nulo sobre la probabilidad de estar en Intermediate.
# -> P-valor: 0.990 => No significativo.

# Age
# -> Coef: 0.051
# -> Interpretación: A mayor edad, ligera tendencia a estar en Intermediate en lugar de High.
# -> P-valor: 0.055 => Casi significativo. Puede tener valor predictivo en modelos futuros.

# --------------------------------------------------------------------

# RESUMEN:
# 1. Variables confiables: Stage[T.II], Stage[T.III], Age (borderline).
# 2. Problemas de estimación con Stage[T.IV], Recurred y Response.
# 3. Solución recomendada: Simplificar el modelo, agrupar categorías y probar regresión ordinal si Risk es ordenado.


# In[ ]:


# Interpretación del modelo MNLogit para y = Risk[Low]
# --------------------------------------------------------------------

# Intercept
# -> Coef: 16.68
# -> Interpretación: Valor base en log-odds para pertenecer al grupo Low en lugar de la categoría de referencia (probablemente High).
# -> Coeficiente alto e inestable (std err: 5086.39). Señal de separación perfecta o sobreajuste.

# Stage[T.II]
# -> Coef: -6.65
# -> Interpretación: Estar en Stage II reduce significativamente la probabilidad (log-odds) de ser Low Risk frente a High Risk.
# -> P-valor: 0.000 => Altamente significativo.
# -> Efecto fuerte y confiable.

# Stage[T.III]
# -> Coef: -81.40
# -> Interpretación: Coeficiente extremadamente negativo, lo que sugiere problemas en la estimación.
# -> std err: 5.8e+16 y p-valor: 1.000 => Mala estimación, no confiable.

# Stage[T.IVA]
# -> Coef: -75.36
# -> std err: 2.98e+15
# -> Interpretación: Valor muy inestable, sin significancia. Mala estimación.

# Stage[T.IVB]
# -> Coef: -85.12
# -> std err: 1.85e+17
# -> Interpretación: Igual que los anteriores. Inestabilidad total, no confiable.

# Recurred[T.Yes]
# -> Coef: -18.36
# -> std err: 5086.39
# -> P-valor: 0.997 => Sin significancia. Probable redundancia o colinealidad.

# Response[T.Excellent]
# -> Coef: 51.37
# -> std err: nan
# -> Interpretación: No se pudo calcular. Problema de separación perfecta o insuficiencia de datos.

# Response[T.Indeterminate]
# -> Coef: 22.91
# -> std err: 1.21e+05
# -> P-valor: 1.000 => Estimación inestable y sin significado estadístico.

# Response[T.Structural Incomplete]
# -> Coef: -0.90
# -> std err: 1.32
# -> P-valor: 0.497 => No significativo. El efecto es bajo y poco relevante.

# Age
# -> Coef: 0.0786
# -> Interpretación: Por cada año adicional de edad, aumenta ligeramente la probabilidad de estar en Low Risk en vez de High Risk.
# -> P-valor: 0.009 => Significativo.
# -> Efecto pequeño, pero confiable.

# --------------------------------------------------------------------

# RESUMEN PARA Risk[Low]:
# 1. Variables significativas:
#    - Stage[T.II] (reducción de odds).
#    - Age (incremento de odds).
# 2. Variables problemáticas:
#    - Stage[T.III], Stage[T.IVA], Stage[T.IVB]: Coeficientes extremos e inestables.
#    - Recurred y Response variables: Algunos coeficientes y errores estándar indican separación perfecta o multicolinealidad.
# 3. Recomendación:
#    - Simplificar el modelo: eliminar o agrupar categorías problemáticas.
#    - Considerar regresión ordinal si Risk tiene orden lógico.


# In[ ]:


# 2do
# Y= Recurred
# Age + Risk + Stage + Response + smoking + focalidad + subtipo histologico + funcion tiroidea + TNM + gender + adenopatias
# antecedentes de radio o habito tabaquico muchas diferencia en la distribucion no las uso


# In[ ]:


# Verificar balance de clases en la variable dependiente
print("Distribución de clases en 'Recurred':\n", data['Recurred'].value_counts())


# In[ ]:


# Verificar balance de clases antecedente de radioterapia
print("Distribución de clases en 'Hx Radiothreapy':\n", data['Hx Radiothreapy'].value_counts())


# In[ ]:


# Verificar balance de clases en la variable antecedente tabaquico.
print("Distribución de clases en 'Hx Smoking':\n", data['Hx Smoking'].value_counts())


# In[ ]:


# Verificar balance de clases en la variable focalidad
print("Distribución de clases en 'Focality':\n", data['Focality'].value_counts())


# In[ ]:


# Verificar balance de clases en la variable T
print("Distribución de clases en 'T':\n", data['T'].value_counts())


# In[ ]:


# Verificar balance de clases en la variable adenopatias
print("Distribución de clases en 'Adenopathy':\n", data['Adenopathy'].value_counts())


# In[ ]:


# Verificar balance de clases en la variable funcion tiroidea
print("Distribución de clases en 'Thyroid Function':\n", data['Thyroid Function'].value_counts())


# In[ ]:


# Definir la fórmula 'Recurred ~ .' y convertir con patsy
formula = "Recurred ~ Age + Risk + Stage + Response + Focality + Smoking + Pathology + Gender + Adenopathy "  # VER CON GISE No me deja agregar mas variables (M, Thyroid Function)
y, X = patsy.dmatrices(formula, data, return_type="dataframe")


# In[ ]:


# Verificar colinealidad antes de entrenar el modelo
corr_matrix = X.corr().abs()
print("\nMatriz de correlación:\n", corr_matrix)


# In[ ]:


modelo = sm.MNLogit(y, X)
resultado = modelo.fit()


# In[ ]:


#  Mostrar resultados
print(resultado.summary())


# In[ ]:


# Extraer coeficientes y p-values
print("\nCoeficientes del Modelo:\n", resultado.params)
print("\nP-values:\n", resultado.pvalues)


# In[ ]:


# Calcular Odds Ratios
odds_ratios = np.exp(resultado.params)
print("\nOdds Ratios:\n", odds_ratios)


# In[ ]:


# IMPECABLE EVALUACION MARCE, PASADA AL WORD

# Tenemos un Pseudo R-squared = 0.8559, lo cual sugiere que el modelo explica aproximadamente el 86% de la variabilidad de Recurred

# Interpretación del modelo MNLogit para y = Recurred

# ------------------------------------------------

# Intercept
# -> Coef: 16.244414
# -> Interpretación: Valor alto, probablemente la separacion sea buena en funcion de la explicacion desde la Patologia????
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# STAGE. GRUPO DE REFERENCIA STAGE I
# Stage[T.II]
# coef: 2.054684. Interpretacion: Estar en Stage II aumenta la probabilidad (log-odds: 7.804375e+00)
# en relacion al estadio I de pertenecer al grupo recaidos
# p value: 0.089585 => estadísticamente significativo.

# Stage[T.III]
# coef: 17.627231 . Interpretacion: Estar en Stage III aumenta la probabilidad (log-odds:  4.522817e+07), en relacion al grupo de referencia
# de pertenecer al grupo recaidos
# p value: 0.999894 => No es estadísticamente significativo.

# Stage[T.IVA]
# coef: 8.170514. Interpretacion: Estar en Stage IVA aumenta la probabilidad (log-odds:  3.535160e+03), en relacion al grupo de referencia
# de pertenecer al grupo recaidos
# p value: 0.999983 => No es estadísticamente significativo.

# Stage[T.IVB]
# coef: 2.051877. Interpretacion: Estar en Stage IVA aumenta la probabilidad (log-odds:  7.782497e+00), en relacion al grupo de referencia
# de pertenecer al grupo recaidos
# p value:  0.999955 => No es estadísticamente significativo.

# LA MENOR PROBABILIDAD DE PERTENECER AL GRUPO RECAIDOS DE LOS ESTADIO IV A Y B CON RESPECTO AL ESTADIO III
# ES POR EL N BAJO?

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# RISK GRUPO DE REFERENCIA RISK ALTO
# Risk[T.Intermediate]
# Coef: -16.641584. Coeficiente (cuando no esta en el intercepto) Tiene que ver con el grado de adecuacion del grupo que quiere descubrir el modelo.[si toma como referencia recaido = SI, lo que me dice el coeficiente es cuanto me alejo de ese grupo]
#Interpretacion: El riesgo intermedio disminuye la probabilidad (log-odds:  5.924502e-08), en relacion al grupo de referencia
# de pertenecer al grupo recaidos
# p value: 0.997313 => No es estadísticamente significativo.

# Risk[T.low]
# Coef: -19.407220.
#Interpretacion: El riesgo bajo disminuye la probabilidad (log-odds: 3.728648e-09), en relacion al grupo de referencia
# de pertenecer al grupo recaidos
# p value: 0.996866 => No es estadísticamente significativo.

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::.

# RESPONSE GRUPO DE REFERENCIA RESPUESTA BIOQUIMICA INCOMPLETA
# Response[T.Excellent]
# Coef:  -4.746116. Interpretacion: La respuesta excelente al TTo disminuye la probabilidad (log-odds:  8.685366e-03),
# en relacion al grupo de referencia de pertenecer al grupo recaidos
# p value: 0.000323 => Estadísticamente significativo.

# Response[T.Indeterminate]
# Coef:  -2.907076. Interpretacion: La respuesta Indeterminada al TTo disminuye la probabilidad (log-odds:  5.463527e-02),
# en relacion al grupo de referencia de pertenecer al grupo recaidos
# p value:  0.002603 => Estadísticamente significativo.

# Response[T.Structural Incomplete]
# Coef:  4.050960. Interpretacion: La respuesta estructural incompleta al TTo aumenta la probabilidad (log-odds:  5.745257e+01),
# en relacion al grupo de referencia de pertenecer al grupo recaidos
# p value:  0.000841 => Estadísticamente significativo.

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::.::::::::::::

# Age
# -> Coef: 0.017927
# -> Interpretación:
      # La edad no influye en la probabilidad de pertenecer al grupo recaidos?. (log-odds: 1.018088e+00)???
      # Por cada año adicional de edad, las probabilidades de recaída aumentan en un 0.1%.???
#    (Odds Ratio cercano a 1 indica un efecto pequeño).
# -> P-valor: 0.558161 => Estadisticamente no significativo

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# SMOKING GRUPO DE REFERENCIA NO FUMADORES
# Smoking (T. yes)
# Coef: 0.149215
# -> Interpretación:
      # fumar aumenta discretamente la probabilidad, en relacion al grupo de referencia,
      # de pertenecer al grupo recaidos?. (log-odds: 1.160923e+00)
# p value:  0.910012 estadisticamente no significativo

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# FOCALLITY GRUPO DE REFRENCIA MULTIFOCALIDAD. M
# Focalidad [T.unifocal]
# Coef:  0.892985. Interpretacion: Las lesiones unifocales tiene mayor probabilidad (log-odds:  2.442410e+00),
# de pertenecer al grupo recaidos. VER CON GISELA POCO LOGICO
# p value:   0.265315 => Estadísticamente NO significativo.

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::...

# PATHOLOGY. GRUPO DE REFERENCIA CARCINOMA FOLICULAR
# Subtipohistologico [T.Hurthel cell]
# Coef:-3.284357
# -> Interpretación: El subtipo histologico T ce celula de Hurtle disminuye la probabilidad, (log-odds:3.746468e-02),
# en relacion al grupo de referencia de pertenecer al grupo recaidos.
# p value:   0.072457 => Estadísticamente NO significativo.

# Subtipohistologico [T.micropapilar]
# Coef:  -17.384372
# -> Interpretación: El subtipo histologico micropapilar disminuye significativamente la probabilidad,
# en relacion al grupo de referencia, de pertenecer al grupo recaidos. (log-odds: 2.818793e-08)
# p value:   0.999045 => Estadísticamente NO significativo.

# Subtipohistologico [T.papilar]
# Coef: 0.176861
# -> Interpretación: El subtipo histologico papilar no disminuye significativamente la probabilidad,
# en relacion al grupo de referencia, de pertenecer al grupo recaidos. (log-odds:  1.193466e+00)
# p value:   0.854790 => Estadísticamente NO significativo.

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# GENDER GRUPO DE REFERENCIA FEMENINO

#Gender [T.M]
# Coef:0.766565
# -> Interpretación: El genero NO INFLUYE SIGNIFICATIVAMENTE EN LAS PROBABILIDADES DE RECAIDA, (log-odds:2.152360e+00),
# en relacion al grupo de referencia de pertenecer al grupo recaidos.
# p value:   0.393731 => Estadísticamente NO significativo.

# RESUMEN:
# Todas las variables son confiables:


# ---
# # **Agregado de funcion tiroidea**

# In[ ]:


data.columns


# In[ ]:


# Definir la fórmula 'Recurred ~ .' y convertir con patsy
formula = "Recurred ~ Risk + Stage + Response + Focality + Pathology + Gender + Thyroid Function"   # VER CON GISE No me deja agregar mas variables (M, Thyroid Function)
y, X = patsy.dmatrices(formula, data, return_type="dataframe")


# El error que te está dando patsy.dmatrices es por la variable Thyroid Function que contiene un espacio en su nombre. patsy interpreta eso como dos cosas distintas (una variable llamada Thyroid y otra llamada Function) y por eso lanza un error de SyntaxError: invalid syntax.
# 
# Solución rápida:
# Rodeá el nombre de la variable con Q() (de quote) para que patsy entienda que es un nombre literal con espacio.

# In[ ]:


formula = "Recurred ~ Risk + Age + Stage + Response + Focality + Pathology + Gender + Q('Thyroid Function')+ Smoking + Adenopathy+ T+ N"
y, X = patsy.dmatrices(formula, data, return_type="dataframe")


# In[ ]:


# Verificar colinealidad antes de entrenar el modelo
corr_matrix = X.corr().abs()
print("\nMatriz de correlación:\n", corr_matrix)


# In[ ]:


modelo = sm.MNLogit(y, X)
resultado = modelo.fit()


# In[ ]:


print(resultado.summary())


# In[ ]:


print("\nCoeficientes del Modelo:\n", resultado.params)
print("\nP-values:\n", resultado.pvalues)


# In[ ]:


odds_ratios = np.exp(resultado.params)
print("\nOdds Ratios:\n", odds_ratios)


# In[ ]:


# MNLogit para y = Recurred

# Pseudo R-squared =  0.8817, el modelo explica aproximadamente el 88% de la variabilidad de Recurred

# Intercept
# -> Coef: 18.1124
# -> Interpretación:

# STAGE. GRUPO DE REFERENCIA STAGE I
# Stage[T.II]
# coef:  3.449049. Interpretacion: Estar en Stage II aumenta discretamente la probabilidad (log-odds: 3.147045e+01)
# en relacion al estadio I de pertenecer al grupo recaidos
# p value: 0.049932 => EN EL LIMITE estadísticamente significativo.

# Stage[T.III]
# coef: 17.665088 . Interpretacion: Estar en Stage III aumenta la probabilidad (log-odds:  3.147045e+01)
# en relacion al estadio I de pertenecer al grupo recaidos
# p value: 0.999622 => No es estadísticamente significativo.

# Stage[T.IVA]
# coef: 5.528484. Interpretacion: Estar en Stage IVA aumenta la probabilidad (log-odds:  2.517619e+02)
# en relacion al estadio I de pertenecer al grupo recaidos
# p value: 0.999679 => No es estadísticamente significativo.

# Stage[T.IVB]
# coef: 2.325498. Interpretacion: Estar en Stage IVA aumenta la probabilidad (log-odds:  1.023177e+01)
# en relacion al estadio I de pertenecer al grupo recaidos
# p value:  0.999840 => No es estadísticamente significativo.

# LA MENOR PROBABILIDAD DE PERTENECER AL GRUPO RECAIDOS DE LOS ESTADIO IV A Y B CON RESPECTO AL ESTADIO III
# ES POR EL N BAJO? ESTO SE PUEDE SOLUCIONAR DE ALGUNA MANERA. LOS N DEBERIAN DE SER MAS O MENOS IGUALES?. PERO NO ES LA VIDA REAL

# CONCLUSION: LA VARIABLE ESTADIO NO MUESTRA UNA ASOCIACION ESTADISTICAMENTE SIGNIFICATIVA CON LA PROBABILIDAD DE RECAIDA

# Aunque los coeficientes fueron altos 8VER DCON GISELA. eJ Stage[T.III] coef: 17.665088),
# los valores p no respaldan una asociación significativa en la mayoría de los estadios avanzados.

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# RISK GRUPO DE REFERENCIA RIESGO ALTO
# Risk[T.Intermediate]
# Coef: -22.483941. Coeficiente (cuando no esta en el intercepto) Tiene que ver con el grado de adecuacion del grupo que quiere descubrir el modelo.[si toma como referencia recaido = SI, lo que me dice el coeficiente es cuanto me alejo de ese grupo]
#Interpretacion: El riesgo intermedio disminuye la probabilidad (log-odds:  1.719288e-10), en relacion al grupo de referencia
# de pertenecer al grupo recaidos
# p value: 0.999653 => No es estadísticamente significativo.

# Risk[T.low]
# Coef: -24.001393.
#Interpretacion: El riesgo bajo disminuye la probabilidad (log-odds: 3.769878e-11), en relacion al grupo de referencia
# de pertenecer al grupo recaidos
# p value: 0.999630 => No es estadísticamente significativo.

# CONCLUSION: LA VARIABLE RIESGO NO MUESTRA UNA ASOCIACION ESTADISTICAMENTE SIGNIFICATIVA CON LA PROBABILIDAD DE RECAIDA
# EL TIPO QUE MENOS SE ASOCIA CON RECAIDA ES EL RIESGO BAJO


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# RESPONSE GRUPO DE REFERENCIA RTA BIOQUIMICA INCOMPLETA

# Response[T.Excellent]
# Coef:  -5.573510. Interpretacion: La respuesta excelente al TTo disminuye la probabilidad (log-odds:  3.797131e-03),
# en relacion al grupo de referencia de pertenecer al grupo recaidos
# p value: 0.000536 => Estadísticamente significativo.

# Response[T.Indeterminate]
# Coef:  -3.182674. Interpretacion: La respuesta Indeterminada al TTo disminuye la probabilidad (log-odds:  4.147460e-02),
# en relacion al grupo de referencia De pertenecer al grupo recaidos
# p value:  0.005844 => Estadísticamente significativo.

# Response[T.Structural Incomplete]
# Coef:  3.814524. Interpretacion: La respuesta estructural incompleta al TTo aumenta la probabilidad (log-odds:  4.535517e+01),
#en relacion al grupo de referencia de pertenecer al grupo recaidos
# p value:  0.004951 => Estadísticamente significativo.

# CONCLUSION: LA VARIABLE RESPUESTA AL TRATAMIENTO MUESTRA UNA ASOCIACION ESTADISTICAMENTE SIGNIFICATIVA CON LA PROBABILIDAD DE RECAIDA
# EL TIPO QUE MAS SE ASOCIA CON RECAIDA ES LA RESPUESTA ESTRUCTURAL INCOMPLETA

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Age
# -> Coef: 0.020277
# -> Interpretación:
      # La edad no influye en la probabilidad de pertenecer al grupo recaidos?. (log-odds:1.020484e+00)???
      # Por cada año adicional de edad, las probabilidades de recaída aumentan en un 0.2%.???
#    (Odds Ratio cercano a 1 indica un efecto pequeño).
# -> P-valor: 0.594111 => Estadisticamente no significativo

# CONCLUSION: LA VARIABLE EDAD NO MUESTRA UNA ASOCIACION ESTADISTICAMENTE SIGNIFICATIVA CON LA PROBABILIDAD DE RECAIDA
# ESTA CONCLUSION NO COINCIDE CON LAS PREVIAS QUE DECIAN QUE LA EDAD SE SAOCIABA CON LA RECAIDA
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# SMOKIN GRUPO DE REFERENCIA NO FUMADORES
# Smoking (T. yes)
# Coef: -0.828598
# -> Interpretación:
      # fumar disminuye la probabilidad, en relacion al grupo de referencia,
      # de pertenecer al grupo recaidos?. (log-odds: 4.366613e-01)
# encoding inverso??? ( fumar=0, no fumar=1).

# p value:  0.608947 estadisticamente no significativo

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# FOCALIDAD GRUPO DE REFERENCIA MULTIFOCAL
# Focalidad [T.unifocal]
# Coef:  0.851675. Interpretacion: las lesiones unifocales aumentan la probabilidad (log-odds:  2.343569e+00),
# de pertenecer al grupo recaidos. encoding inverso???
# p value:   0.401787 => Estadísticamente NO significativo.

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# SUBTIPO HISTOLOGICO. GRUPO DE REFERENCIA CARCINOMA FOLICULAR.
# Subtipohistologico [T.Hurthel cell]. N=
# Coef:-3.714659
# -> Interpretación: Los pacientes con subtipo histologico micropapilar tienen menos probabilidades que aquellos con
# subtipo folicular de pertenecer al grupo recaidos.(log-odds: 2.436376e-02)
# p value:   0.082174 => Estadísticamente significativo.

# Subtipohistologico [T.micropapilar]
# Coef:  -14.972184
# -> Interpretación: Los pacientes con subtipo histologico micropapilar tienen menos probabilidades que aquellos con
# subtipo folicular de pertenecer al grupo recaidos. (log-odds: 3.145309e-07)
# p value:   0.997331 => Estadísticamente NO significativo.

# Subtipohistologico [T.papilar]
# Coef:-0.906611. COEF PEQUEÑO CERCANO A 1 SIN SIGNIFICANCIA
# -> Interpretación: Los pacientes con subtipo histologico papilar tienen menos probabilidades que aquellos con
# subtipo folicular de pertenecer al grupo recaidos.(log-odds:  4.038909e-01)
# p value: 0.471118 => Estadísticamente NO significativo.

# CONCLUSION: LA VARIABLE SUBTIPO HISTOLOGICO NO MUESTRA UNA ASOCIACION ESTADISTICAMENTE SIGNIFICATIVA CON LA PROBABILIDAD DE RECAIDA
# DE LOS SUBTIPOS HISTOLOGICOS EL QUE MENOS SE ASOCIA CON RECAIDA ES EL MICROPAPILAR
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# GENDER GRUPO DE REFERENCIA FEMENINO
#Gender [T.M]
# Coef:1.981810
# -> Interpretación: El genero MASCULINO AUMENTA LAS PROBABILIDADES DE PERTENECER AL GRUPO RECAIDOS, (log-odds:7.255867e+00),
# en relacion al grupo de referencia de pertenecer al grupo recaidos.
# p value:   0.087600 => Estadísticamente significativo.

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# FUNCION TIROIDEA GRUPO DE REFERENCIA HIPERTIROIDISMO CLINICO

# [T.Clinical Hypothyroidism]
# Coef: 5.243415
# -> Interpretación 1: los pacientes hipotriroideos tienen mayor probabilidad, en relacion al grupo hipertiroideos,
# de pertenecer al grupo recaidos. (log-odds:  1.535832e+01).
# -> Interpretación 2: Es mas frecuente que los pacientes del grupo recaido cursen con hipotiroidimo que con hipertiroidismo (log-odds:  1.893155e+02)
# p value:  0.045310 => Estadísticamente NO significativo.

# [T.Euthyroid]
# Coef: 2.731657
# -> Interpretación 1: los pacientes eutiroideos tienen mayor probabilidad, en relacion al grupo hipertiroideos
# de pertenecer al grupo recaidos. (log-odds:  1.535832e+01)
# Interpretación 2: Es mas frecuente que los pacientes del grupo recaido cursen con eutiroidismo que con hipertiroidismo (log-odds:  1.893155e+02)
# p value:  0.178481 => Estadísticamente NO significativo.

# [T.Subclinical Hyperthyroi...
# Coef: -12.418171
# -> Interpretación 1: los pacientes con hipertiroidismo subclinico tienen menor probabilidad, en relacion al grupo hipertiroideos,
# de pertenecer al grupo recaidos. (log-odds:  4.044427e-06)
# Interpretación 2: Es menos frecuente que los pacientes del grupo recaido cursen con hipertiroidismo subclinico que con hipertiroidismo (log-odds:  1.893155e+02)
# p value: 0.999556=> Estadísticamente NO significativo.

# [T.Subclinical Hypothyroid...
# Coef: -2.829744
# -> Interpretación 1: los pacientes con hipotiroidismo subclinico tienen menor probabilidad, en relacion al grupo hipertiroideos
# de pertenecer al grupo recaidos. (log-odds:  5.902796e-02)
# Interpretación 2: Es menos frecuente que los pacientes del grupo recaido cursen con hipotiroidismo subclinico que con hipertiroidismo (log-odds:  1.893155e+02)
# p value:  0.630416 => Estadísticamente NO significativo.

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# T GRUPO DE REFRENCIA T1a

# T[T.T1b]
# Coef: 2.501315
# -> Interpretación: Los T1b tiene mayor probabilidad, en relacion al grupo T1a,
# de pertenecer al grupo recaidos. (log-odds: 1.219852e+01)
# p value:   0.784747 => Estadísticamente NO significativo.

# T[T.T2]
# Coef: 0.092084
# -> Interpretación: las probabilidades de recaida del grupo T2 son similarares al grupo T1a,
# (log-odds: 1.096457e+00)
# p value: 0.991902 => Estadísticamente NO significativo.


# T[T.T3a]
# Coef: 0.705022
# -> Interpretación: El T3a tiene mayor probabilidad, en relacion al grupo T1a,
# de pertenecer al grupo recaidos. (log-odds: 2.023891e+00)
# p value: => 0.937985 Estadísticamente NO significativo.


# T[T.T3b]
# Coef: 2.727243
# -> Interpretación: El T3b tiene mayor probabilidad, en relacion al grupo T1a,
# de pertenecer al grupo recaidos. (log-odds: 1.529067e+01)
# p value: => 0.766220 Estadísticamente NO significativo.

# T[T.T4a]
# Coef: 1.463610
# -> Interpretación: El T4a tiene mayor probabilidad, en relacion al grupo T1a,
# de pertenecer al grupo recaidos. (log-odds: 4.321532e+00)
# p value: => 0.931172 Estadísticamente NO significativo.

# T[T.T4b]
# Coef: -9.988178
# -> Interpretación: El T4b tiene menor probabilidad, en relacion al grupo T1a,
# de pertenecer al grupo recaidos. (log-odds: 4.593982e-05). RELACIONADO A UN MENOR N????
# p value: => 0.999845 Estadísticamente NO significativo.


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# N GRUPO DE REFERENCIA

RESUMEN:
# Todas las variables son confiables:


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# 3er
# Y= Recurred
# mismas variables que 2do PERO SIN RISK


# In[ ]:


data['Recurred'] = data['Recurred'].astype('category')


# In[ ]:


formula = "Recurred ~ Age + Gender+ Smoking + Stage + Focality + Q('Thyroid Function')+ T + N + Response + Pathology"
y, X = patsy.dmatrices(formula, data, return_type="dataframe")


# In[ ]:


# Verificar colinealidad antes de entrenar el modelo
corr_matrix = X.corr().abs()
print("\nMatriz de correlación:\n", corr_matrix)


# In[ ]:


corr_matrix = X.corr().abs()
print("\nMatriz de correlación:\n", corr_matrix)


# In[ ]:


modelo = sm.MNLogit(y, X)
resultado = modelo.fit()


# El error `LinAlgError: Singular matrix` al ajustar el modelo con `sm.MNLogit(y, X)` significa que **la matriz de diseño X no es invertible**. Esto ocurre cuando hay **colinealidad perfecta o casi perfecta entre algunas variables** (es decir, una variable puede expresarse como combinación lineal de otras).
# 
# ### Causas frecuentes del error "Singular matrix" en tu modelo:
# 
# 1. **Colinealidad alta entre variables dummy**  
#    - Por ejemplo, en tu matriz de correlación, vemos:
#      - `Pathology[T.Papillary]` y `Pathology[T.Micropapillary]` → 0.65
#      - `Response[T.Excellent]` y `Response[T.Structural Incomplete]` → 0.60
#      - `Smoking[T.Yes]` y `Gender[T.M]` → 0.62
# 
# 2. **Dummy trap**: si estás incluyendo **todas** las categorías de una variable categórica como dummies (sin eliminar una como referencia), entonces hay redundancia matemática.
# 
# 3. **Pocas observaciones para algunas categorías**  
#    - Si hay niveles de una variable categórica con pocos casos (por ejemplo `Pathology[T.Hurthel cell]` o `Stage[T.IVB]`), puede generar inestabilidad.
# 
# ---
# 
# ### Posibles soluciones:
# 
# #### 1. **Eliminar una categoría de cada variable categórica**
# Usá `drop_first=True` en la creación de dummies o asegurate de que `patsy` no esté codificando todas las categorías.
# 
# Ejemplo para `patsy`:
# ```python
# formula = "Recurred ~ C(Stage, Treatment(reference='I')) + C(Response, Treatment(reference='Biochemical Incomplete')) + C(Pathology, Treatment(reference='Folicular')) + ... "
# ```
# 
# #### 2. **Revisar y reducir colinealidad**
# Calculá el **VIF (Variance Inflation Factor)** y eliminá variables con VIF > 10.
# 
# ```python
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import pandas as pd
# 
# vif = pd.DataFrame()
# vif["feature"] = X.columns
# vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# print(vif)
# ```
# 
# #### 3. **Ajustar el modelo con regularización**
# Si queremos mantener todas las variables por motivos exploratorios, podés usar:
# 
# ```python
# resultado = modelo.fit_regularized()
# ```
# 
# Esto ayuda a manejar colinealidad y matrices singulares porque aplica penalizaciones (Ridge o Lasso) que suavizan los coeficientes.
# 
# ---
# 

# In[ ]:





# In[ ]:


formula = "Recurred ~ C(Pathology, Treatment(reference='Micropapillary')) + C(Response, Treatment(reference='Biochemical Incomplete')) + Age + Gender+ Smoking + Stage + Focality + Q('Thyroid Function')+ T + N"
y, X = patsy.dmatrices(formula, data, return_type="dataframe")


# In[ ]:





# In[ ]:


corr_matrix = X.corr().abs()
print("\nMatriz de correlación:\n", corr_matrix)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)


# In[ ]:





# In[ ]:





# ## Este Modelo tambien mostro colinearidad y error Singular Matrix asi que podes aplicar las mismas posibles soluciones que describimos mas arriba

# In[ ]:


# Convertir variable de respuesta 'Recurred' en categórica
data['Recurred'] = data['Recurred'].astype('category')


# In[ ]:


# Verificar balance de clases en la variable dependiente
print("Distribución de clases en 'Recurred':\n", data['Recurred'].value_counts())


# In[ ]:


# Definir la fórmula 'Recurred ~ .' y convertir con patsy
formula = "Recurred ~ Age + Stage + Response + Smoking + Focality + Pathology "
y, X = patsy.dmatrices(formula, data, return_type="dataframe")


# In[ ]:


# Verificar colinealidad antes de entrenar el modelo
corr_matrix = X.corr().abs()
print("\nMatriz de correlación:\n", corr_matrix)


# In[ ]:


# Ajustar el modelo de Regresión Logística Multinomial
modelo = sm.MNLogit(y, X)
resultado = modelo.fit()


# # **LogisticRegression()**

# In[ ]:


# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


# Cargar datos y seleccionar variables relevantes
data = pd.read_csv ("/content/drive/MyDrive/Copia de Thyroid_Diff.csv")


# In[ ]:


# Definir variables predictoras (X) y variable objetivo (y)
X = data[['Age', 'Stage', 'Risk', "Pathology", "Response", 'Gender','Smoking', "Thyroid Function", "Adenopathy", "T", "N" ]]  # Variables clínicas
y = data['Recurred']  # Variable objetivo (0 = No recayó, 1 = Recayó)


# In[ ]:


y


# In[ ]:


#  Convertir variables categóricas a numéricas
# y es solo la columna Recurred, podemos convertirla directamente:


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Definir la variable objetivo directamente desde el dataframe
y = data['Recurred']  # Esta es una serie, no un dataframe

# Crear el encoder y transformar 'Recurred'
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # 0 = No, 1 = Yes

# Verificar el resultado
print(y_encoded)


# In[ ]:


# El modelo LogisticRegression() de scikit-learn no acepta variables categóricas en string, como 'Stage', 'Risk', o 'Response'.
# Estos todavía tienen valores como 'I' o 'Low', y el modelo espera números (floats o ints).
# ya habiamos codificado y, ahora coficiamos x


# In[ ]:


X = data[['Age', 'Stage', 'Risk', "Pathology", "Response", 'Gender','Smoking', "Thyroid Function", "Adenopathy", "T","N", "M"]].copy()


# 

# In[ ]:


risk_map = {
    'Low': 0,
    'Intermediate': 1,
    'High': 2
}


# In[ ]:


data['Risk_Coded'] = data['Risk'].map(risk_map)


# In[ ]:


gender_map = {
    'M': 0,
    'F': 1,
}


# In[ ]:


data['Gender_Coded'] = data['Gender'].map(gender_map)


# In[ ]:


stage_map = {
    'I': 0,
    'II': 1,
    'III': 2,
    'IVA': 3,
    'IVB': 4,
}


# In[ ]:


data['Stage_Coded'] = data['Stage'].map(stage_map)


# In[ ]:


smoking_map = {
    'No': 0,
    'Yes': 1,
}


# In[ ]:


data['Smoking_Coded'] = data['Smoking'].map(smoking_map)


# In[ ]:


response_map = {
    'Excellent': 0,
    'Indeterminate': 1,
    'Structural Incomplete': 2,
    'Biochemical Incomplete': 3,

}


# In[ ]:


data['Response_Coded'] = data['Response'].map(response_map)


# In[ ]:


thyroid_function_map = {
    'Euthyroid': 0,
    'Clinical Hypothyroidism': 1,
    'Clinical Hyperthyroidism': 2,
    'Subclinical Hypothyroidism': 3,
    'Subclinical Hyperthyroidism': 4
}


# In[ ]:


data ['Thyroid Function_Coded'] = data['Thyroid Function'].map(thyroid_function_map)


# In[ ]:


data['Adenopathy'].unique()


# In[ ]:


adenopathy_map = {
    'No': 0,
    'Right': 1,
    'Left': 2,
    'Extensive': 3,
    'Bilateral': 4,
    'Posterior': 5,

}


# In[ ]:


data ['Adenopathy_Coded'] = data['Adenopathy'].map(adenopathy_map)


# In[ ]:


pathology_map = {
    'Micropapillary': 0,
    'Papillary': 1,
    'Follicular': 2,
    'Hurtle cell': 3,

}


# In[ ]:


data ['Pathology_Coded'] = data['Pathology'].map(pathology_map)


# In[ ]:


hx_smoking_map = {
    'No': 0,
    'Yes': 1,

}


# In[ ]:


data ['Hx Smoking_Coded'] = data['Hx Smoking'].map(hx_smoking_map)


# In[ ]:


print("Distribución de clases en 'T':\n", data['T'].value_counts())


# In[ ]:


T_map = {
    'TIa': 0,
    'TIb': 1,
    "T2": 2,
    "T3a": 3,
    "T3b": 4,
    "T4a": 5,
    "T4b": 6,


}


# In[ ]:


data ['T_Coded'] = data['T'].map(T_map)


# In[ ]:


print("Distribución de clases en 'N':\n", data['N'].value_counts())


# In[ ]:


N_map = {
    'N0': 0,
    'NIa': 1,
    'NIb': 2,
   }


# In[ ]:


data ['N_Coded'] = data['N'].map(N_map)


# In[ ]:


print("Distribución de clases en 'M':\n", data['M'].value_counts())


# In[ ]:


M_map = {
    'M0': 0,
    'M1': 1,

   }


# In[ ]:


data ['M_Coded'] = data['M'].map(M_map)


# In[ ]:


hx_radiothreapy_map = {
    'No': 0,
    'Yes': 1,

}


# In[ ]:


data ['Hx Radiothreapy_Coded'] = data['Hx Radiothreapy'].map(hx_radiothreapy_map)


# In[ ]:


X = data[["Age", "Stage_Coded", "Risk_Coded", "Pathology_Coded","Response_Coded", "Gender_Coded","Smoking_Coded", "Thyroid Function_Coded", "T_Coded", 'Adenopathy_Coded', "N_Coded", "M_Coded"]].copy()


# In[ ]:


for col in ['Age', 'Stage_Coded', 'Risk_Coded', "Pathology_Coded",'Response_Coded', 'Gender_Coded','Smoking_Coded', "Thyroid Function_Coded","T_Coded","N_Coded" ]:
    X[col] = encoder.fit_transform(X[col])


# In[ ]:


# una vez hecho el encoding completo, dividimos los datos y corremos el modelo


# In[ ]:


# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión logística
modelo = LogisticRegression(max_iter=1000)  # Aumentamos iteraciones por si tarda en converger
modelo.fit(X_train, y_train)


# In[ ]:


# Hacer predicciones en los datos de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
print("Precisión:", accuracy_score(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))


# In[ ]:


# Precisión (Accuracy): 95% de los casos totales fueron correctamente clasificados.


# In[ ]:


# Reporte de Clasificación:
# No (No recayó): Precisión (precision): 95% → De los que predijo "No", el 95% era correcto.
# Recall (sensibilidad): 98% → Detectó correctamente el 98% de los que realmente no recayeron. #no detecta recaidos porque la data es inconsistente o no es lo esperable medicamente
#Yes (Recayó): Precisión: 94% → De los que predijo "Yes", el 94% era correcto.
# Recall: 84% → Detectó el 84% de los que realmente recayeron.


# In[ ]:


# Balance: Buen desempeño, aunque el modelo se le escapan algunos casos de recaída (falsos negativos).


# In[ ]:


# Ver coeficientes de cada variable
coeficientes = modelo.coef_[0]
intercepto = modelo.intercept_[0]

# Asociar coeficientes con nombres de variables
for feature, coef in zip(X.columns, coeficientes):
    print(f"{feature}: {coef:.4f}")

print(f"\nIntercepto: {intercepto:.4f}")


# In[ ]:


# Coeficientes del modelo de regresión logística. COEFICIENTE HABLA DE LAS PROBABILIDADES
# ----------------------------------------------

# Age: 0.0063
# -> Interpretación:
#    Cada año adicional de edad aumenta ligeramente la probabilidad de recaída.
#    El efecto es pequeño (este dato de efecto pequeño se desprende de comparar los coeficientes entre las distintas variav+blaes.
# En este caso la edad el la variable con menor coeficiente respuesta, riesgo y N son las de mayor coeficiente).

# Stage:  1.1218

# -> Interpretación:
#    A medida que aumenta el estadio (por ejemplo, de I a IV), aumenta la probabilidad de recaída.
#    El coeficiente positivo indica que un estadio más avanzado incrementa los log-odds de recaída.

# Risk: 1.6987
# -> Interpretación:
#    A mayor valor de la variable 'Risk', aumenta la probabilidad de recaída.


# Response:  1.7805
# -> Interpretación:
#    Respuestas peores al tratamiento incrementan la probabilidad de recaída.
#    El coeficiente positivo indica que ciertas categorías de respuesta (probablemente peores respuestas)
#    aumentan los log-odds de recaída.

# Intercepto: -4.5963
# -> Interpretación:
#    Es el valor base de los log-odds cuando todos los predictores son cero.
#    No tiene un significado clínico directo por sí solo, pero forma parte del cálculo de la probabilidad.


# In[ ]:


odds_ratios = np.exp(coeficientes)

for feature, odds in zip(X.columns, odds_ratios):
    print(f"{feature}: {odds:.4f}")


# In[ ]:


# Interpretación de los Odds Ratios del modelo de regresión logística
# ----------------
# Age: 1.0063
# -> Interpretación:
#    Por cada año adicional de edad, las probabilidades de recaída aumentan en un 0.6%.
#    (Odds Ratio cercano a 1 indica un efecto pequeño).
# Stage
# -> Interpretación: 3.0704
#    Por cada aumento en el estadio, las probabilidades de recaída se multiplican por 3.07???? VER GISELA O en un 11% (COEF 1.12).
#    (Más del doble de riesgo de recaída a medida que el estadio es más avanzado).

#  Risk:  5.4668
# -> Interpretación:
#    Cada aumento en el nivel de 'Risk' aumenta las probabilidades de recaída en un 16% (proviene del 1.6 de los coeficiente)

# coeficiente es el que me habla de la probabilidad, y el odds es el que me habla del efecto (se ve entre las variables)

# Response:5.9326
# -> Interpretación:
#    Ciertas categorías de 'Response' aumentan las probabilidades de recaída casi 6 veces.
#    (Probable que las peores respuestas al tratamiento incrementen el riesgo de recaída).

# -------------------------------------------------------------------

# Nota general:
# -> Odds Ratios > 1 indican aumento en la probabilidad de recaída.
# -> Odds Ratios < 1 indican disminución en la probabilidad de recaída.


# In[ ]:


# MODELO DE REGRESION LOGISTICA
# Y= Recurred  todas las variables menos RISK


# In[ ]:


# Definir variables predictoras (X) y variable objetivo (y)
X = data[['Age', 'Stage', "Adenopathy", "Pathology", "Response", 'Gender','Smoking', "Thyroid Function", "T", "Hx Smoking",	"Hx Radiothreapy", "T", "N", "M"]]  # Variables clínicas
y = data['Recurred']  # Variable objetivo (0 = No recayó, 1 = Recayó)


# In[ ]:


X = data[["Age", "Stage_Coded", "Pathology_Coded","Response_Coded", "Gender_Coded","Smoking_Coded", "Thyroid Function_Coded", "T_Coded", "N_Coded", "M_Coded"]].copy()


# In[ ]:


for col in ['Age', 'Stage_Coded', "Pathology_Coded",'Response_Coded', 'Gender_Coded','Smoking_Coded', "Thyroid Function_Coded","T_Coded","N_Coded", "M_Coded" ]:
    X[col] = encoder.fit_transform(X[col])


# In[ ]:


# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión logística
modelo = LogisticRegression(max_iter=1000)  # Aumentamos iteraciones por si tarda en converger
modelo.fit(X_train, y_train)


# In[ ]:


# Hacer predicciones en los datos de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
print("Precisión:", accuracy_score(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))


# In[ ]:


# Precisión (Accuracy): 92% de los casos totales fueron correctamente clasificados.


# In[ ]:


# Reporte de Clasificación: .
# No (No recayó): Precisión (precision): 93% → De los que predijo "No", el 93% era correcto.
# Recall (sensibilidad): 97% → Detectó correctamente el 97% de los que realmente no recayeron.
# Yes (Recayó): Precisión: 88% → De los que predijo "Yes", el 88% era correcto.
# Recall: 79% → Detectó el 79% de los que realmente recayeron.

# Reporte de Clasificación CON RISK:
# No (No recayó): Precisión (precision): 95% → De los que predijo "No", el 95% era correcto.
# Recall (sensibilidad): 98% → Detectó correctamente el 98% de los que realmente no recayeron. #no detecta recaidos porque la data es inconsistente o no es lo esperable medicamente
#Yes (Recayó): Precisión: 94% → De los que predijo "Yes", el 94% era correcto.
# Recall: 84% → Detectó el 84% de los que realmente recayeron.


# In[ ]:


# Ver coeficientes de cada variable
coeficientes = modelo.coef_[0]
intercepto = modelo.intercept_[0]

# Asociar coeficientes con nombres de variables
for feature, coef in zip(X.columns, coeficientes):
    print(f"{feature}: {coef:.4f}")

print(f"\nIntercepto: {intercepto:.4f}")


# In[ ]:


# Coeficientes del modelo de regresión logística sin RISK
# ----------------------------------------------

# Age: 0.0055
# -> Interpretación:
#    Cada año adicional de edad aumenta ligeramente la probabilidad de recaída.
#    El efecto es pequeño debido al valor bajo del coeficiente.

# Stage: 1.6167
# -> Interpretación:
#    A medida que aumenta el estadio (por ejemplo, de I a IV), aumenta la probabilidad de recaída.
#    El coeficiente positivo indica que un estadio más avanzado incrementa los log-odds de recaída.

# Adenopathy: 0.9803
# -> Interpretación:
#    la presencia de ganglios palpables, aumenta ligeramente la probabilidad de recaída.

# Pathology: 0.3287
# -> Interpretación:
#    el subtipos histologico, se relaciona ligeramente con la probabilidad de recaída (micropapilar<papilar<folicular<celulas de hurtle).

# Response:  1.8279
# -> Interpretación:
#    Respuestas peores al tratamiento incrementan la probabilidad de recaída.
#    El coeficiente positivo indica que ciertas categorías de respuesta (probablemente peores respuestas)
#    aumentan los log-odds de recaída.


# In[ ]:


# Balance: Buen desempeño, aunque el modelo se le escapan algunos casos de recaída (falsos negativos). Al quitar la variable Risk disminuyo la especificidad y


# In[ ]:


odds_ratios = np.exp(coeficientes)

for feature, odds in zip(X.columns, odds_ratios):
    print(f"{feature}: {odds:.4f}")


# # Modelos - Preparar la data

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


# Importar librerías necesarias
import statsmodels.api as sm
import patsy


# In[ ]:


data = pd.read_csv("/content/drive/MyDrive/Copia de Thyroid_Diff.csv")


# In[ ]:


# Definir variables predictoras (X) y variable objetivo (y)
X = data[['Age', 'Stage', "Pathology", "Response", 'Gender','Smoking', "Thyroid Function", "T", "N", "M"]]  # Variables clínicas
y = data['Recurred']  # Variable objetivo (0 = No recayó, 1 = Recayó)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Definir la variable objetivo directamente desde el dataframe
y = data['Recurred']  # Esta es una serie, no un dataframe

# Crear el encoder y transformar 'Recurred'
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # 0 = No, 1 = Yes

# Verificar el resultado
print(y_encoded)


# In[ ]:


X = data[['Age', 'Stage', 'Risk', "Pathology", "Response", 'Gender','Smoking', "Thyroid Function", "Adenopathy", "T","N", "M"]].copy()


# In[ ]:


risk_map = {
    'Low': 0,
    'Intermediate': 1,
    'High': 2
}


# In[ ]:


data['Risk_Coded'] = data['Risk'].map(risk_map)


# In[ ]:


T_map = {
    'TIa': 0,
    'TIb': 1,
    "T2": 2,
    "T3a": 3,
    "T3b": 4,
    "T4a": 5,
    "T4b": 6,


}


# In[ ]:


data['T_Coded'] = data['T'].map(T_map)


# In[ ]:


pathology_map = {
    'Micropapillary': 0,
    'Papillary': 1,
    'Follicular': 2,
    'Hurtle cell': 3,

}


# In[ ]:


data['Pathology_Coded'] = data['Pathology'].map(pathology_map)


# In[ ]:


X = data[["Age", "Risk_Coded", "T_Coded", 'Pathology_Coded']].copy()


# In[ ]:


for col in ['Age', "Risk_Coded","T_Coded", 'Pathology_Coded' ]:
    X[col] = encoder.fit_transform(X[col])


# In[ ]:


# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # REGRESION LOGISTICA MULTINOMINAL

# In[ ]:


print("Valores nulos en el dataset:\n", data.isnull().sum())
data = data.dropna()


# In[ ]:


formula = "Recurred ~ Stage + Focality + Pathology + Gender + Q('Thyroid Function')"
y, X = patsy.dmatrices(formula, data, return_type="dataframe")


# In[ ]:


corr_matrix = X.corr().abs()
print("\nMatriz de correlación:\n", corr_matrix)


# In[ ]:


modelo = sm.MNLogit(y, X)
resultado = modelo.fit()


# In[ ]:


print(resultado.summary())


# In[ ]:


# Extraer coeficientes y p-values
print("\nCoeficientes del Modelo:\n", resultado.params)
print("\nP-values:\n", resultado.pvalues)


# # Decision Tree - Arboles de Decision

# In[ ]:


# Inicializar el modelo de Árbol de Decisión y entrenarlo
modelo = DecisionTreeClassifier(max_depth=4) # Fue error mio de codigo anterior las iterations son logistic regression
modelo.fit(X_train, y_train)  # ←


# In[ ]:


y_pred = modelo.predict(X_test)

# Evaluar el modelo
print("Precisión:", accuracy_score(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier #faltaba traer el modelo especificamente


# In[ ]:


# Inicializar el modelo de Árbol de Decisión y entrenarlo
modelo = RandomForestClassifier(n_estimators=2, max_depth=1000, random_state=42)
modelo.fit(X_train, y_train)  #


# In[ ]:


y_pred = modelo.predict(X_test)

# Evaluar el modelo
print("Precisión:", accuracy_score(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))


# In[ ]:


# Fin de modelos previos


# In[ ]:


# si queres crear tus propio grupo de recurrencia para testear modelos y comparar


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




