# Gestor-SR
## Desafío técnico Gestor SR con énfasis en IA.

Este protecto desarrolla un modelo de prediccion de series temporales y adicionalmente un modelo de clasificación, esto con el fin de dar solución en primera instancia al calculo de la demanda para tres periodos datos y adicionalmente encontrar un modelo de clasiificación alpha beta, para contestar la siguiente problematica:

Summa-sci requiere construir un servicio (Backend) para una aplicación de experiencia del colaborador, esta aplicación lo que hace es recibir en la interfaz de usuario (Frontend) unos parámetros de una lista desplegable ingresados por el usuario y con esto un valor en números que corresponde al pronóstico de la demanda de las compras de la compañía Cementos Argos, asociado a esos parámetros.

Para el modelo de clasificación se expuso una api utilizando FastApi ejecutable desde un archivo main.py, este proyecto deberá consume el modelo y recibe un json de una solicitud post, realiza la clasificación y entregar la respuesta. Se realizó un despliege en Render cuya dirección es:

[Visita mi despliege de FastAPI en Render](https://clasificacion-alpha-betha.onrender.com/docs#/default/predict_predict_post)

 
### Forecasting

Este código implementa un sistema robusto de pronóstico de demanda usando el modelo SARIMA con transformaciones y validaciones avanzadas:

#### Estructura Principal
* Configuración inicial: Importa bibliotecas (pandas, statsmodels, pmdarima) y define estilos de gráficos.
* Función robust_demand_forecast:
* Carga datos temporales (serie mensual de demanda)
* Limpia outliers usando medianas móviles e IQR
* Aplica transformación Box-Cox para normalizar datos

#### Modelado:

* Usa auto_arima para encontrar automáticamente los mejores parámetros SARIMA
* Ajusta el modelo final con los parámetros óptimos
* Realiza diagnóstico de residuos (autocorrelación, QQ plot, histograma)

#### Validación:

* Walk-forward validation para evaluar rendimiento en múltiples ventanas
* Calcula métricas de error (RMSE, MAPE)

#### Pronóstico:

* Genera predicciones con intervalos de confianza
* Aplica ajuste especial para valores de enero (decaimiento del 30%)
* Visualiza resultados con gráficos comparativos

#### Salida:

* Devuelve métricas, pronósticos futuros e información del modelo

#### Puntos Clave

* Manejo robusto de datos: limpia outliers y transforma la serie
* Automatización de selección de parámetros SARIMA
* Validación rigurosa con múltiples técnicas
* Ajustes estacionales específicos (para enero)
* Visualización profesional de resultados y diagnósticos

### Clasificación

Este código implementa un modelo de clasificación usando XGBoost con preparación de datos y evaluación:

#### Flujo Principal
Carga y preparación de datos:

* Lee el dataset y elimina la columna autoID
* Identifica columnas categóricas y las convierte a tipo string
* Aplica one-hot encoding a las variables categóricas
* Codifica la variable objetivo (Class) con LabelEncoder

#### Preprocesamiento:

* Estandariza los datos con StandardScaler
* Divide en conjuntos de entrenamiento (80%) y prueba (20%) manteniendo proporción de clases

#### Modelado en dos fases:

* Primer modelo base: XGBoost para identificar características importantes
* Selección de características: Usa el modelo base para seleccionar las más relevantes (umbral mediano)
* Modelo final: XGBoost entrenado solo con características seleccionadas

#### Persistencia:

* Guarda el modelo, scaler, selector y encoder en archivos .pkl para uso futuro

#### Evaluación:

* Calcula accuracy, reporte de clasificación y matriz de confusión
* Guarda métricas en archivo y muestra visualizaciones
* Preprocesamiento robusto: Encoding de variables categóricas y estandarización
* Selección de características: Usa importancia de XGBoost para reducir dimensionalidad
* Validación: 20% de datos reservados para prueba con estratificación
* Persistencia: Guarda todos los componentes necesarios para reutilización

#### Visualizaciones:

* Matriz de confusión (heatmap)
* Gráfico de importancia de características (15 más relevantes)
* Modelo entrenado (modelo_clasificacion_xgb.pkl)
* Transformadores guardados (scaler, selector, encoder)
* Archivo con métricas (metricas_modelo.txt)
* Visualizaciones interactivas de evaluación

El código sigue prácticas de ML: separación train-test, estandarización, selección de features y evaluación rigurosa. Terminado el modelo se creó el archivo main.py para realizar la api con FastAPI.

