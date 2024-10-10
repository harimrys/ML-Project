# ML Project

## Descripción
Somos un banco ficticio llamado **HC Bank** y queremos predecir con machine learning el **Spending Score** de nuestros clientes para poder proporcionar a cada uno el producto financiero idóneo. En este repositorio, vamos a limpiar un dataset inicial que se incluye y a probar distintos modelos de clasificación para elegir el que mejor se adapte a nuestro propósito. Este trabajo es parte del bootcamp de **Ironhack** de análisis de datos, donde tratamos de aprender a utilizar machine learning.

## Contexto
El objetivo principal de este proyecto es predecir el **Spending Score** de nuestros clientes utilizando diversas variables como la edad, el estado civil, la profesión y si están graduados, entre otros. Con esta información, podremos ofrecer el producto financiero más adecuado a cada cliente.

## Estructura del Proyecto
- **notebooks/**: 
  - `test_main.ipynb`: Pruebas iniciales y exploración de datos.
  - `main.ipynb`: Limpieza de datos y resultados finales con conclusiones.
  - `ML_Models.ipynb`: Evaluación de distintos modelos de clasificación.
  - `functions.py`: Archivo que contiene funciones auxiliares utilizadas en los notebooks.
- **data/**: 
  - Carpeta que contiene todos los DataFrames, tanto limpios como el inicial, y los generados a partir de los resultados.
- **requirements.txt**: Lista de dependencias necesarias para el proyecto.
- **config.yaml**: Archivo de configuración.

## Modelos Utilizados y Resultados
Se han evaluado los siguientes modelos de clasificación:
- **K-Nearest Neighbors (KNN)**: Accuracy de 0.75.
- **Bagging**: Accuracy de 0.77.
- **Pasting**: Accuracy de 0.78.
- **Random Forest**: Accuracy de 0.82.
- **Gradient Boosting**: Accuracy de 0.82.
- **Adaptive Boosting (AdaBoost)**: Accuracy de 0.82.

## Conclusiones
Después de evaluar todos los modelos, **Gradient Boosting** y **Random Forest** mostraron las mejores métricas de precisión, con un **accuracy** del 82%. Aunque ambos modelos son efectivos, se recomienda elegir **Gradient Boosting** por su capacidad para manejar datos complejos y su rendimiento consistente en la validación cruzada.

## Links de interés

[Presentación del proyecto](https://www.canva.com/design/DAGTLdbbIYM/SL_qq0Znn5OHaQz9oHNAUg/view?utm_content=DAGTLdbbIYM&utm_campaign=designshare&utm_medium=link&utm_source=editor)

## Autoras
- Haridian Morays
- Cristina Ramírez
