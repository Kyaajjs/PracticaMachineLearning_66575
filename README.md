# Práctica Machine Learning
Ulises Zarate Concha | Matrícula 66575 | Universidad Autónoma de Campeche

## Programas públicos
- Clasificación: https://practica-ml-ulises-66575.equal-knoll-6193.chatgpt.site/?ejemplo=clasificacion
- Regresión: https://practica-ml-ulises-66575.equal-knoll-6193.chatgpt.site/?ejemplo=regresion
- Series: https://practica-ml-ulises-66575.equal-knoll-6193.chatgpt.site/?ejemplo=series

En navegador: carga una muestra, pulsa Ejecutar predicción o Evaluar todo el test.
Se ejecuta Python real con NumPy mediante Pyodide. Se necesita internet para la
primera descarga del intérprete desde cdn.jsdelivr.net. El entrenamiento se realiza
con los scripts locales o notebooks; el sitio ejecuta los modelos ya entrenados.
Los valores mostrados en tarjetas son resultados registrados; el botón Evaluar
recalcula las métricas. No se usa una API de predicciones ni se envían tus entradas.

## Código descomprimido en GitHub
Los scripts, notebooks, datasets y pesos están también como archivos individuales
en la raíz del repositorio. Tras clonar o descargar el repositorio:
```
python -m pip install -r requirements.txt
python entrenar.py todos
python verificar.py
```
Los siguientes pasos corresponden a la estructura del ZIP entregable.

## Instalación desde el ZIP (Python 3.12 recomendado)
Desde esta carpeta descomprimida:
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python programas/entrenar.py todos
```
También puedes ejecutar un solo caso:
```
python programas/entrenar.py clasificacion
python programas/entrenar.py regresion
python programas/entrenar.py series
```
La copia de los datos está incluida; tras instalar dependencias no se necesita red.
No es necesario abrir los scripts como archivo de texto: se ejecutan desde terminal.
Los resultados se escriben en resultados/ y la salida se imprime en pantalla.
Para inferencia sin reentrenar: `python programas/inferencia.py series`.
Para comprobar equivalencia y validación: `python verificar.py`.

## Google Colab
Abre https://colab.research.google.com/ → Archivo → Subir cuaderno.
Elige programas/clasificacion.ipynb, regresion.ipynb o series.ipynb.
Después Entorno de ejecución → Ejecutar todas. Cada notebook descarga este ZIP
del enlace público e instala las dependencias. No están alojados como notebooks
editables en una cuenta de Google; son archivos importables y reproducibles.

## Datos y alcance
Wine: UCI, 178 muestras, 13 entradas, 3 clases; etiquetas transformadas a 0,1,2.
https://archive.ics.uci.edu/dataset/109/wine
Energy Efficiency: UCI, 768 simulaciones de edificios, 8 entradas; se predice Y1.
No se usa Y2 como predictor. https://archive.ics.uci.edu/dataset/242/energy+efficiency
Wafer: UCR/UEA, 1000 entrenamiento y 6164 prueba; 152 pasos; -1 defectuosa → 1,
+1 normal → 0. https://www.timeseriesclassification.com/description.php?Dataset=Wafer
Fuente directa original: https://www.timeseriesclassification.com/aeon-toolkit/Wafer.zip
Las copias CSV conservan la información relevante. Wafer se redondea a 8 cifras
significativas; Wine/Energy a 10. Se calculan todos los resultados con estas copias.
No se inventan timestamps ni se presenta Wafer como pronóstico de valores futuros.

## Método y límites
Semilla 42. Ajuste por validación; ningún candidato se elige por la prueba final.
Wine y Energy: 60/20/20 aproximadamente, con estratificación en Wine.
Wafer: train original dividido 800/200, prueba original intacta (6164).
Después de elegir arquitectura se entrena con train+validación y se mide el test.
Los escaladores solo se ajustan sobre datos de entrenamiento de cada etapa.
ESN: reservorio recurrente fijo de 24 unidades, radio espectral 0.8, fuga 0.5;
se concatenan estados final, medio y máximo; capa de lectura MLP entrenable.
Es aprendizaje supervisado del readout, sin backpropagation through time.
El desbalance de Wafer exige observar recall y F1 de defectos además de accuracy.
Energy proviene de simulación y la partición aleatoria puede compartir geometrías
similares; no demuestra generalización a edificios reales o geometrías no vistas.
Los resultados son académicos. Ver métricas JSON y salidas TXT para detalles.

## Estructura
- programas/: entrenamiento completo, inferencia, lanzadores y notebooks.
- datos/: datasets empleados.
- resultados/: pesos JSON, métricas, salidas, pruebas reservadas y gráficas.
- evidencias/: registros visuales de salidas y código.
- ORIGEN_IA_Y_PROMPTS.txt: atribución y prompt usado.
- entorno_verificado.json: versiones reales del entorno de ejecución.
- FUENTES_Y_DATOS.md: procedencia y hashes.

## Evidencia y revisión
Las figuras se generaron ejecutando los programas. Los registros visuales de salida
rasterizan fielmente los TXT de ejecución; no son fotografías de una pantalla ni
capturas de un IDE. Se verificó la inferencia en Python y en Pyodide WebAssembly con NumPy 2.0.2;
las métricas coincidieron. No se realizó una prueba visual interactiva completa
en navegador; se verificaron además los datos, el JavaScript y el despliegue.
El profesor puede reproducir cada salida con los comandos anteriores.

## Origen
Código generado con ChatGPT de OpenAI. Ver ORIGEN_IA_Y_PROMPTS.txt.
Asignatura y docente no proporcionados; completar esos datos en el reporte editable.
