# Proyecto_14_Opiniones_Negativas / Negative opinions
Aprendizaje automático para texto / Machine learning for text

# Descripción del proyecto / Project description

SPA: Film Junky Union, una nueva comunidad vanguardista para los aficionados de las películas clásicas, está desarrollando un sistema para filtrar y categorizar reseñas de películas. Tu objetivo es entrenar un modelo para detectar las críticas negativas de forma automática. Para lograrlo, utilizarás un conjunto de datos de reseñas de películas de IMDB con etiquetado para construir un modelo que clasifique las reseñas como positivas y negativas. Este deberá alcanzar un valor F1 de al menos 0.85.

EN: Film Junky Union, a new cutting-edge community for fans of classic films, is developing a system for filtering and categorising film reviews. Your goal is to train a model to detect negative reviews automatically. To achieve this, you will use a dataset of tagged IMDB movie reviews to build a model that classifies reviews as positive and negative. This should reach an F1 value of at least 0.85.

## Instrucciones del proyecto

   1. Carga los datos.
   2. Preprocesa los datos, si es necesario.
   3. Realiza un análisis exploratorio de datos y haz tu conclusión sobre el desequilibrio de clases.
   4. Realiza el preprocesamiento de datos para el modelado.
   5. Entrena al menos tres modelos diferentes para el conjunto de datos de entrenamiento.
   6. Prueba los modelos para el conjunto de datos de prueba.
   7. Escribe algunas reseñas y clasifícalas con todos los modelos.
   8. Busca las diferencias entre los resultados de las pruebas de los modelos en los dos puntos anteriores. Intenta explicarlas.
   9. Muestra tus hallazgos.

¡Importante! Para tu comodidad, la plantilla del proyecto ya contiene algunos fragmentos de código, así que puedes usarlos si lo deseas. Si deseas hacer borrón y cuenta nueva, simplemente elimina todos esos fragmentos de código. Aquí está la lista de fragmentos de código:

   - un poco de análisis exploratorio de datos con algunos gráficos;
   - 'evaluate_model()': una rutina para evaluar un modelo de clasificación que se ajusta a la interfaz de predicción de scikit-learn;
   - 'BERT_text_to_embeddings()': una ruta para convertir lista de textos en insertados con BERT.

Tu trabajo principal es construir y evaluar modelos.

Como puedes ver en la plantilla del proyecto, te sugerimos probar modelos de clasificación basados en regresión logística y potenciación del gradiente, pero puedes probar otros métodos. Puedes jugar con la estructura de la plantilla del proyecto siempre y cuando sigas sus instrucciones.

No tienes que usar BERT para el proyecto porque requiere mucha potencia computacional y será muy lento en la CPU para el conjunto de datos completo. Debido a esto, BERT generalmente debe ejecutarse en GPU para tener un rendimiento adecuado. Sin embargo, puedes intentar incluir BERT en el proyecto para una parte del conjunto de datos. Si deseas hacer esto, te sugerimos hacerlo de manera local y solo tomar un par de cientos de objetos por cada parte del conjunto de datos (entrenamiento/prueba) para evitar esperar demasiado tiempo. Asegúrate de indicar que estás usando BERT en la primera celda (el encabezado de tu proyecto).

EN: 
   1. Load the data.
   2. Pre-process the data, if necessary.
   3. Conduct an exploratory data analysis and make your conclusion on class imbalance.
   4. Perform data pre-processing for modelling.
   5. Train at least three different models for the training data set.
   6. Test the models for the test dataset.
   7. Write some reviews and rank them with all models.
   8. Look for the differences between the results of the model tests in the two points above. Try to explain them.
   9. Show your findings.

Important! For your convenience, the project template already contains some code snippets, so you can use them if you wish. If you want to wipe the slate clean, simply delete all those code snippets. Here is the list of code snippets:

   - a bit of exploratory data analysis with some graphs;
   - 'evaluate_model()': a routine to evaluate a classification model that fits the scikit-learn prediction interface;
   - 'BERT_text_to_embeddings()': a path to convert text lists to BERT inserts.

Your main job is to build and evaluate models.

As you can see in the project template, we suggest you try classification models based on logistic regression and gradient boosting, but you can try other methods. You can play with the structure of the project template as long as you follow its instructions.

You don't have to use BERT for the project because it requires a lot of computational power and will be very slow on the CPU for the full data set. Because of this, BERT should generally run on GPUs for adequate performance. However, you can try to include BERT in the project for a part of the dataset. If you want to do this, we suggest you do it locally and only take a couple of hundred objects for each part of the dataset (training/testing) to avoid waiting too long. Make sure you indicate that you are using BERT in the first cell (the header of your project).

# Descripción de los datos / Data description

###SPA:
Los datos se almacenan en el archivo imdb_reviews.tsv.

Los datos fueron proporcionados por Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, y Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. La Reunión Anual 49 de la Asociación de Lingüística Computacional (ACL 2011).

Aquí se describen los campos seleccionados:

   - 'review': el texto de la reseña
   - 'pos': el objetivo, '0' para negativo y '1' para positivo
   - 'ds_part': 'entrenamiento'/'prueba' para la parte de entrenamiento/prueba del conjunto de datos, respectivamente

Hay otros campos en el conjunto de datos, puedes explorarlos si lo deseas.

EN: 
The data is stored in the file imdb_reviews.tsv.

Data were provided by Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts (2011). Ng, and Christopher Potts (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

The selected fields are described here:
   - 'review': the text of the review
   - 'pos': the target, ‘0’ for negative and ‘1’ for positive
   - 'ds_part': 'train'/'test' for the training/testing part of the dataset, respectively

There are other fields in the dataset, you can explore them if you wish.
   
