# ML2
Repo para proyectos de ML2. Somos el grupo 10.

# 1st project Folder: ENTREGA ENSEMBLE LEARNING

 + Estimar las utilizaciones horarias de un día a partir de las variables disponibles:

    + Utilizando técnicas de ensamblado, comparando con alguna más directa.

    + Haciendo una validación honesta para elegir el mejor enfoque entre los ensayados.

    + Incluir algún ejemplo de casos mal estimados.

    + Aportar resultados intermedios.

    + Realizar un análisis exploratorio previo (incluirlo en la entrega).

 + Entrega:

    + Informe (se entrega en formato pdf).

    + Cuaderno/s jupyter utilizados para realizar el informe, organizado de manera similar a la del informe presentado.

 + Solo entrega un miembro del grupo (en Moodle).


## Ayuda carga de datos

```

nom_fi_datos_Irrad = "nombre de mi fichero de irradiación"
df_orig_Irrad = pd.read_csv(nom_fi_datos_Irrad)

# parseamos la fecha (cadena) para que sea un datetime con formato yyyy/mm/dd

df_orig_Irrad['FECHA'] = pd.to_datetime(df_orig_Irrad['FECHA'], format='%Y-%m-%d')

s = df_orig_Irrad.loc[:,'FECHA']
df_orig_Irrad['FECHA'] =  s.dt.date

```

# 2nd project Folder: CANONIST.IA (ENTREGA CNN)

Ideas a realizar sobre el código base, ya con la conexión a Weights & Biases hecha:

 + Usar otros modelos preentrenados (Tampoco se tienen que probar todos, pero hacer research y explicar un poco cada uno, y por qué escogemos y probamos los que escojamos)

 + Decongelar más capas (aprende más de nuestros datos) (en cnn.py). Esto es fine tunning, entrenar el modelo con nuestros datos y que se especialice

 + Cambiar batch size y número de epochs

 + Crear nuestra propia CNN

 + RAG y Fine tunning (¿Lo estamos haciendo ya?). Diría que sí, fine tunning es descongelar capas, y RAG es hacer que el modelo responda con los datos dados por nosotros. Al darle datos de entrenamiento etiquetados diría que ya se hace (creo). Se podría hacer transfer learning (coger un modelo que haga otra cosa, como object detection, y cambiar la capa de object detection por una de classification)

 + Data augmentation (rotar imágenes, espejos, etc). Creo que ya se hace, investigar más (en cnn.py)

 + Más datos (creo que no vale)

 + No se pueden usar en color, verdad?

 + Cambiar optimizer y criterion

 + Cambiar learning rate y los pesos

 + ¿Grid search? Quizás tarda demasiado

 + Cargar un modelo preentrenado de HuggingFace más novedoso




