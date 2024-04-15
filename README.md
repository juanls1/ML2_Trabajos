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

## CONTEXTO DEL TRABAJO

Explicado en profundidad en el README dedicado a dicho proyecto (interior de la carpeta). 

El trabajo se basa en la búsqueda del mejor modelo posible de clasificación multiclase de imágenes de idealista basándose en las principales CNN preentrenadas, junto a su presentación profesional end-to-end (repositorio, entorno, aplicación y presentación).


# 3rd project Folder: ENTREGA APRENDIZAJE POR REFUERZO

## CONTEXTO DEL TRABAJO

Será obligatorio realizar un trabajo final que aplique las técnicas de aprendizaje por refuerzo vistas en esta parte de la asignatura. Se usará cualquier entorno disponible en Gym (https://gym.openai.com/)  y se le aplicarán las técnicas de aprendizaje por refuerzo que se estime oportuno. Se usará un caso distinto a los suministrados como apoyo al material de clase.

El trabajo podrá hacerse de manera individual o en grupos. En caso de grupos cada componente deberá realizar un algoritmo  distinto y se ha de indicar quien lo ha desarrollado. En este caso, la comparativa de resultados se supone es labor conjunta. Es de esperar que un trabajo hecho por más componentes conlleve una mayor complejidad, volumen y alcance de las tareas realizadas.

## CRITERIOS DE EVALUACIÓN

La evaluación se realizará en base a los siguientes criterios:

 + Definición del contexto del trabajo, objetivos y métodos de RL a aplicar, así como su idoneidad. (1 Pto.)

 + Cantidad, profundidad y calidad del trabajo desarrollado. (2 Ptos)

 + Número de técnicas empleadas en el proyecto, así como su idoneidad.(2.5 Ptos.)

 + Interpretación de resultados y pruebas realizadas. (1.5 puntos)

 + Comparativa de métodos. (1.5 puntos)

 + Claridad de escritura y presentación del trabajo en la memoria del mismo. (1.5 Ptos.)

Este trabajo formará parte de la media de la nota obtenida por trabajos (labs) en la asignatura. Junto a la memoria del trabajo se adjuntará un video de como máximo 1 minuto de duración con los resultados de los algoritmos ensayados.
