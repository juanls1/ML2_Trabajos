# ML2
Repo para proyectos de ML2. Somos el grupo 10.

# 1st project Folder: ASSESSMENT ENSEMBLE LEARNING

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


# Ayuda carga de datos

```

nom_fi_datos_Irrad = "nombre de mi fichero de irradiación"
df_orig_Irrad = pd.read_csv(nom_fi_datos_Irrad)

parseamos la fecha (cadena) para que sea un datetime con formato yyyy/mm/dd

df_orig_Irrad['FECHA'] = pd.to_datetime(df_orig_Irrad['FECHA'], format='%Y-%m-%d')

s = df_orig_Irrad.loc[:,'FECHA']
df_orig_Irrad['FECHA'] =  s.dt.date

```




