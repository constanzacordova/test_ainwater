import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import funciones_test as ft

###### IMPORTACION DATAFRAME ######
df = pd.read_csv('base_test_planta_tupiniquim.csv', parse_dates={'date_time': ['date', 'time']}).drop('Unnamed: 0', axis =1)
df['date'] = df['date_time'].dt.date
df['time'] = df['date_time'].dt.time

###### DATAFRAME REFACTORIZADO ######
df_resume_cycle = ft.transformar_dataframe(df)

###### IMPORTACION FIGURAS ######
image_hist = open('graficos/histograma_general.png', 'rb')
image_boxplt = open('graficos/boxplot_general.png', 'rb')
image_dist_do = open('graficos/do_level_general.png', 'rb')
image_dist_do_bloweron = open('graficos/do_level_bloweon.png', 'rb')
image_cluster = open('graficos/cantidad de cluster.png', 'rb')


st.beta_set_page_config( page_title="Análisis Ainwater ", page_icon='')
############# MAIN #############
def main():

    #### INSPECCION DE LOS DATOS  ####
    st.title("Inspección de los datos")
    
    st.subheader("Importacion DataFrame")
    st.dataframe(df)
    st.write("Se inspecciona el tipo de datos del Dataframe y se observa que no hay valores nulos, por lo tanto se considera que el archivo cuenta con todas las observaciones.")
    
    st.subheader("Tiempos de lectura entre cada observación")
    df_delta = df.iloc[:, :-1] - df.iloc[:, :-1].shift(1)
    st.write(df_delta['date_time'].describe())
    st.write("Se observa que las mediciones fueron tomadas cada 3 minutos aproximadamente de forma constante")
    st.write(" ")
    st.write(" ")

    st.subheader("Inspección de las variables de operación")
    st.write(df.iloc[:, :-3].describe())
    st.write(" ")
    st.write("Histogramas de las variables de operación")
    st.image(image_hist.read(), width= 700, caption= 'Histogramas de variables de operación')
    st.write(" ")
    st.write("Boxplot de las variables de operación")
    st.image(image_boxplt.read(), width= 700, caption= 'Boxplot de variables de operación')
    st.write("De acuerdo a los gráficos boxplot se observa que los ciclos tienen una distribución distinta entre ellos en el nivel de oxigeno y agua. Sin embargo, se observa que existen ciclos que tienen similitud en su distribución y la variación de los niveles de oxigeo y agua son ciclicas.")
    st.write(" ")
    st.write(" ")

    st.subheader("Generación de arreglo para analisis")

    # Refactorizacion del dataframe
    st.markdown("A continuación se presenta script para relalizar arreglo del data set y observar de forma resumida las variables.")
    st.markdown("Los pasos de transformación fueron los siguientes:")
    st.markdown("- 1. Total de Hz del soplador por ciclo: suma de las observaciones para la variable Hz agrupada por ciclo")
    st.markdown("- 2. Niveles de DO y H2O promedio por ciclo: Media aritmetica de las observaciones de DO y H2O por ciclo")
    st.markdown("- 3. Niveles de DO cuando está el soplador encendido: Media aritmetica de las observaciones de DO cuando blower_hz > 0")
    st.markdown("- 4. Datos sobre temporalidad de los ciclos: Se determina la duración de cada ciclo")
    st.markdown("- 5. Momento del día del ciclo: Se realiza un arreglo de acuerdo a la hora de inicio y termino del ciclo. Se establecen 5 momentos del dia de ciclos:")
    st.markdown("    -- Turno Mañana que comprende desde las 6 hrs hasta las 12 hrs ")
    st.markdown("    -- Turno Media-Mañana que comprende un ciclo iniciado durante el Turno Mañana pero que finalizó pasado las 12 hrs")
    st.markdown("    -- Turno Tarde que comprende desde las 12 hrs hasta las 19 hrs")
    st.markdown("    -- Turno Noche que comprende desde las 19 hrs hasta las 24 hrs")
    st.markdown("    -- Turno Madrugada que comprende desde las 24 hrs hasta las 6 hrs") 
    st.markdown("Adicionalmente se omite el ciclo 0 dado que las mediciones registradas corresponden a 1 hora 41 minutos del ciclo versus el tiempo de los ciclos siguientes. Dada esta información se puede deducir que los registros del ciclo 0 se encuentran incompletos, además se observa que el ciclo 0 no tiene operación del soplador por lo que no nos entregará información relevante para el análisis")
 
    st.write("Nuevo Dataframe")
    st.dataframe(df_resume_cycle)

    
    
    #### CARACTERIZACION DE LA PLANTA ####
    colors = ft.colors(df_resume_cycle)
    fig_blower_cycle, fig_h2o_cycle, fig_do_cycle = ft.resume_cycle(df_resume_cycle, colors)
    st.title("Caracterización de la planta")
    st.subheader("Histórico variables de operación")
    st.write(ft.historico_variables(df))
    st.write('Luego de graficar los datos históricos de las tres variables se observa a alto nivel que un día de operación de la planta está compuesto por 5 ciclos, los primeros dos ciclos comienzan con un mayor nivel de oxigeno, menor nivel de agua y la duración en que el soplador funciona a 50 hz es menor que en los tres ciclos restantes. Se observa además que a partir del tercer ciclo del día hay un aumento del nivel de agua y el nivel de oxigeno cae en relación a los ciclos anteriores, en estos casos se observa que la duración de operación del soplador a 50 hz es mayor')
    st.write(" ")

    st.subheader('Resumen de variables por ciclo')
    st.write(fig_blower_cycle)
    st.write(fig_h2o_cycle)
    st.write(fig_do_cycle)
    st.write("De acuerdo al gráfico de Hz totales por ciclo se observa que existen ciclos bien definidos durante el día y responden a los turnos del día. Se obverva que los ciclos 3, 8, 13, 18, 23, 28 y 33 son los ciclos en donde más se hace uso del soplador durante cada día, estos ciclos coindicen en que son los que comenzaron en el turno identificado como media mañana, lo cual puede tener relación con el aumento del nivel de agua para ese turno, se puede suponer que durante ese horario existe mayor uso de agua por parte de la población y por ende existe mayor volumen de agua a tratar y el nivel de oxigeno también es menor. Tambien se observa que los ciclos que corresponde a turno de madrugada y tarde son los que tiene mayor nivel de oxigeno, menor nivel de agua y se hace menor uso de los sopladores")
    st.write(" ")

    st.subheader("Comparación de gasto energético entre ciclos")
    df_heatmap, df_heatmap_matrix = ft.matriz_turnos(df_resume_cycle)
    st.write(ft.heatmap(df_heatmap_matrix))
    st.write(ft.scatter_hz(df_heatmap))
    st.write("A partir del mapa de calor y considerando que la relación entre Hz es directamente proporcional al gasto energético se observan diferencias en el gasto energético en dos niveles. Se podría decir que hay una banda de bajo consumo que corresponde a los ciclos de la madrugada y mañana, y una banda de alto consumo que corresponde a los ciclos de media mañana, tarde y noche. Se oserva, al igual que en el análisis anterior, que el horario de mayor consumo energético es la de media mañana, siendo el día 18 de abril el que tuvo mayor consumo en ese horario")
    st.write(" ")

    st.subheader("Distribución del nivel de oxigeno")
    st.write(ft.oxigeno_medio(df_resume_cycle))
    st.markdown("#### Distribución del nivel de oxigeno por ciclo de operación")
    st.image(image_dist_do.read(), width= 700)
    st.markdown("#### Distribución del nivel de oxigeno por ciclo de operación cuando el soplador está en funcionamiento")
    st.image(image_dist_do_bloweron.read(), width= 700)


    #### CLUSTERIZACION DE CICLOS
    st.title("Clusterización de los ciclos")
    st.write("Para el estudio de clusterización de los ciclos se utilizará el dataframe preprocesado (df_resume_cycle) que resume las variables de cada ciclo. En este caso se utilizará para la construcción de la matriz la siguientes variables:")
    st.markdown("- total_hz")
    st.markdown("- do_mean_blwon")
    st.markdown("- do_mean")
    st.markdown("- h2o_mean")
    st.write("Dado que las unidades de medida de las variables son distintas se estándarizarán  los valores aplicando StandarScaler")

    X_matrix = ft.matriz_kmeans(df_resume_cycle)

    st.markdown("Estudio de número de cluster - Elbow graph")
    st.plotly_chart(ft.elbow_graph(X_matrix))
    st.write("De acuerdo al gráfico se observa que los ciclos se pueden categorizar en dos cluster por lo que realizaremos el modelo con un nùmero de cluster igual a 2")

    st.subheader('Clusterizacion con KMean nro cluster = 2')
    kmeans_model = KMeans(n_clusters=2).fit(X_matrix)
    df_cluster, color, C = ft.label_cluster(kmeans_model, df_resume_cycle)
    
    st.markdown("##### Resumen de dataframe con clusterización")
    st.dataframe(df_cluster)
    st.markdown("##### Total Hz vs Media Oxigeno con soplador en funcionamiento")
    st.plotly_chart(ft.grafico_cluster(X_matrix, C, color, 0, 1))
    st.markdown("##### Total Hz vs Media Oxigeno General")
    st.plotly_chart(ft.grafico_cluster(X_matrix, C, color, 0, 2))
    st.markdown("##### Total Hz vs Nivel medio de agua")
    st.plotly_chart(ft.grafico_cluster(X_matrix, C, color, 0, 3))
    st.markdown("##### Nivel medio de agua vs Nivel medio de oxigeno gral")
    st.plotly_chart(ft.grafico_cluster(X_matrix, C, color, 3, 2))
    st.write("De acuerdo a la clusterización mediante KMeans con nro cluster = 2, hay 21 ciclos que quedaron agrupados en el cluster 1 y 14 ciclos que quedaron agrupados en el cluster 2. Los ciclos que corresponden al cluster 1 fueron aquellos que se etiquetaron como horario Media-mañana, tarde y noche y que coinciden con ser los momentos del día con mayor consumo energético. En cambio los ciclos que corresponden al claster 2 son aquellos que se podrían clasificar como bajo consumo energético")

    
    
    


    
    
if __name__ == "__main__":
    main()