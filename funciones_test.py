import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

### Función que entrega histogramas para las variables estadisticas del data set, es decir DO level, H2O Level y Blower Hz
def histograma_data(dataframe, variable):
    '''
    Histograma_data: entrega gráfico histograma para la variable en estudio
    Parámetros:
    dataframe: dataframe que contiene la muestra de observaciones 
    variable: variable en estudio tipo string
    '''
    mean = round(np.mean(dataframe[variable]),2)
    sns.distplot(dataframe[variable], bins=10, kde=False).set_title(f'Histograma de {variable}')
    plt.axvline(mean, color= 'r', ls='dotted', label = f'media: {mean}')
    plt.legend()


### Función que entrega boxplot de variable agrupadas por cycle_id por defecto
def boxplot_group(dataframe, variable, group_by = 'cycle_id'):
    '''
    boxplot_group: Entrega gráficos boxplot de una variable agrupadas por categoria 
    
    Parámetros
    dataframe: dataframe que contiene la muestra de observaciones
    variable: variable en estudio tipo string
    groupby: variable que regresenta la categoría por la cual se quiere agrupar 
    '''
    sns.set(font_scale=1.3)
    sns.boxplot(x= dataframe[group_by], y= dataframe[variable], width = 0.3) 

### Función que entrega un grafico histórico de las variables observadas oxigeno, nivel de agua y potencia de soplador
def historico_variables(df):
    '''
    historico_variables: entrega un gráfico con la disperción de las mediciones de las variables DO level, H2O level,
    Blower_hz en función a la fecha y hora de medición 
    '''
    # crear figura
    fig = go.Figure()

    #agregar variables a trazar
    fig.add_trace(go.Scatter(x=df['date_time'], y= df['do_level'], name= 'DO Level', yaxis= 'y1'))
    fig.add_trace(go.Scatter(x=df['date_time'], y= df['h2o_level'], name= 'H2O Level', yaxis= 'y2'))
    fig.add_trace(go.Scatter(x=df['date_time'], y= df['blower_hz'], name= 'Blower Hz', yaxis= 'y3'))

    fig.update_layout(title_text = 'Históricos del proceso')


    # Ejes del gráfico
    fig.update_layout(
        xaxis=dict(
            autorange=False,
            range=["2021-04-14 01:44:36", "2021-04-14 06:35:38"],
            rangeslider=dict(autorange=False, range=[df['date_time'].min(), df['date_time'].max()]), type="date"),
        
        yaxis=dict(
            domain=[0, 0.3],
            range=[df['do_level'].min(), df['do_level'].max()],
            title="DO level",
            showline=True,
            tickmode="auto",
            titlefont = {"color": "#1052b5"},
            tickfont={"color": "#1052b5"},
            linecolor="#1052b5",
            zeroline=True
            ),
        
        yaxis2=dict(
            domain=[0.3, 0.6],
            range=[df['h2o_level'].min(), df['h2o_level'].max()],
            title="H2O level",
            showline=True,
            tickmode="auto",
            titlefont = {"color": "#b53110"},
            tickfont={"color": "#b53110"},
            linecolor="#b53110",
            zeroline=True
            ),
        
        yaxis3=dict(
            domain=[0.6, 1.0],
            range=[df['blower_hz'].min(), df['blower_hz'].max()],
            title="Hz blower",
            showline=True,
            tickmode="auto",
            titlefont={"color": "#56b030"},
            tickfont={"color": "#56b030"},
            linecolor="#56b030",
            type="linear",
            zeroline=True
        )
    )

    # Incorporar etiqueta de los ciclos 
    fig.update_layout(
        annotations=[
            dict(x= "2021-04-14 00:00:28" , text="ciclo 0"),
            dict(x= "2021-04-14 01:44:36" , text="ciclo 1"),
            dict(x= "2021-04-14 06:35:38" , text="ciclo 2"),
            dict(x= "2021-04-14 11:23:40" , text="ciclo 3"),
            dict(x= "2021-04-14 16:14:42" , text="ciclo 4"),
            dict(x= "2021-04-14 21:02:44" , text="ciclo 5"),
            dict(x= "2021-04-15 01:50:46" , text="ciclo 6"),
            dict(x= "2021-04-15 06:38:48" , text="ciclo 7"),
            dict(x= "2021-04-15 11:29:50" , text="ciclo 8"),
            dict(x= "2021-04-15 16:17:52" , text="ciclo 9"),
            dict(x= "2021-04-15 21:05:54" , text="ciclo 10"),
            dict(x= "2021-04-16 01:53:56" , text="ciclo 11"),
            dict(x= "2021-04-16 06:44:58" , text="ciclo 12"),
            dict(x= "2021-04-16 11:33:00" , text="ciclo 13"),
            dict(x= "2021-04-16 16:24:02" , text="ciclo 14"),
            dict(x= "2021-04-16 21:12:04" , text="ciclo 15"),
            dict(x= "2021-04-17 02:00:06" , text="ciclo 16"),
            dict(x= "2021-04-17 06:48:08" , text="ciclo 17"),
            dict(x= "2021-04-17 11:39:11" , text="ciclo 18"),
            dict(x= "2021-04-17 16:27:13" , text="ciclo 19"),
            dict(x= "2021-04-17 21:15:15" , text="ciclo 20"),
            dict(x= "2021-04-18 02:06:17" , text="ciclo 21"),
            dict(x= "2021-04-18 06:54:20" , text="ciclo 22"),
            dict(x= "2021-04-18 11:42:22" , text="ciclo 23"),
            dict(x= "2021-04-18 16:33:24" , text="ciclo 24"),
            dict(x= "2021-04-18 21:21:26" , text="ciclo 25"),
            dict(x= "2021-04-19 02:09:28" , text="ciclo 26"),
            dict(x= "2021-04-19 07:00:30" , text="ciclo 27"),
            dict(x= "2021-04-19 11:48:34" , text="ciclo 28"),
            dict(x= "2021-04-19 16:39:39" , text="ciclo 29"),
            dict(x= "2021-04-19 21:27:41" , text="ciclo 30"),
            dict(x= "2021-04-20 02:15:43" , text="ciclo 31"),
            dict(x= "2021-04-20 07:03:45" , text="ciclo 32"),
            dict(x= "2021-04-20 11:54:47" , text="ciclo 33"),
            dict(x= "2021-04-20 16:42:49" , text="ciclo 34"),
            dict(x= "2021-04-20 21:30:51" , text="ciclo 35"),
        ])

    # Update layout
    fig.update_layout(
        dragmode="zoom",
    # legend=dict(traceorder="reversed"),
        height=600,
        template="plotly_white",
        margin=dict(
            t=100,
            b=100
        ),
    )

    fig.show()

# Función para obtener el histograma con la distribución de una variable categorizada por ciclo
def distribucion_clase(dataframe, variable, categoria, clase, titulo):
    '''
    distribución_ciclo: entrega histograma con la distribución de la variable en estudio, esta función tiene la 
    particularidad de filtrar el dataframe de acuerdo a una categoria y clase especifica
    Parámetros
    dataframe: dataframe que contiene la muestra de observaciones
    variable: variable en estudio tipo string
    categoria: categoria por la cual se quiere filtrar el dataframe tipo string
    clase: clase distintiva de la categoria tipo string
    titulo:  titulo del histograma
    '''
    df_tmp = dataframe[dataframe[categoria]==clase]
    mean = round(np.mean(df_tmp[variable]),2)
    sns.distplot(df_tmp[variable]).set_title(titulo)
    plt.axvline(mean, color= 'r', ls='dotted', label = f'media: {mean}')
    plt.legend()

def transformar_dataframe(df):
    '''
    Retorna dataframe resumido por ciclos, el dataframe original debe tener la siguiente estrucutra:
        date: fecha de la lectura
        time: hora de la lectura
        do_level: nivel de oxígeno
        h2o_level: nivel de agua
        blower_hz: hz de giro del motor
        cycle_id: label para identificar los ciclos de funcionamiento´
    El dataframe de retorno tiene la siguiente estructura:
        ciclo : Id del ciclo
        total_hz: Total de Hz por ciclo
        do_mean_blwon: Media de oxigeno cuando el soplador está en operación durante el ciclo
        do_mean: Media de oxigeno durante el ciclo
        h2o_mean: Media de nivel de agua del ciclo
        inicio_ciclo: Fecha y hora en que inició el ciclo
        fin_ciclo: Fecha y hora en que terminó el ciclo
        duracion: Tiempo de duración del ciclo
        time_of_day: Momento del día en que transcurre el ciclo

    '''
    # 1. Total de Hz del soplador por ciclo
    # resumen de hz por ciclo se considerará la suma de todas las frecuencias registradas para cada ciclo
    d_hz = df.loc[:, ['cycle_id', 'blower_hz']].groupby(['cycle_id']).sum()
    d_hz.rename(columns= {'blower_hz': 'total_hz'}, inplace = True)

    # 2. Niveles de DO y H2O promedio por ciclo
    # resumen DO y H2O Level se considerará como la media de las observaciones resgitradas para cada ciclo
    d_do_h2o = df.loc[:, ['cycle_id', 'do_level', 'h2o_level']].groupby(['cycle_id']).mean()
    d_do_h2o.rename(columns= {'do_level': 'do_mean', 'h2o_level': 'h2o_mean'}, inplace = True)

    # 3. Niveles de DO cuando está el soplador encendido:
    d_do_blower = df[df['blower_hz']>0].loc[:, ['cycle_id', 'do_level']].groupby(['cycle_id']).mean()
    d_do_blower.rename(columns= {'do_level': 'do_mean_blwon'}, inplace = True)


    # 4. Datos sobre temporalidad de los ciclos
    d_min_time = df.loc[:, ['cycle_id', 'date_time']].groupby(['cycle_id']).min()
    d_min_time.rename(columns= {'date_time': 'inicio_ciclo'}, inplace = True)
    d_max_time = df.loc[:, ['cycle_id', 'date_time']].groupby(['cycle_id']).max()
    d_max_time.rename(columns= {'date_time': 'fin_ciclo'}, inplace = True)

        # Se Concatenan los arreglos realizados previamente 
    df_resume_cycle = pd.concat([d_hz, d_do_blower, d_do_h2o, d_min_time, d_max_time], axis = 1)
    df_resume_cycle.index = df_resume_cycle.index.map('Ciclo {}'.format)

        # duración del ciclo
    df_resume_cycle['duracion'] = df_resume_cycle['fin_ciclo'] - df_resume_cycle['inicio_ciclo']

    # 5. Momento del día del ciclo
    # Arreglo para identificar momento del dia que inicia el ciclo
    turno = {}

    for i, hora_inicio in enumerate(df_resume_cycle['inicio_ciclo'].dt.hour):
        key = df_resume_cycle.index[i]
        hora_termino = df_resume_cycle['fin_ciclo'].dt.hour.iloc[i]
        
        if hora_inicio >=6 and hora_inicio < 12 and hora_termino > 12:
            turno[key] = 'Media-Mañana'
        
        elif hora_inicio >= 6 and hora_inicio < 12:
            turno[key] = 'Mañana'
        
        elif hora_inicio >= 12 and hora_inicio < 19:
            turno[key] = 'Tarde'
            
        elif hora_inicio >= 19 and hora_inicio < 24:
            turno[key] = 'Noche'
            
        else:
            turno[key] = 'Madrugada'

    d_turno = pd.DataFrame([[key, turno[key]] for key in turno.keys()], columns=['cycle_id', 'time_of_day'])
    d_turno.set_index('cycle_id', inplace = True)

    # Se concatena el arreglo de turno al dataframe de resumen de ciclos
    df_resume_cycle = pd.concat([df_resume_cycle, d_turno], axis = 1)

    df_resume_cycle = df_resume_cycle.iloc[1:]
    df_resume_cycle = df_resume_cycle.rename_axis('ciclo').reset_index()
    return df_resume_cycle

def matriz_kmeans(df):
    '''
    Genera matriz estandarizada para ser utilizada en el modelo kmeans de clusterización de ciclos
    se debe entregar como input el dataframe retornado por la funcion transformar_dataframe  
    '''
    X_matrix = StandardScaler().fit_transform(df.iloc[:, 1:5])
    return X_matrix
