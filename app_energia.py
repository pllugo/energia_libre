#!/usr/bin/env python
'''
API Machine Learning
---------------------------
Autor: Pedro Luis Lugo García
Version: 1.0
 
Descripcion:
Se utiliza Flask para crear un WebServer que levanta un
modelo de inteligencia artificial con machine learning
y realizar predicciones o clasificaciones

Ejecución: Lanzar el programa y abrir en un navegador la siguiente dirección URL
http://127.0.0.1:5000/

'''

__author__ = "Pedro Luis Lugo Garcia"
__email__ = "pedro.lugo@unc.edu.ar"
__version__ = "1.0"

from typing import TYPE_CHECKING
import math
import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect, session, Response, send_file
from werkzeug.exceptions import abort
import numpy as np
import csv
import traceback
import io
import sys
import os
import base64
import json
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns

from flask_sqlalchemy import SQLAlchemy
import matplotlib
matplotlib.use('Agg')   # For multi thread, non-interactive backend (avoid run in main loop)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from decimal import Decimal
import pickle
model = pickle.load(open('clasificador.pkl', 'rb'))
regresion_saturados = pickle.load(open('covs_saturados.pkl', 'rb'))
regresion_insaturados = pickle.load(open('covs_insaturados.pkl', 'rb'))





app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key'

def tipo_cov(df): #Función para buscar dobles o triples enlaces en un compuesto
    lista_cov = list(df['Compuestos'])
    lista_tipo = list(df['Tipo'])
    lista_enlace = []
    for i in range(len(lista_cov)):
        cov = lista_cov[i]
        if '=' in cov or '=-' in cov:
            lista_enlace.append('insaturado')
        else:
            if 'insaturado-ciclico' in lista_tipo[i]:
                lista_enlace.append('insaturado')
            else:
                lista_enlace.append('saturado')
    return lista_enlace


def one_hot_encoding(df, column):
    df_copy = df.copy()
    # LabelEncoder
    le = LabelEncoder()
    label_encoding = le.fit_transform(df_copy[column])
    # OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoding = onehot_encoder.fit_transform(label_encoding.reshape(-1, 1))
    # Crear las columnas con el resultado del encoder
    one_hot_encoding_df = pd.DataFrame(one_hot_encoding, columns=le.classes_, dtype=int)
    # Agregar sufijo
    one_hot_encoding_df = one_hot_encoding_df.add_prefix(column+'_')
    # Unir nuevas columnas al dataset
    df_copy = df_copy.join(one_hot_encoding_df)
    # Eleminar vieja columna del dataset
    #df_copy = df_copy.drop([column], axis=1)
    return df_copy, label_encoding, one_hot_encoding


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


def get_cov(cov_id):
    conn = get_db_connection()
    cov = conn.execute('SELECT * FROM covs WHERE id = ?',
                        (cov_id,)).fetchone()
    conn.close()
    if cov is None:
        abort(404)
    return cov

@app.route('/')
def index():
    conn = get_db_connection()
    covs = conn.execute('SELECT * FROM covs').fetchall()
    conn.close()
    return render_template('index.html', covs=covs)

@app.route('/<int:cov_id>')
def cov(cov_id):
    cov = get_cov(cov_id)
    return render_template('cov.html', cov=cov)

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        formula = request.form['formula']
        tipo = request.form['tipo']
        kCl = float(request.form['kCl'])
        kOH = float(request.form['kOH'])
        logkCl = math.log10(kCl)
        logkOH = math.log10(kOH)
        df_datos = pd.DataFrame({"Compuestos": [formula],
                                    "Tipo": [tipo],
                                        "kcl": [kCl],
                                        "koh": [kOH],                 
                                    "logkCl": [logkCl],
                                    "logkOH": [logkOH]}
                                     )
        lista_enlace = tipo_cov(df_datos)
        if lista_enlace[0] == 'insaturado':
            enlace = 'insaturado'
        else:
            enlace = 'saturado'
        if not formula:
            flash('La Formula es requerida!')
        else:
            conn = get_db_connection()
            conn.execute('INSERT INTO covs (formula, tipo, kcl, koh, logkCl, logkOH, enlace) VALUES (?, ?, ?, ?, ?, ?, ?)',
                         (formula, tipo, kCl, kOH, logkCl, logkOH, enlace))
            conn.commit()
            conn.close()
        return redirect(url_for('index'))
    return render_template('create.html')


@app.route('/<int:id>/edit', methods=['GET', 'POST'])
def edit(id):
    cov = get_cov(id)

    if request.method == 'POST':
        formula = request.form['formula']
        tipo = request.form['tipo']
        kCl = float(request.form['kCl'])
        kOH = float(request.form['kOH'])
        logkCl = math.log10(kCl)
        logkOH = math.log10(kOH)
        df_datos = pd.DataFrame({"Compuestos": [formula],
                                    "Tipo": [tipo],
                                        "kcl": [kCl],
                                        "koh": [kOH],                 
                                    "logkCl": [logkCl],
                                    "logkOH": [logkOH]}
                                     )
        lista_enlace = tipo_cov(df_datos)
        if lista_enlace[0] == 'insaturado':
            enlace = 'insaturado'
        else:
            enlace = 'saturado'
        if not formula:
            flash('Formula es requerida!')
        else:
            conn = get_db_connection()
            conn.execute('UPDATE covs SET formula = ?, tipo = ?, kCl = ?, kOH = ?, logkcl = ?, logkOH = ?, enlace = ?'
                         ' WHERE id = ?',
                         (formula, tipo, kCl, kOH, logkCl, logkOH, enlace, id))
            conn.commit()
            conn.close()
            return redirect(url_for('index'))

    return render_template('edit.html', cov=cov)


@app.route('/<int:id>/delete', methods=['GET', 'POST'])
def delete(id):
    cov = get_cov(id)
    conn = get_db_connection()
    conn.execute('DELETE FROM covs WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash('"{}" was successfully deleted!'.format(cov['formula']))
    return redirect(url_for('index'))


def custumer_overview(): #Función para generar la grafica de COVs insaturados y saturados
    df = pd.read_csv("base_datos.csv")
    df6 = df.copy()
    lista_enlace = tipo_cov(df6)
    df2 = pd.DataFrame({'enlace': lista_enlace})
    df = df.join(df2)
    fig = plt.figure()
    ax = fig.add_subplot()
    sns.scatterplot(data=df, x="logkCl", y="logkOH", hue="enlace")

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    encoded_img = base64.encodebytes(output.getvalue())
    plt.close(fig)  # Cerramos la imagen para que no consuma memoria del sistema
    return encoded_img


def grafica_compuesto(x_compuesto, y_compuesto): #Función para comprobar donde se encuentra ubicado
    df = pd.read_csv("base_datos.csv")           # un COV en la grafica
    df6 = df.copy()
    lista_enlace = tipo_cov(df6)
    df2 = pd.DataFrame({'enlace': lista_enlace})
    df = df.join(df2)
    df_saturados = df[df['enlace'] == 'saturado']
    df_insaturados = df[df['enlace'] == 'insaturado']
    df_compuesto = pd.DataFrame({'logkCl': x_compuesto, 'logkOH':y_compuesto})
    fig = plt.figure()
    ax = fig.add_subplot()
    sns.scatterplot(x=df_insaturados['logkCl'], y=df_insaturados['logkOH'], color = 'darkOrange',label='Insaturados', ax=ax)
    sns.scatterplot(x=df_saturados['logkCl'], y=df_saturados['logkOH'], color = 'dodgerblue',label='Saturados', ax=ax)
    sns.scatterplot(x=df_compuesto['logkCl'], y=df_compuesto['logkOH'], color = 'red', label='COVs Ingresados', ax=ax)
   
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    encoded = base64.encodebytes(output.getvalue())
    plt.close(fig)  # Cerramos la imagen para que no consuma memoria del sistema
    return encoded


def calculo_regresión(lista_x, lista_y): #Función para calcular la regresión lineal
    n = len(lista_x)
    x = np.array(lista_x)
    y =np.array(lista_y)
    suma_x = sum(x)
    suma_y = sum(y)
    suma_x2 = sum(x*x)
    suma_y2 = sum(y*y)
    sum_xy = sum(x*y)
    prom_x = suma_x / n
    prom_y = suma_y / n
    pendiente = (suma_x * suma_y - n*sum_xy) / (suma_x**2 - n*suma_x2)
    intercepto = prom_y - pendiente * prom_x
    sigmax = np.sqrt(suma_x2 / n - prom_x**2)
    sigmay = np.sqrt(suma_y2 / n - prom_y**2)
    sigmaxy = sum_xy / n - prom_x * prom_y
    r2 = (sigmaxy / (sigmax * sigmay))**2
    return pendiente, intercepto, r2


def grafica_insaturado(df_compuesto): #Función para generar la grafica de COVs insaturados 
    df = pd.read_csv("base_datos.csv")
    df6 = df.copy()
    lista_enlace = tipo_cov(df6)
    df2 = pd.DataFrame({'enlace': lista_enlace})
    df = df.join(df2)
    df = df.append(df_compuesto)
    y_compuesto = list(df_compuesto['Compuestos'].values)
    df_insaturados = df[df['enlace'] == 'insaturado']
    x_ins = df_insaturados['logkCl'].values
    y_ins = df_insaturados['logkOH'].values
    m, b, r2 = calculo_regresión(x_ins, y_ins)
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    fig = plt.figure()
    ax = fig.add_subplot()
    sns.scatterplot(x=df_insaturados['logkCl'], y=df_insaturados['logkOH'], color = 'darkOrange',label='Insaturados', ax=ax)
    sns.scatterplot(x=df_compuesto['logkCl'], y=df_compuesto['logkOH'], color = 'red', label='COVs Ingresados', ax=ax)
    sns.lineplot(x=df_insaturados['logkCl'], y=m*x_ins + b, color = 'blue',label='Ajuste', ax=ax)
    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.05, y=0.10, units='inches')
    i = 0
    for x, y in zip(df_compuesto['logkCl'], df_compuesto['logkOH']):
        plt.plot(x, y, 'ro')
        plt.text(x, y, y_compuesto[i], transform=trans_offset, fontsize=7, horizontalalignment='center')
        i = i + 1

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    encoded = base64.encodebytes(output.getvalue())
    plt.close(fig)  # Cerramos la imagen para que no consuma memoria del sistema
    return encoded, round(m,2), round(b,2), round(r2,2)


def grafica_saturado(df_compuesto): #Función para generar la grafica de COVs saturados
    df = pd.read_csv("base_datos.csv")
    df6 = df.copy()
    lista_enlace = tipo_cov(df6)
    df2 = pd.DataFrame({'enlace': lista_enlace})
    df = df.join(df2)
    df = df.append(df_compuesto)
    y_compuesto = list(df_compuesto['Compuestos'].values)
    df_saturados = df[df['enlace'] == 'saturado']
    x_sat = df_saturados['logkCl'].values
    y_sat = df_saturados['logkOH'].values
    m, b, r2 = calculo_regresión(x_sat, y_sat)
    
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    fig = plt.figure()
    ax = fig.add_subplot()
    sns.scatterplot(x=df_saturados['logkCl'], y=df_saturados['logkOH'], color = 'dodgerblue',label='Saturados', ax=ax)
    sns.scatterplot(x=df_compuesto['logkCl'], y=df_compuesto['logkOH'], color = 'red', label='COVs Ingresados', ax=ax)
    sns.lineplot(x=df_saturados['logkCl'], y=m*x_sat + b, color = 'blue',label='Ajuste', ax=ax)
    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.05, y=0.10, units='inches')
    i = 0
    for x, y in zip(df_compuesto['logkCl'], df_compuesto['logkOH']):
        plt.plot(x, y, 'ro')
        plt.text(x, y, y_compuesto[i], transform=trans_offset, fontsize=7, horizontalalignment='center')
        i = i + 1
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    encoded = base64.encodebytes(output.getvalue())
    plt.close(fig)  # Cerramos la imagen para que no consuma memoria del sistema
    return encoded, round(m,2), round(b,2), round(r2,2)


def format_e(n): #Función para decimales
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


@app.route('/resultados')
def grafica():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    lista_cov = []
    for row in c.execute('SELECT formula, tipo, kcl, koh, logkCl, logkOH, enlace FROM covs'):
        lista_cov.append(row)
    resultado = []
    lista_compuestos = []
    lista_tipos = []
    lista_kcl = []
    lista_koh = []
    lista_logkcl = []
    lista_logkoh = []
    lista_enlace = []
    for i in range(len(lista_cov)):
        resultado = list(lista_cov[i])
        lista_compuestos.append(resultado[0])
        lista_tipos.append(resultado[1])
        lista_kcl.append(resultado[2])
        lista_koh.append(resultado[3])
        lista_logkcl.append(resultado[4])
        lista_logkoh.append(resultado[5])
        lista_enlace.append(resultado[6])
        #Tomo los datos AP, POCP, tiempo y GWP     
        df_datos = pd.DataFrame({"Compuestos": lista_compuestos,
                                    "Tipo": lista_tipos,
                                    "kcl": lista_kcl,
                                    "koh": lista_koh,                 
                                "logkCl": lista_logkcl,
                                "logkOH": lista_logkoh}
                                )
        df_prueba = pd.DataFrame({'enlace': lista_enlace})
        df_modelo = df_datos.join(df_prueba)
        dfcod, insaturado, saturado = one_hot_encoding(df_modelo, 'enlace')
        df_juego = dfcod.copy()
    conn.close()
    encoded_img = custumer_overview()
    prediction_image = grafica_compuesto(lista_logkcl, lista_logkoh)
    prediccion_koh = []
    if 8 in dfcod.shape:
        if lista_enlace[0] == 'saturado':
            #Todos los COVs son Saturados
            df_juego['anexo'] = df_juego.apply(lambda x: 0 if x['enlace'] == 'saturado' else 0, axis=1)
            X_prueba = df_juego.drop(['Compuestos', 'Tipo', 'enlace'], axis=1).values
            df_saturados = df_juego[df_juego['enlace'] == 'saturado']
            #Predecir el valor logkOH con el lr de pickle
            y_sat = regresion_saturados.predict(df_saturados[['logkCl']].values)
            for i in range(len(y_sat)):
                prediccion_koh.append(format_e(Decimal(10**y_sat[i])))
        else:
            #Todos los COVs son Insaturados
            df_juego['anexo'] = df_juego.apply(lambda x: 0 if x['enlace'] == 'insaturado' else 0, axis=1)
            X_prueba = df_juego.drop(['Compuestos', 'Tipo', 'enlace'], axis=1).values
            df_insaturados = df_juego[df_juego['enlace'] == 'insaturado']
            #Predecir el valor logkOH con el lr de pickle
            y_ins = regresion_insaturados.predict(df_insaturados[['logkCl']].values)
            for i in range(len(y_ins)):
                prediccion_koh.append(format_e(Decimal(10**y_ins[i])))
    else:#Existen compuestos tanto saturados como insaturados
        X_prueba = df_juego.drop(['Compuestos', 'Tipo', 'enlace'], axis=1).values
        df_saturados = df_juego[df_juego['enlace'] == 'saturado']
        y_sat = regresion_saturados.predict(df_saturados[['logkCl']].values)
        for i in range(len(y_sat)):
            prediccion_koh.append(format_e(Decimal(10**y_sat[i])))
        df_insaturados = df_juego[df_juego['enlace'] == 'insaturado']
        y_ins = regresion_insaturados.predict(df_insaturados[['logkCl']].values)
        for i in range(len(y_ins)):
            prediccion_koh.append(format_e(Decimal(10**y_ins[i])))
    prediction = model.predict(X_prueba)
    lista_insaturados = []
    lista_saturados =[]
    for i in range(len(lista_compuestos)):
        lista_insaturados.append('insaturado')
        lista_saturados.append('saturado')
    df_enlaceins = pd.DataFrame({'enlace': lista_insaturados})
    df_ins= df_datos.join(df_enlaceins)
    regresion_image, m_ins, b_ins, r2_ins = grafica_insaturado(df_ins)
    df_enlacesat = pd.DataFrame({'enlace': lista_saturados})
    df_satu= df_datos.join(df_enlacesat)
    saturado_image, m_sat, b_sat, r2_sat = grafica_saturado(df_satu)
    return render_template('resultados.html', overview_graph=encoded_img, prediction=prediction, prediction_image=prediction_image, lista_compuestos=lista_compuestos, prediccion_koh=prediccion_koh, regresion_image=regresion_image, saturado_image=saturado_image,  m_sat=m_sat, b_sat=b_sat, r2_sat=r2_sat, m_ins=m_ins, b_ins=b_ins, r2_ins=r2_ins)

if __name__ == "__main__":
    app.run(debug=True)