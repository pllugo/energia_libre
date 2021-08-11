# App Química Atmosférica (Energía Libre) Machine Learning 📊 🏭
Este es un proyecto de creación de una app para conocer el comportamiento de un grupo de COVs, dependiento si su estructura se encuentran enlaces saturados e insaturados.

## Metodologia Experimental

### Constantes cinéticas con radicales OH y Cl atmosféricos
La oxidación troposférica de compuestos orgánicos volátiles con radicales atmosféricos como OH (radical mas abundante en la atmosfera), Cl (radical en altas concentraciones en zonas maritimas y costeras) es importante en la química del aire para conocer cuan veloz los COVs reaccionan con dichos oxidantes y asi determinar su tiempo de residencia en la atmosfera y otros parámetros como el POCP (potencial de creación de ozono troposferico), AP (potencial de acidificación), entre otros. Conociendo estos impactos y su estructura de enlace, insaturado (C=C ó C☰C) o saturado (C-C), se podria predecir su comportamiento mediante una gráfica que represente el logkCl vs logkOH. Además de conocer si la reacción que se lleva a cabo es mediante adición al doble enlace (C=C) o abstracción de hidrogeno (C-H).

Como se puede observar en la imagen de abajo, en donde se quiere conocer el comportamiento de un grupo de dicetonas fluoradas CF3C(O)CH=C(OH)CH3, CF3C(O)CH=C(OH)CH2CH3, CF3C(O)CH=C(OH)CH(CH3)2 conociendo sus constantes cinéticas con radicales OH (kOH) y Cl (kCl)

Las dicetonas fluoradas a estudiar tienen esta estructura, pueden comportarse de forma ceto y enol (Perdro y col, 2021)

![image](https://user-images.githubusercontent.com/72478141/129092711-985cc293-5ebb-4380-9cb4-46660cb8f3d5.png)


![image](https://user-images.githubusercontent.com/72478141/129085073-0e098bd7-2237-4072-87f6-11fb7c444673.png)

 ### Técnias de medición (datos)

1. FTIR: Espectroscopía Infrarroja con Transformada de Fourier
2. GC-FID: Cromatografía Gaseosa con Detector de Llama

# Pre-requisitos de la app 📋

La app requiere la instalación de las siguientes librerias:

* from typing import TYPE_CHECKING
* import math
* import sqlite3
* from flask import Flask, render_template, request, url_for, flash, redirect, session, Response, send_file
* from werkzeug.exceptions import abort
* import numpy as np
* import csv
* import traceback
* import io
* import sys
* import os
* import base64
* import json
* from datetime import datetime, timedelta
* import pandas as pd
* import seaborn as sns

* from flask_sqlalchemy import SQLAlchemy
* import matplotlib
* matplotlib.use('Agg')   # For multi thread, non-interactive backend (avoid run in main loop)
* import matplotlib.pyplot as plt
* from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
* from matplotlib.figure import Figure
* import matplotlib.image as mpimg
* import matplotlib.transforms as mtransforms

* from sklearn.cluster import KMeans
* from sklearn.metrics import silhouette_score
* from sklearn.preprocessing import LabelEncoder
* from sklearn.preprocessing import OneHotEncoder

* from decimal import Decimal
* import pickle

### Estructura de la App Química Atmosférica ⌨

- _app_energia_ : El programa app es donde se desarrolla la parte central de la aplicación y llama a las demás funciones, métodos, etc.
- _init_db_ : El programa crea la base de datos donde se guardará la información de los compuestos a ingresar en la app
- _clasificador.pkl_: Este clasificador se obtiene del análisis en jupyter lab de los datos que se encuentran en el archivo (base_datos.csv) donde se encuentra codificado el clasificador por Regresion logistic.
- _cov_insaturados_:Este archivo es el label encoder para la creación del dataframe a partir del archivo (base_datos.csv).
- _cov_saturados_: Este archivo es el label encoder para la creación del dataframe a partir del archivo (base_datos.csv).

### Datos de entrada de la App 💻

1. Se debe dar click al boton que dice Ingresar COV y colocar los datos del compuesto como aparece en la imagen 


![image](https://user-images.githubusercontent.com/72478141/129087373-25fbf071-c198-4cc8-9a43-e69ed6a031f7.png)


2. Luego automaticamente la app lleva a la pagina donde hay que ingresar los datos que alli se piden (las constantes cinéticas kOH y KCl se deben ingresar en notación cientifica y el simbolo separador de decimales es el punto (.), luego dar click en el boton submit

![image](https://user-images.githubusercontent.com/72478141/129087903-e0f84977-7bcc-42f3-b7cc-23a495a6d3aa.png)

3. Después de dar click vuelve a la pagina de inicio ya con el COV ingresado

![image](https://user-images.githubusercontent.com/72478141/129088375-dfea4fe1-bf9c-44e2-93c1-75b7ed37a24d.png)

4. Si desea ingresar mas COVs solo debe dar click a la pestaña ingresar COV. Luego de agregar los demás compuestos, la página queda de ésta manera:

![image](https://user-images.githubusercontent.com/72478141/129089638-e6f52aa9-7a94-4594-bb11-d01e007c93e7.png)

5. Luego al dar click en la pestaña de Resultados, dará lugar a la siguiente pagina, donde en un recuadro, se dan los resultados de los COVs ingresados, su comportamiento y predicción de constantes cinéticas con radical OH:

![image](https://user-images.githubusercontent.com/72478141/129090080-6659282e-7466-4263-a8de-50767a87ed89.png)

6. En la parte inferior se encuentran las gráficas de energía libre, en donde la primera gráfica representa los valores de la base de datos de COVs insaturados y saturados

![image](https://user-images.githubusercontent.com/72478141/129090242-3fbcb39b-64ff-4639-b55a-f5675dd77fca.png)

7. La siguiente gráfica es la representación de los COVs ingresados para conocer en que parte se encuentran (insaturados o saturados)

![image](https://user-images.githubusercontent.com/72478141/129090326-309b7959-1a4f-4b6f-af99-9544bd7d7776.png)

8.Ahora se quiere representar los COVs ingresados si éstos fueran insaturados en una gráfica de energía libre, por lo cual se da la información de la ecuación de la recta obtenida por minimos cuadrados o regresión lineal

![image](https://user-images.githubusercontent.com/72478141/129090866-b2a88e08-272d-45c9-85da-e98c3678d3a2.png)

9. Por último se muestra la grafica de los COVs si éstos fueran saturados, con su correspondiente ecuación de la recta obtenida por minimos cuadrados o regresión lineal

![image](https://user-images.githubusercontent.com/72478141/129091268-f62d7034-11f4-4062-8910-f5c1a01dda55.png)


### Conclusión

Se puede ver en la Figura de compuestos saturados junto a los COVs ingresados, que las dicetonas fluoradas están lejos de la correlación lineal de los datos (en el extremo), por lo tanto muestra un comportamiento de reacción en donde el mecanismo de abstracción de hidrógeno juega un papel menos importante.
En contraste, la Figura de compuestos insaturados junto a los COVs ingresados muestra un comportamiento de reacción por adición al doble enlace (C=C), evidenciando que las dicetonas fluoradas (COVs ingresados) en la fase gaseosa se encuentran principalmente en su forma enol.


# Notas Importantes 📉

1. El archivo a ingresar para realizar las graficas de energía libre y la predicción de cálculo de constantes cinéticas con radicales OH, debe contener la información sobre la formula, tipo de compuesto (en el archivo es Tipo), kCl, kOH, logkCl y logkOH.


2. Si desea editar o eliminar algún compuesto, puede hacerlo en el boton EDIT que se encuentra debajo de cada COV ingresado 

# Contribución 🚀

Esta app de química atmosferica orientado al estudio de comportamiento de cinética de reacción química (Insaturados a traves de adicción al doble enlace C=C y Saturados a traves de la abstracción de átomos de hidrogeno C-H) de un Compuesto Orgánico Volátil (COV) con radicales OH y átomos de Cl en la atmósfera, contribuye al estudio que actualmente se esta llevando acabo en los laboratorios: LUQCA de la Universidad Nacional de Córdoba (Argentina) y en BUW de la Universidad de Wuppertal (Alemania) y con la finalidad de que sea un área de investigación (simulaciones, data science, etc) en ambos centros de investigación de quimica atmosferica.

# Versión 📌

Versión 1.01

# Autor ✒

* MSc. Pedro Lugo
- Becario Doctoral CONICET-UNC

