from flask import (Flask, render_template, url_for, request,redirect, jsonify,flash, session)
from flask_mysqldb import MySQL
import os as os
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix

app = Flask(__name__)

UPLOAD_FOLDER = 'static/files'



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key ="mineriadedatos"
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'crudjg'
engine = create_engine('mysql+pymysql://root:@localhost/crudjg')
mysql = MySQL(app)

@app.route('/')
def main():
    link = mysql.connection.cursor()
    link.execute("SELECT * FROM jugadores")
    data = link.fetchall()
    return render_template('index.html', empleados=data)

@app.route('/login')
def login():
    return "en construcción!!"

@app.route('/viewemployes', methods=['POST', 'GET'])
def viewemployes():
    if request.method == 'POST':
        id = request.form['ID']
        link = mysql.connection.cursor()
        link.execute("SELECT * FROM jugadores WHERE ID = %s", [id])
        data = link.fetchall()
    return jsonify({'htmlresponse': render_template('viewemployes.html', empleados=data)})

@app.route('/addemployes', methods=['POST', 'GET'])
def addemployes():
    if request.method == 'POST':
        Id = request.form["ID"]
        Nombre = request.form["Nombre"]
        Clasificacion = request.form["Clasificacion"]
        Altura_Jugador = request.form["Altura_Jugador"]
        Peso_Jugador = request.form["Peso_Jugador"]
        Pie_Dominante = request.form["Pie_Dominante"]
        Edad = request.form["Edad"]
        Control_Balon = request.form["Control_Balon"]
        Posicion_Ataque = request.form["Posicion_Ataque"]
        Pase_Corto = request.form["Pase_Corto"]
        Pase_Largo = request.form["Pase_Largo"]
        Velocidad = request.form["Velocidad"]
        Stamina = request.form["Stamina"]
        Agilidad = request.form["Agilidad"]
        Cabezazo = request.form["Cabezazo"]
        Potencia_Disparo = request.form["Potencia_Disparo"]
        Remate = request.form["Remate"]
        Disparo_Largo = request.form["Disparo_Largo"]
        Penaltis = request.form["Penaltis"]
        link = mysql.connection.cursor()
        link.execute("INSERT INTO `jugadores` (`ID`,`Nombre`, `Clasificacion`, `Altura_Jugador`, `Peso_Jugador`, `Pie_Dominante`, `Edad`, `Control_Balon`, `Posicion_Ataque`, `Pase_Corto`, `Pase_Largo`, `Velocidad`, `Stamina`, `Agilidad`, `Cabezazo`, `Potencia_Disparo`, `Remate`, `Disparo_Largo`, `Penaltis`) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        ([Id], [Nombre], [Clasificacion], [Altura_Jugador], [Peso_Jugador], [Pie_Dominante], [Edad], [Control_Balon], [Posicion_Ataque]
        , [Pase_Corto], [Pase_Largo], [Velocidad], [Stamina], [Agilidad],[Cabezazo], [Potencia_Disparo], [Remate], [Disparo_Largo], [Penaltis]))
        mysql.connection.commit()
        link.close()
        flash("Jugador registrado correctamente")
        return redirect(url_for('main'))


@app.route('/updateemployes', methods=['POST', 'GET'])
def updateemployes():
    if request.method == 'POST':
        Id = request.form["ID"]
        Nombre = request.form["Nombre"]
        Clasificacion = request.form["Clasificacion"]
        Altura_Jugador = request.form["Altura_Jugador"]
        Peso_Jugador = request.form["Peso_Jugador"]
        Pie_Dominante = request.form["Pie_Dominante"]
        Edad = request.form["Edad"]
        Control_Balon = request.form["Control_Balon"]
        Posicion_Ataque = request.form["Posicion_Ataque"]
        Pase_Corto = request.form["Pase_Corto"]
        Pase_Largo = request.form["Pase_Largo"]
        Velocidad = request.form["Velocidad"]
        Stamina = request.form["Stamina"]
        Agilidad = request.form["Agilidad"]
        Cabezazo = request.form["Cabezazo"]
        Potencia_Disparo = request.form["Potencia_Disparo"]
        Remate = request.form["Remate"]
        Disparo_Largo = request.form["Disparo_Largo"]
        Penaltis = request.form["Penaltis"]
        link = mysql.connection.cursor()
        link.execute("UPDATE jugadores SET Nombre= %s, Clasificacion= %s, Altura_Jugador= %s, Peso_Jugador= %s, Pie_Dominante= %s, Edad= %s, Control_Balon= %s, Posicion_Ataque= %s, Pase_Corto= %s, Pase_Largo= %s, Velocidad= %s, Stamina= %s, Agilidad= %s, Cabezazo= %s, Potencia_Disparo= %s, Remate= %s, Disparo_Largo= %s, Penaltis= %s WHERE ID=%s",
                        (Nombre, Clasificacion, Altura_Jugador, Peso_Jugador, Pie_Dominante, Edad, Control_Balon, Posicion_Ataque
        , Pase_Corto, Pase_Largo, Velocidad, Stamina, Agilidad,Cabezazo, Potencia_Disparo, Remate, Disparo_Largo, Penaltis, Id))
        mysql.connection.commit()
        link.close()
        flash("Jugador actualizado correctamente")
        return redirect(url_for('main'))

@app.route('/deleteemployes/<string:Id>', methods=['POST', 'GET'])
def deleteemployes(Id):
    if request.method == 'GET':
        link = mysql.connection.cursor()
        link.execute("DELETE FROM jugadores WHERE ID=%s",[Id])
        mysql.connection.commit()
        link.close()
        flash("Jugador eliminado correctamente")
        return redirect(url_for('main'))

@app.route('/cargarcsv')
def cargarcsv():
    return render_template('cargarcsv.html')


@app.route('/uploadcsv', methods=['POST','GET'])
def uploadcsv():
    if request.method == 'POST':
        upload_file = request.files['csvfile']
        if upload_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],
             upload_file.filename)
            upload_file.save(file_path)
            grabarCSV(file_path)
        flash("DataSet Cargado correctamente!!!")
        return redirect(url_for('cargarcsv'))

def grabarCSV(file_path):
    columnas = ['ID','Nombre','Clasificacion','Altura_Jugador','Peso_Jugador','Pie_Dominante'
                ,'Edad','Control_Balon','Posicion_Ataque','Pase_Corto','Pase_Largo','Velocidad'
                ,'Stamina','Agilidad','Cabezazo','Potencia_Disparo','Remate','Disparo_Largo','Penaltis']
    csvData = pd.read_csv(file_path)
    link = mysql.connection.cursor()
    for i, row in csvData.iterrows():
        sql = "INSERT INTO jugadores (ID,Nombre,Clasificacion,Altura_Jugador,Peso_Jugador,Pie_Dominante ,Edad,Control_Balon,Posicion_Ataque,Pase_Corto,Pase_Largo,Velocidad,Stamina,Agilidad,Cabezazo,Potencia_Disparo,Remate,Disparo_Largo,Penaltis) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        valores = (row['ID'],row['Nombre'],row['Clasificacion'],row['Altura_Jugador'],row['Peso_Jugador'],row['Pie_Dominante']
                ,row['Edad'],row['Control_Balon'],row['Posicion_Ataque'],row['Pase_Corto'],row['Pase_Largo'],row['Velocidad']
                ,row['Stamina'],row['Agilidad'],row['Cabezazo'],row['Potencia_Disparo'],row['Remate'],row['Disparo_Largo'],row['Penaltis'])
        link.execute(sql,valores)
        mysql.connection.commit()

@app.route('/kmeans')
def K_Means():
    return render_template('kmeans.html')

@app.route('/EKMeans', methods=['POST'])
def EKMeans():
        if request.method == "POST":
         variable1 = request.form.get('variable1')
         variable2 = request.form.get('variable2')
         if variable1 == variable2:
             return redirect(url_for('K_Means'))
        NumCentr = request.form.get('NumCentroides', type=int)
        df = pd.read_sql("SELECT * FROM jugadores", engine)
        X = df.filter([variable1, variable2])
        k_means = KMeans(n_clusters=NumCentr).fit(X)
        centroides = k_means.cluster_centers_
        etiquetas = k_means.labels_
        pca_X = PCA(n_components=2).fit(X)
        pca = pca_X.transform(X)
        centroides_pca = pca_X.transform(centroides)
        colores = ['yellow','green','pink','cyan','black']
        colores_cluster = [colores[etiquetas[i]] for i in range(len(pca))]
        plt.figure(figsize=(10,6))
        plt.scatter(pca[:,0],pca[:,1], c= colores_cluster, marker = 'o', alpha=0.5)
        plt.scatter(centroides_pca[:,0],centroides_pca[:,1], c= 'brown', marker = 'D',s =150)
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        plt.title('KMeans')
        plt.savefig('static/kmeans.png')
        flash("static/kmeans.png")
        return redirect(url_for('K_Means'))


@app.route('/dbscan')
def DBscan():
    return render_template('dbscan.html')

@app.route('/EDBScan', methods=['POST'])
def EDBScan():
        if request.method == "POST":
         variable1 = request.form.get('variable1')
         variable2 = request.form.get('variable2')
         if variable1 == variable2:
             return redirect(url_for('DBscan'))
        df = pd.read_sql("SELECT * FROM jugadores", engine)
        X = df.filter([variable1, variable2])
        cluster = DBSCAN(eps=2, min_samples=6).fit(X)
        DBSCAN_dataset = X.copy()
        DBSCAN_dataset.loc[:,'Cluster'] = cluster.labels_ 
        outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster']==-1]
        fig, axe = plt.subplots(figsize=(10,10))
        sns.scatterplot(data=DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1],x= variable1, y= variable2, hue='Cluster', ax=axe, palette='Set2', legend='full', s=200)
        axe.scatter(outliers[variable1], outliers[variable2], 
                s=10, label='outliers', c="k")
        axe.legend()
        plt.setp(axe.get_legend().get_texts(), fontsize='12')
        plt.title('DBSCAN')
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        fig.savefig('static/dbSscan.png')
        flash("static/dbSscan.png")
        return redirect(url_for('DBscan'))


@app.route('/prediction')
def prediction():

        ## Grafica de barras ##
    X_barra = ["Support Vector Machine", "Tree Regressor", "Random Forest Regressor", "Nearest Neighbors Regressor"] 
    y_barra = []

    df1 = pd.read_sql("SELECT * FROM jugadores", engine)
    df1 = df1.drop(['ID', 'Nombre'], axis= 1)
    sc = StandardScaler()
    df2 = sc.fit_transform(df1)
    df = pd.DataFrame(df2, columns=df1.columns)
    X = df.iloc[:, 1:16].values
    y = df.iloc[:,0].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    ## Support Vector Machine ##

    regressor = SVR(kernel='rbf')
    regressor.fit(X_train, y_train)
    y_predSVR = regressor.predict(X_test)
    precisionSVR = r2_score(y_test, y_predSVR)*100
    y_barra.append(precisionSVR)

    ## Decision Tree Regressor ##

    model = DecisionTreeRegressor(max_depth = 5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    precisionTree = r2_score(y_test, predictions)*100
    y_barra.append(precisionTree)


    ## Random Forest Regressor ##
    
    randomf = RandomForestRegressor(n_estimators=100, random_state=0)
    randomf.fit(X_train, y_train)
    y_predRF = randomf.predict(X_test)
    precisionRFR = r2_score(y_test, y_predRF)*100
    y_barra.append(precisionRFR)

    ## Nearest Neighbors Regressor ##

    neigh = KNeighborsRegressor(n_neighbors=5)
    neigh.fit(X_train, y_train)
    y_predict = neigh.predict(X_test)
    precisionNN =  r2_score(y_test, y_predict)*100
    y_barra.append(precisionNN)

    plt.figure(figsize=(10,6))
    plt.bar(X_barra, y_barra, color=['orange', 'forestgreen', 'mediumorchid', 'royalblue'])
    plt.savefig('static/graficobarrapre.png')


    return render_template(
        'prediccion.html', 
        SVRpresicion = precisionSVR, 
        RLTpresicion = precisionTree, 
        RFRpresicion = precisionRFR, 
        NNpresicion = precisionNN)


@app.route('/clasification')
def clasification():

        ## Grafica de barras ##
    X_barra = ["KNeighbors", "Naive Bayes", "Tree Classifier", "Random Forest"] 
    y_barra = []

    df = pd.read_sql("SELECT * FROM jugadores", engine)
    df1 = df.drop(['ID', 'Nombre'], axis= 1)
    X = df1.iloc[:, df1.columns != 'Pie_Dominante'].values
    y = df.iloc[:,5].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    ##  KNeighbors Classifier  ##

    clasificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)
    Clasifkneighbor = precision_score(y_test, y_pred,  average='micro')*100
    y_barra.append(Clasifkneighbor)
    plt.figure(figsize=(5,6))
    plot_confusion_matrix(clasificador, X_test, y_test)  
    plt.savefig('static/kneighborgrafic.png')

    ## Gaussian NB Classifier ##

    naivebayes = GaussianNB()
    naivebayes = naivebayes.fit(X_train, y_train)
    y_predNB = naivebayes.predict(X_test)
    ClasifNBPRec = accuracy_score(y_test, y_predNB)*100
    y_barra.append(ClasifNBPRec)
    plt.figure(figsize=(5,6))
    plot_confusion_matrix(naivebayes, X_test, y_test)  
    plt.savefig('static/NBPRecgrafic.png')
    

    ## DecisionTree Classifier ##
    
    arbolito = DecisionTreeClassifier()
    arbolito = arbolito.fit(X_train, y_train)
    y_predTree = arbolito.predict(X_test)
    ClasifTree = precision_score(y_test, y_predTree,  average='micro')*100
    y_barra.append(ClasifTree)
    plt.figure(figsize=(5,6))
    plot_confusion_matrix(arbolito, X_test, y_test)  
    plt.savefig('static/Treegrafic.png')

    ## Random Forest Classifier ##

    random = RandomForestClassifier(max_depth=2, random_state=0)
    random.fit(X_train, y_train)
    y_predict = random.predict(X_test)
    ClasifRandomF =  accuracy_score(y_test, y_predict)*100
    y_barra.append(ClasifRandomF)
    plt.figure(figsize=(5,6))
    plot_confusion_matrix(random, X_test, y_test)  
    plt.savefig('static/RandomFgrafic.png')


    plt.figure(figsize=(10,6))
    plt.bar(X_barra, y_barra, color=['deeppink', 'goldenrod', 'lightcyan', 'maroon'])
    plt.savefig('static/graficobarra.png')

    return render_template(
        'clasificacion.html', 
        Clasifkneighbor = Clasifkneighbor, 
        ClasifNBPRec = ClasifNBPRec, 
        ClasifTree = ClasifTree, 
        ClasifRandomF = ClasifRandomF)


if __name__=='__main__':
    app.run(port=5000, debug=True)