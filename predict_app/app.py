import joblib
import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    provincia = request.form['provincia']
    cultivo = request.form['cultivo']
    idcompania = int(request.form['id-compania'])
    sup_sembrada = float(request.form['sup-sembrada'])
    sup_cosecha = float(request.form['sup-cosechada'])
    produccion = float(request.form['produccion'])

    #Aqui podrias llamar a tu funcion prediction
    #prediction(provincia, cultivo, idcompania, sup_sembrada, sup_cosecha, produccion)

    # Aquí puedes agregar el código para procesar los datos y hacer la predicción
    # Por ahora, solo devolvemos los datos recibidos
    result = {
        'provincia':provincia,
        'cultivo':cultivo,
        'idcompania':idcompania,
        'sup_sembrada':sup_sembrada,
        'sup_cosechada':sup_cosecha,
        'produccion':produccion
    }

    #values=[provincia, cultivo, idcompania, sup_sembrada, sup_cosecha, produccion]
    model = joblib.load(open('D:\Ale\est_ale\predict_app\model_2', 'rb'))
    processor = joblib.load(open('D:\Ale\est_ale\predict_app\processor.pkl', 'rb'))
    

    features = np.array([[provincia, cultivo, idcompania, sup_sembrada, sup_cosecha, produccion]], dtype=object)

    # Transform the features using the preprocessor
    transformed_features = processor.transform(features)

    # Make the prediction
    predicted_yield = model.predict(transformed_features).reshape(1, -1)

    print(predicted_yield)

    return f"Resultado: {predicted_yield}"

if __name__ == '__main__':
    app.run(debug=True)