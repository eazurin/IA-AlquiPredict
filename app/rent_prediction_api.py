from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Cargar el modelo, el scaler y las columnas
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('columns.pkl', 'rb') as columns_file:
    columns = pickle.load(columns_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del request
        input_data = request.json
        print("Datos de entrada recibidos:")
        print(input_data)

        # Crear un DataFrame con los datos de entrada
        input_df = pd.DataFrame([input_data])

        # Asegurarnos de que todas las columnas necesarias estén en el DataFrame
        for column in columns:
            if column not in input_df.columns:
                input_df[column] = 0  # Añadimos una columna con valor 0 si falta

        # Ordenar las columnas para que coincidan con el orden usado en el entrenamiento
        input_df = input_df[columns]

        # Escalar los datos de entrada utilizando el scaler cargado
        input_scaled = scaler.transform(input_df)

        # Hacer la predicción
        prediction = model.predict(input_scaled)[0]

        # Devolver la predicción en formato JSON
        return jsonify({"predicted_rent": prediction})

    except Exception as e:
        return str(e), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
