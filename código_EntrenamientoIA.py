# Paso 1: Crear el Dataset
# --------------------------

import pandas as pd

# Crear el conjunto de datos
data = {
    'Horas de Estudio': [5, 3, 8, 2, 7, 6, 4, 9, 3, 6],
    'Nivel de Conocimiento Previo': [7, 5, 8, 3, 9, 6, 4, 10, 5, 7],
    'Asistencia a Clases': [80, 60, 90, 50, 85, 75, 65, 95, 70, 80],
    'Promedio de Tareas': [8, 5, 9, 4, 8, 7, 6, 9, 5, 7],
    'Tipo de Estudiante': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    'Resultado': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 1 = Aprobado, 0 = No Aprobado
}

# Convertir a un DataFrame
df = pd.DataFrame(data)

# Mostrar el DataFrame
print("Dataframe:")
print(df)



# Paso 2: Preprocesar los Datos
# --------------------------

from sklearn.preprocessing import StandardScaler

# Separar las características (entradas) y la etiqueta (salida)
X = df.drop('Resultado', axis=1)  # Características
y = df['Resultado']  # Etiqueta (resultado)

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Mostrar las características normalizadas
print("Características normalizadas:")
print(X_scaled)



# Paso 3: Dividir los Datos en Entrenamiento y Prueba
# --------------------------

from sklearn.model_selection import train_test_split

# Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Mostrar las dimensiones de los conjuntos de entrenamiento y prueba
print("Dimensiones de los conjuntos de entrenamiento y prueba:")
print(f"Datos de entrenamiento: {X_train.shape}")
print(f"Datos de prueba: {X_test.shape}")



# Paso 4: Elegir y Entrenar el Modelo
# --------------------------

from sklearn.tree import DecisionTreeClassifier

# Crear el modelo de Árbol de Decisión
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)



# Paso 5: Evaluar el Modelo
# --------------------------

from sklearn.metrics import accuracy_score, confusion_matrix

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Mostrar la matriz de confusión
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))



# Paso 6: Hacer Predicciones
# --------------------------


# Ejemplo de predicción para un nuevo estudiante
# Un nuevo estudiante tiene: 
# Estudiante A
# 6 horas de estudio, nivel de conocimiento previo 7, 80% de asistencia a clases, promedio de tareas 7 y estudia solo (1)


new_student = pd.DataFrame([[6, 7, 80, 7, 1]], columns=['Horas de Estudio', 'Nivel de Conocimiento Previo', 
                                                    'Asistencia a Clases', 'Promedio de Tareas', 'Tipo de Estudiante'])


# Estudiante B
# 10 horas de estudio, nivel de conocimiento previo 5, 20% de asistencia a clases, promedio de tareas 6 y estudia en grupo (0)


# new_student = pd.DataFrame([[10, 5, 20, 6, 0]], columns=['Horas de Estudio', 'Nivel de Conocimiento Previo', 
#                                                     'Asistencia a Clases', 'Promedio de Tareas', 'Tipo de Estudiante'])


# Normalizar las características del nuevo estudiante
new_student_scaled = scaler.transform(new_student)

# Hacer la predicción
prediction = model.predict(new_student_scaled)

# Mostrar el resultado

print("\nNuevo estudiante:\n")
print(new_student)

if prediction == 1:
    print("\nEl estudiante ha aprobado.\n")
else:
    print("\nEl estudiante no ha aprobado.\n")
