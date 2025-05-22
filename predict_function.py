import tensorflow as tf
import numpy as np
import cv2

def carregar_modelo_e_prever(model_path, image_path):
    # Carregar o modelo
    modelo = tf.keras.models.load_model(model_path)

    # Carregar e pré-processar a imagem
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (299,299))
    img = img.squeeze()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)     # Adiciona dimensão de batch: (1, 299, 299, 3)

    # Fazer a previsão
    previsao = modelo.predict(img)

    # Obter o nome da classe
    probabilidade = previsao[0][1]

    return  probabilidade
carregar_modelo_e_prever(r'model\osteoarthritis_inception.keras', r'img_test\9094693_1.png')