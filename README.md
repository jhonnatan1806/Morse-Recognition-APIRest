# Modelo de reconocimiento de vocales en código morse

Integrantes:
- Gabriel Barrientos
- Jhonnatan Espinoza

## Descripción

En esta oportunidad estamos usando Flask como un servidor que se comunica con un cliente en Javascript atraves de protocolos http,
nuestro servidor actua como un APIRest el cual nos permite interactuar con el modelo.

## Prerequisitos

Se a probado usando python en 2 versiones
- Python 3.9 (entorno virtual)
- Python 3.11 (despliege)

Para instalar todas las librerias necesarias debemos ejecutar el comando

```bash
pip install -r requirements.txt
```

## Rutas disponibles

**/upload [POST]**

Recibimos una imagen desde el cliente del tipo base64, ademas recibimos el directorio al cual pertenece,
esto atraves del verbo POST el cual recibimos como json con las variables `image` y `subfolder` respectivamente.

**/prepare [GET]**

Este metodo nos permite tomar todas las imagenes de la carpeta `/data/train` y generar nuestros archivos `X.npy` y  `y.npy`

**/train [GET]**

Con este metodo ponemos a entrenar al modelo con 80% de data para entrenamiento y 20% para validacion y un epochs de 200,
luego de entrenar el modelo lo serializamos y lo guardamos en `data/test/trained_model.h5` para su uso posterior.

**/download/<file> [GET]**

Nos permite descargar cualquier archivo que se encuentre alojado dentro de la carpeta `data/test/`, entre ellos:

- trained_model.h5 (modelo entrenado y serializado)
- X.npy (imagenes)
- y.npy (etiquetas correspondientes)

**/predict [POST]**

Recibimos un json desde el cliente con la imagen en base64 la decodificamos y la almacenamos temporalmente en `data/test`,
luego de eso obtenemos la imagen, la normalizamos, cargamos nuestro modelo serializado y evaluamos la imagen para obtener
un array con los resultados de la prediccion, finalmente es devuelto al cliente con la `letra` y el `accuracy` obtenido.
