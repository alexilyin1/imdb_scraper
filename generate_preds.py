# from model import model, X_test, y_test
import gunicorn
import plotly.express as px
import numpy as np
import cv2
import os
import base64
from itertools import chain
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.preprocessing import image
from keras.models import model_from_json
from keras.optimizers import adamax

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


def create_model():
    model = Sequential()
    model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(4))
    model.add(Activation('sigmoid'))
    return model


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__)
# server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    html.Div(id='movie_poster',
             style=dict(display='flex', justifyContent='center', height='500', width='250')),

    html.Div(id='classification',
             style=dict(display='flex', justifyContent='center')),

    dcc.Upload(html.Button('Upload a Poster'),
               id='upload_image',
               style=dict(display='flex', justifyContent='center')
               )
])


def parse_image(contents):
    return html.Div([
        html.Img(src=contents)
    ])


def image_to_cv(contents):
    to_base = contents.encode('utf8').split(b";base64,")[1]
    img = np.fromstring(base64.b64decode(to_base), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

# def save_file(contents, filename):
    # data = content.encode('utf8').split(b";base")

@app.callback(Output('movie_poster', 'children'),
              [Input('upload_image', 'contents')],
              [State('upload_image', 'filename')])
def show_image(image, filename):
    # img_array = cv2.imread(image)
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # reduce_img = cv2.resize(img_array, (256, 256))

    # dat = []
    # dat.append(reduce_img)

    # px.imshow(dat[0])
    children = [
        parse_image(image)
    ]
    return children


@app.callback(Output('classification', 'children'),
              [Input('upload_image', 'contents')],
              [State('upload_image', 'filename')])
def model_output(image, filename):
    img = image_to_cv(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reduce_img = cv2.resize(img, (224, 224))

    dat = []
    dat.append(reduce_img)

    with open('model_architecture.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('weights.hdf5')

    input_dat = np.array(dat[0]).reshape((-1, 224, 224, 3))
    preds = model.predict(input_dat)

    genres = ['Animation', 'Sci-Fi', 'War', 'Western']
    pred = list(chain(*preds))
    pred = max(range(len(pred)), key=lambda i: pred[i])
    children = [
        genres[pred]
    ]
    return children

if __name__ == '__main__':
    app.server.run(threaded=False)
