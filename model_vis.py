from model import X_train
from keras.models import Model, model_from_json
from keract import get_activations, display_heatmaps
import matplotlib.pyplot as plt

with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('weights.hdf5')

activations = get_activations(model, X_train[0].reshape([-1, 224, 224, 3]), auto_compile=True)
[print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]

plt.imshow(X_train[0][:, :, 0])
display_heatmaps(activations, X_train[0], save=True)