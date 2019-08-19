from keras.models import *
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.normalization import *

net_shape = (512, 512)

def create_model():
    def conv(nb_filters, normalize, border_mode='same'):
        def ret(x):
            y = x
            for nb_filter in nb_filters:
                y = Convolution2D(nb_filter, 3, 3, border_mode=border_mode)(y)
                if normalize: y = BatchNormalization()(y)
                y = LeakyReLU(alpha=0.3)(y)
            return y
        return ret

    input = Input(shape=net_shape+(3,), dtype='float32')
    x1 = ZeroPadding2D((4, 4))(input)
    x1 = conv([16] * 2, False)(x1)
    x2 = conv([16] * 2, True)(x1)

    npooling = 4
    p = x2
    pi = []
    for npool in range(npooling):
        pi.append(p)
        p = MaxPooling2D((2, 2))(p)
        p = conv([24] * 3, True)(p)
    for npool in range(npooling):
        p = UpSampling2D((2, 2))(p)
        p = conv([24] * 1, True)(p)
        p = merge([p, pi[-1-npool]], mode='concat', concat_axis=3)
    x3 = p
    x4 = conv([32] * 3, True, border_mode='valid')(x3)
    output_area = Convolution2D(1, 3, 3, border_mode='valid')(x4)
    output_area = Activation('sigmoid', name='area')(output_area)
    output_edge = Convolution2D(1, 3, 3, border_mode='valid')(x4)
    output_edge = Activation('sigmoid', name='edge')(output_edge)

    model = Model(input=input, output=[output_area, output_edge])
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model


def load_model(model_path):
    pass