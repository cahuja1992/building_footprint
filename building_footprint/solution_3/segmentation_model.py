from keras.models import *
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.normalization import *
from .data_prep import get_val_and_generator
import os 
import sys
from .data_prep import *
# import keras.optimizers.Adam

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
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def load_model(model_path):
    pass

def train(nepoch=100):

    os.makedirs('model', exist_ok=True)
    save_interval = 1

    model = create_model()
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    validation_data, gen = get_val_and_generator()
    with open('log_keras.csv'.format(), 'w') as logfile: 
        logfile.write("i, loss\n")
        for i in range(1, nepoch + 1):        
            loss = model.fit_generator(gen, validation_data=validation_data, samples_per_epoch=1000, nb_epoch=1, verbose=1, pickle_safe=True)

            logline = "{:>3}, {}\n".format(i, loss.history)
            print(logline, end="")
            logfile.write(logline)

        model.save_weights('model/2_{}.h5'.format(i), overwrite=True)

def pred_proc(arg):
    path = arg[0]
    filenames = arg[1]

    model_file = '2_96'
    model = create_model()
    model.load_weights('/ws/building_footprint/solution_3/model/{}.h5'.format(model_file))

    for fn in filenames:    
        dt1 = datetime.datetime.now()
        area, edge = model.predict(CreateData(path, fn)[np.newaxis, :])
        area = area[0, :, :, 0]
        edge = edge[0, :, :, 0]
        area.dump('area/{}.npz'.format(fn))
        edge.dump('edge/{}.npz'.format(fn))
        dt2 = datetime.datetime.now()
        print(fn, int((dt2-dt1).total_seconds()))


def predict(path, nthread=4):
    all_files = GetFileList(path)
    n = len(all_files)
    filelist_group = [all_files[n*i//nthread:n*(i+1)//nthread] for i in range(nthread)]

    if True:
        with multiprocessing.Pool(nthread) as pool:
            pool.map(pred_proc, [(path, filelist) for filelist in filelist_group])
    else:
        pred_proc((path, all_files))

if __name__ == "__main__":
    command = sys.argv[1]
    if command == 'train':
        train()
    if command == 'test':
        predict(sys.argv[2])