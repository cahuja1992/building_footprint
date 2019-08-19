from building_footprint.solution_3.model import create_model
from .data_prep import get_val_and_generator
import os 

if __name__ == "__main__":
    nepoch=100

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