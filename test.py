import tensorflow as tf
from tensorflow import keras
import time
import os
config ={
'instance_name' :'Laptop',
'specs' :'4C|16R|0G|0$',
'epoch' : 1,
'ispaperspace':False

}

def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(5000, activation='relu'),
        keras.layers.Dense(4000, activation='relu'),
        keras.layers.Dense(4000, activation='relu'),
        keras.layers.Dense(2000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')    
    ])
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


if __name__ =='__main__':

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    print("Data loaded")

    print('scaling image values between 0-1')
    X_train_scaled = X_train/255
    X_test_scaled = X_test/255

    # one hot encoding labels
    y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'float32')
    y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')

    epoch =config.get('epoch' )


    s = time.time()
    print(" Traning on  GPU")
    with tf.device('/GPU:0'):
        model_gpu = get_model()
        model_gpu.fit(X_train_scaled, y_train_encoded, epochs = epoch)
    e = time.time()
    time_take = (e-s)
    perf =f"Instance name {config.get('instance_name' )}"+ "\n"+f"specs name {config.get('specs' )}" +"\n"+f"time taken  {time_take} "
    path =''
    if not config['ispaperspace']:
        os.mkdir("demo-dataset")
        path=f"demo-dataset/{config.get('instance_name' )}.txt"
    else:
        path=f"/storage/{config.get('instance_name' )}.txt"
    print(perf)
    with open(path ,'w') as f:
        f.write(perf)

    
