import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense




(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float32')

x_test = x_test.reshape(10000,784).astype('float32')

x_train = x_train / 255.0

x_test = x_test / 255.0

y_train = keras.utils.to_categorical(y_train,10)

y_test = keras.utils.to_categorical(y_test,10)



model = Sequential()
model.add(Dense(256,input_shape=(784,), activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='sigmoid'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

#model.fit(x_train,y_train,epochs=100)
model.fit(x_train, y_train, batch_size=10_000, epochs=100, validation_data=(x_test,y_test))

          
