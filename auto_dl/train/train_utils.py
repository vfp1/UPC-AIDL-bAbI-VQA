from keras.models import Sequential, Model

from architectures.architectures import Arquitectures

class ModelCompiler(object):

    def babl_RNN_compile(self, input_sequence, question, answer):

        # Import the architecture
        c = Arquitectures()



        # Compile the model
        model = Model([input_sequence, question], answer)
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

class Train(object):

    def babl_RNN_train(self, inputs_train, queries_train, answers_train):

        c = ModelCompiler()
        model_RNN = c.babl_RNN_compiler()

        # train
        model.fit([inputs_train, queries_train], answers_train,
                  batch_size=32,
                  epochs=120,
                  validation_data=([inputs_test, queries_test], answers_test))