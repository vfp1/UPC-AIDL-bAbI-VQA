import sys, os
sys.path.insert(0, os.path.dirname(sys.path[0]))

from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import os

from utils.parameter_loading import ParameterLoading, GitFolder

class ModelCompiler(object):

    def babl_RNN_compile(self, timestamp, input_sequence, question, answer, save_model_image=True, tensorboard=True):

        """
        This compiles the model from the arquitecture.
        The arquitecture needs to be called before.
        :param input_sequence: the input sequence from the arquitecture
        :param question: the question from the arquitecture
        :param answer: the answer from the arquitecture
        :return: model
        """
        global tboard

        p = ParameterLoading()
        loss, epoch, bsize, optimizer, metrics, logs_dir, plot_images_dir = p.get_hyperparameters()


        # Compile the model
        try:

            model = Model([input_sequence, question], answer)
            model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

        except ValueError:

            raise Exception("Something is not right with the values given. Check the naming of your parameters")

        if save_model_image:

            git_root = GitFolder().get_git_root()

            loss_string = 'LOSS_{}-'.format(loss)
            optimizer_string = 'OPT_{}-'.format(optimizer)
            metrics_string = 'METRICS_{}-'.format(metrics)

            name_string = timestamp + loss_string + optimizer_string + metrics_string

            path_file = os.path.join(git_root, "auto_dl/experiments/network_images/{}.png".format(name_string))

            plot_model(model, to_file=path_file)

        if tensorboard:

            git_root = GitFolder().get_git_root()

            loss_string = 'LOSS_{}-'.format(loss)
            epoch_string = 'EPOCH_{}-'.format(epoch)
            batch_size = 'BSIZE_{}-'.format(bsize)
            optimizer_string = 'OPT_{}-'.format(optimizer)
            metrics_string = 'METRICS_{}-'.format(metrics)

            log_string = timestamp + loss_string + epoch_string + batch_size + optimizer_string + metrics_string

            path_file = os.path.join(git_root, "auto_dl/experiments/tb_logs/{}".format(log_string))

            tboard = TensorBoard(log_dir=path_file, write_graph=True, write_grads=True, batch_size=bsize, write_images=True)

        print(model.summary())

        return model, tboard

class Train(object):

    def babl_RNN_train(self, model, tboard, inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test):
        """
        The train function for the model
        :param model: a compiled model from ModelCompiler.babl_RNN_compile
        :param tboard: the tensorboard path
        :param inputs_train: the train inputs from DataPreprocessing.get_train_test()
        :param queries_train: the train queries inputs from DataPreprocessing.get_train_test()
        :param answers_train: the train answers from DataPreprocessing.get_train_test()
        :param inputs_test: the test inputs from DataPreprocessing.get_train_test()
        :param queries_test: the test queries from DataPreprocessing.get_train_test()
        :param answers_test: the test answers from DataPreprocessing.get_train_test()
        :return: the trained model
        """

        p = ParameterLoading()
        loss, epoch, bsize, optimizer, metrics, logs_dir, plot_images_dir = p.get_hyperparameters()

        # train
        model.fit([inputs_train, queries_train], answers_train,
                  batch_size=bsize,
                  epochs=epoch,
                  validation_data=([inputs_test, queries_test], answers_test),
                  callbacks=[tboard])

        return model