import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from ingestion.data_ingestion import DataPreprocessing
from architectures.architectures import Arquitectures
from train.train_utils import ModelCompiler, Train
import datetime

# Start the time string
time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

# Get the data and preprocess it
print("Preparing data")
vocab_size, story_maxlen, query_maxlen, inputs_train, queries_train, \
answers_train, inputs_test, queries_test, answers_test = DataPreprocessing().get_train_test()

# Get the arquitecture
print("Preparing the arquitecture")
input_sequence, question, answer = Arquitectures().babl_RNN(story_maxlen=story_maxlen,
                                                            query_maxlen=query_maxlen,
                                                            vocab_size=vocab_size)

# Compile the model, save the image model and quickstart tensorboard
print("Computing the model")
model, tboard = ModelCompiler().babl_RNN_compile(timestamp=time_string, input_sequence=input_sequence,
                                                 question=question, answer=answer,
                                                 save_model_image=False,
                                                 tensorboard=True)

# Train the model
print("Training the model")
trained_model = Train().babl_RNN_train(model=model, tboard=tboard, inputs_train=inputs_train,
                                       queries_train=queries_train, answers_train=answers_train,
                                       inputs_test=inputs_test, queries_test=queries_test,
                                       answers_test=answers_test)

