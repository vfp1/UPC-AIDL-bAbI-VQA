import json
import os
import git

class GitFolder(object):
    """
    This class deals with all the path functionalities within the git repo
    """

    def get_git_root(self):
        git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")

        return git_root

class ParameterLoading(object):

    def get_preprocessing_params(self, preprocessing_param_file='preprocessing_parameters.json'):
        """
        Get the challenge type to build up the train test examples
        :param preprocessing_param_file: the preprocessing parameter location, set as default
        :return: the QA1 and QA2
        """

        git_root = GitFolder().get_git_root()

        path_file = os.path.join(git_root, "auto_dl/experiments/parameters/{}".format(preprocessing_param_file))

        try:

            try:

                with open(path_file, "r") as read_file:
                    params = json.load(read_file)

                    QA1 = params["single_supporting_fact_10k"]
                    QA2 = params["two_supporting_facts_10k"]
                    chosen_challenge = params["chosen_challenge"]

                return QA1, QA2, chosen_challenge

            except FileNotFoundError:

                raise Exception("The path to the preprocessing parameter file is not correct")

        except json.decoder.JSONDecodeError:

            raise Exception("Parameter file contains a typo. Probably you forgot a comma somewhere. Exiting execution")

    def get_architecture_params(self, architecture_param_file='architecture_parameters.json'):
        """
        Get the challenge type to build up the train test examples
        :param preprocessing_param_file: the preprocessing parameter location, set as default
        :return: the QA1 and QA2
        """

        git_root = GitFolder().get_git_root()

        path_file = os.path.join(git_root, "auto_dl/experiments/parameters/{}".format(architecture_param_file))

        try:

            try:

                with open(path_file, "r") as read_file:
                    params = json.load(read_file)

                    babl_RNN_dropout = params["babl_RNN"][0]["Dropout"]
                    babl_RNN_activation = params["babl_RNN"][0]["Activation"]

                return babl_RNN_dropout, babl_RNN_activation

            except FileNotFoundError:

                raise Exception("The path to the preprocessing parameter file is not correct")

        except json.decoder.JSONDecodeError:

            raise Exception("Parameter file contains a typo. Probably you forgot a comma somewhere. Exiting execution")


    def get_hyperparameters(self, hyperparams_file='hyperparameters.json'):
        """
        Get the challenge type to build up the train test examples
        :param hyperparams_file: the preprocessing parameter location, set as default
        :return: the QA1 and QA2
        """

        git_root = GitFolder().get_git_root()

        path_file = os.path.join(git_root, "auto_dl/experiments/parameters/{}".format(hyperparams_file))

        try:

            try:

                with open(path_file, "r") as read_file:
                    params = json.load(read_file)

                    loss = params["Loss"]
                    epoch = params["Epoch"]
                    bsize = params["Batch_size"]
                    optimizer = params["Optimizer"]
                    metrics = params["Metrics"]
                    logs_dir = params["Logs_dir"]
                    plot_images_dir = params["Plot_images_dir"]

                return loss, epoch, bsize, optimizer, metrics, logs_dir, plot_images_dir

            except FileNotFoundError:

                raise Exception("The path to the hyperparameter file is not correct")

        except json.decoder.JSONDecodeError:

            raise Exception("Parameter file contains a typo. Probably you forgot a comma somewhere. Exiting execution")