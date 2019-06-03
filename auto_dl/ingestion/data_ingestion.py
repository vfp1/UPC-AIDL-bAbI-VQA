from utils.nlp_utils import get_stories, vectorize_stories
from utils.parameter_loading import ParameterLoading
from keras.utils.data_utils import get_file
import tarfile

class DataAcquisition(object):
    """
    This class deals with all the data acquisition and preparation for
    """

    def get_data(self):
        """
        Gets the data from the bAbl project. Unless changed, usually saves the dataset in
        ./keras/datasets
        :return: a path to the downloaded data
        """

        try:
            path = get_file('babi_tasks_1-20_v1-2.tar.gz.tar.gz',
                            origin='https://s3.amazonaws.com/text-datasets/'
                                   'babi_tasks_1-20_v1-2.tar.gz')
            return path

        except:
            print('Error downloading dataset, please download it manually:\n'
                  '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
                  '.tar.gz\n'
                  '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise

class DataPreprocessing(object):
    """
    This class deals with the whole data preprocessing issues
    """

    def get_train_test(self):
        """
        Generates the train and test sets for the project
        :return: a train and test vector
        """
        # Set challenge variable
        global challenge

        # Download the data
        acquisition = DataAcquisition()
        babi_tasks = acquisition.get_data()

        print("bAbL tasks path", babi_tasks)

        # Build up the challenges
        p = ParameterLoading()
        QA1, QA2, chosen_challenge = p.get_preprocessing_params()

        try:
            if chosen_challenge == "single_supporting_fact_10k":
                challenge = QA1
            elif chosen_challenge == "two_supporting_facts_10k":
                challenge = QA2
        except:

            raise Exception("The chosen challenge should be the same name as one of the possible challenges")

        print('Extracting stories for the challenge:', challenge)

        with tarfile.open(babi_tasks) as tar:
            train_stories = get_stories(tar.extractfile(challenge.format('train')))
            test_stories = get_stories(tar.extractfile(challenge.format('test')))

        vocab = set()
        for story, q, answer in train_stories + test_stories:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        vocab_size = len(vocab) + 1
        story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
        query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

        print('-')
        print('Vocab size:', vocab_size, 'unique words')
        print('Story max length:', story_maxlen, 'words')
        print('Query max length:', query_maxlen, 'words')
        print('Number of training stories:', len(train_stories))
        print('Number of test stories:', len(test_stories))
        print('-')
        print('Here\'s what a "story" tuple looks like (input, query, answer):')
        print(train_stories[0])
        print('-')
        print('Vectorizing the word sequences...')

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

        inputs_train, queries_train, answers_train = vectorize_stories(data=train_stories, word_idx=word_idx,
                                                                       story_maxlen=story_maxlen, query_maxlen=query_maxlen)
        inputs_test, queries_test, answers_test = vectorize_stories(data=test_stories, word_idx=word_idx,
                                                                       story_maxlen=story_maxlen, query_maxlen=query_maxlen)

        print('-')
        print('inputs: integer tensor of shape (samples, max_length)')
        print('inputs_train shape:', inputs_train.shape)
        print('inputs_test shape:', inputs_test.shape)
        print('-')
        print('queries: integer tensor of shape (samples, max_length)')
        print('queries_train shape:', queries_train.shape)
        print('queries_test shape:', queries_test.shape)
        print('-')
        print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
        print('answers_train shape:', answers_train.shape)
        print('answers_test shape:', answers_test.shape)
        print('-')
        print('Compiling...')

        return vocab_size, story_maxlen, query_maxlen, inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test