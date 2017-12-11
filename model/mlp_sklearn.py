from sklearn.neural_network import MLPClassifier

class MlpClassifier_P(MLPClassifier):

    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------------------------------------------------
        # container for evaluation
        # --------------------------------------------------------------------------------------------------------------
        self.accuracy_list = []
        self.average_f1_list = []
        self.label_tp_fp_tn_dict = {}
        # --------------------------------------------------------------------------------------------------------------


        # --------------------------------------------------------------------------------------------------------------
        # container for n-fold validation for different hidden layer, tp is topology, cv is cross-validation
        # --------------------------------------------------------------------------------------------------------------
        self.tp_cv_average_average_f1_list = []
        self.tp_cv_average_accuracy_list = []
        # --------------------------------------------------------------------------------------------------------------

    def set_mlp_clf(self, hidden_layer_sizes, tol=1e-8, learning_rate_init=0.0001, random_state=1,
                           verbose = False, learning_rate = 'constant', early_stopping =False, activation  = 'relu',
                           validation_fraction  = 0.1, alpha  = 0.0001):
        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        self.mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                     tol=tol, learning_rate_init=learning_rate_init,
                                     max_iter=10000, random_state=random_state, verbose=verbose, learning_rate =
                                     learning_rate, early_stopping =early_stopping, alpha= alpha,
                                              validation_fraction = validation_fraction, activation = activation)