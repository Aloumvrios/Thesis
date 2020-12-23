class REPR:
    """
    This class is used when we load representations from arff files.
    """

    def __init__(self):
        self.name = None
        self.repr_df = None
        self.df_list = None
        self.descriptions = None
        self.scores = None

    def set_repr_df_list(self, df_list):
        self.df_list = df_list

    def set_repr_name(self, name):
        self.name = name

    def set_repr_df(self, df):
        self.repr_df = df
