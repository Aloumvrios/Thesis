from repr_core.repr_FCGSR import FCGSR
from repr_core.repr_obj import REPR


class ReprCore:
    def __init__(self):
        """
        Constructor of representations
        """
        self.available_reprs = (#"FCGSR",
                                # "NGG"
                                )
        self.reprs = {name: None for name in self.available_reprs}
        self.df = None

    def set_dataframe(self, dataframe):
        self.df = dataframe

    def create_reprs(self, kmer):
        for repr_name in self.available_reprs:
            self.create_repr(repr_name)
            self._construct_repr(repr_name, kmer)

    def create_repr(self, repr_name):
        """
        add the repr to the repr catalog dict
        :param repr_name:
        :return:
        """
        if self.reprs[repr_name]:
            print("repr exists")
            return
        if repr_name == "FCGSR":
            print("repr does not exist")
            self.reprs[repr_name] = FCGSR()

    def create_reprs_from_dict(self, repr_dict):
        for repr_name, df in repr_dict.items():
            if repr_name in self.reprs:
                print("repr exists")
                continue
            else:
                print("repr,", repr_name, "does not exist. Let's create it!")
                self.reprs[repr_name] = REPR()
                self.reprs[repr_name].set_repr_name(repr_name)
                self.reprs[repr_name].set_repr_df(df)

    def add_reprs(self,repr_names):
        for repr_name in repr_names:
            if repr_name in self.reprs:
                print("repr exists")
                continue
            else:
                print("repr,", repr_name, "does not exist. Let's create it!")
                self.reprs[repr_name] = REPR()
                self.reprs[repr_name].set_repr_name(repr_name)



    def _construct_repr(self, repr_name, kmer):
        """
        executes the representation functions according to the repr
        :param repr_name:
        :param kmer:
        :return:
        """
        self.reprs[repr_name].construct_repr(self.df, kmer)
