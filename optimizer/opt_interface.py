from abc import ABC, abstractclassmethod

class opt_interface(ABC):
    def create_opt(self, opt_name):
        raise NotImplementedError