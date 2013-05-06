
class Plan(object):

    def __init__(self, dct):
        self.__dict__.update(dct)

    def __call__(self):
        self._fn(*self._fn_args)
        self._fn_args[0].finish()


