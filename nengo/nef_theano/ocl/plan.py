
class Plan(object):

    def __init__(self, dct):
        self.__dict__.update(dct)

        self._fn.set_args(*self._fn_args[3:])
        self._enqueue_args = (
            self.queue,
            self._fn,
            self._fn_args[1],
            self._fn_args[2],
            )

    def __call__(self):
        self._fn(*self._fn_args)
        self._fn_args[0].finish()




