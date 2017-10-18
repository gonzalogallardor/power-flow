class Error(Exception):
    """ Base class for exceptions in this module """
    def __init__(self, message):
        self.message = message


class DataException(Error):
    pass


class YBusDataTypeError(Error):
    pass


class LoadFlowDataError(Error):
    pass


class LoadFlowIndexError(Error):
    pass


class CLIError(Error):
    pass
