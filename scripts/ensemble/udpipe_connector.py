from connector import Connector

from ufal.udpipe import Model, Pipeline, ProcessingError

class UDPipeConnector(Connector):
    def __init__(self, model, uas, las) -> None:
        super().__init__(model, uas, las)

    