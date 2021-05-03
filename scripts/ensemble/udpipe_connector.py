from connector import Connector

class UDPipeConnector(Connector):
    def __init__(self, model, uas, las) -> None:
        super().__init__(model, uas, las)

    