class AITraderError(Exception):
    pass


class DataFetchError(AITraderError):
    pass


class InsufficientDataError(AITraderError):
    pass


class DataValidationError(AITraderError):
    pass


class ModelError(AITraderError):
    pass


class ModelNotTrainedError(ModelError):
    pass


class ConfigError(AITraderError):
    pass


class FeatureMismatchError(ModelError):
    pass


class PortfolioError(AITraderError):
    pass
