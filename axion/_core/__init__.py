# Suppress Pydantic v2 warnings about validate_default in Field()
import warnings

warnings.filterwarnings(
    'ignore',
    message='.*validate_default.*',
    module='pydantic.*',
)
