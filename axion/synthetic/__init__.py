from axion.synthetic.schema import GenerationParams


def __getattr__(name):
    if name == 'DocumentQAGenerator':
        from axion.synthetic.document_generator import DocumentQAGenerator

        return DocumentQAGenerator
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['DocumentQAGenerator', 'GenerationParams']
