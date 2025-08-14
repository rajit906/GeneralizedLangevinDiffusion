# models/utils.py

MODEL_REGISTRY = {}

def register_model(name):
    """
    New models can be added to the registry with the following decorator:
    @register_model('model_name')
    """
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f'Cannot register duplicate model {name}')
        MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls

def get_model(name, **kwargs):
    """
    Get a model from the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Model {name} not in registry')
    return MODEL_REGISTRY[name](**kwargs)