import importlib

def test_app_imports():
    # make sure the FastAPI app module is loadable
    mod = importlib.import_module("serving.fastapi_app")
    assert hasattr(mod, "app")
