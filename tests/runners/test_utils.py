import pandas as pd
from pydantic import BaseModel

from axion.runners.utils import models_to_dataframe


class DummyModel(BaseModel):
    id: int
    name: str
    active: bool = True
    score: float = 0.0


def test_models_to_dataframe_basic():
    models = [
        DummyModel(id=1, name='Alice', active=True, score=9.5),
        DummyModel(id=2, name='Bob', active=False, score=8.1),
    ]
    df = models_to_dataframe(models)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['id', 'name', 'active', 'score']
    assert df.shape == (2, 4)
    assert df.iloc[0]['name'] == 'Alice'
    assert not df.iloc[1]['active']


def test_models_to_dataframe_with_exclude_unset():
    models = [
        DummyModel(id=1, name='Alice'),
        DummyModel(id=2, name='Bob'),
    ]
    df = models_to_dataframe(models, exclude_unset=True)

    assert isinstance(df, pd.DataFrame)
    # Default values (like active=True) won't be included
    assert 'active' not in df.columns or all(df['active'].isnull())


def test_models_to_dataframe_empty():
    df = models_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_models_to_dataframe_with_missing_fields():
    class PartialModel(BaseModel):
        id: int
        name: str = None

    models = [
        PartialModel(id=1, name='Alice'),
        PartialModel(id=2),
    ]
    df = models_to_dataframe(models)

    assert 'name' in df.columns
    assert pd.isna(df.iloc[1]['name'])
