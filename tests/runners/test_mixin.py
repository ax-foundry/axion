import pandas as pd

from axion._core.metadata.schema import ToolMetadata
from axion.dataset import Dataset, DatasetItem
from axion.runners.mixin import RunnerMixin
from axion.runners.utils import input_to_dataset


class DummyRunner(RunnerMixin):
    name = 'dummy-runner'


def test_get_tool_metadata():
    runner = DummyRunner()
    meta = runner.get_tool_metadata()
    assert isinstance(meta, ToolMetadata)
    assert meta.name == 'dummy-runner'
    assert meta.owner == 'AI Engineering'
    assert meta.version == '1.0.0'


def test_to_queries_from_string_list():
    runner = DummyRunner()
    input_data = ['What is AI?', 'Define ML.']
    queries = runner.to_queries(input_data)
    assert queries == input_data


def test_to_queries_from_dataset():
    runner = DummyRunner()
    dataset = Dataset(
        name='sample',
        items=[
            DatasetItem(id='1', query='Q1', expected_output='A1'),
            DatasetItem(id='2', query='Q2', expected_output='A2'),
        ],
    )
    queries = runner.to_queries(dataset)
    assert queries == ['Q1', 'Q2']


def test_to_queries_from_datasetitem_list():
    runner = DummyRunner()
    dataset_items = [
        DatasetItem(id='1', query='Q1', expected_output='A1'),
        DatasetItem(id='2', query='Q2', expected_output='A2'),
    ]
    queries = runner.to_queries(dataset_items)
    assert queries == ['Q1', 'Q2']


def test_to_queries_from_dataframe():
    runner = DummyRunner()
    df = pd.DataFrame(
        {
            'query': ['What is AI?', 'What is ML?'],
            'expected_output': ['Artificial Intelligence', 'Machine Learning'],
        }
    )
    queries = runner.to_queries(df, dataset_name='df_dataset')
    assert queries == ['What is AI?', 'What is ML?']


def test_to_dataset_from_dataframe():
    _ = DummyRunner()
    df = pd.DataFrame({'query': ['Prompt1'], 'expected_output': ['Answer1']})
    dataset = input_to_dataset(df, name='test')
    assert isinstance(dataset, Dataset)
    assert dataset.name == 'test'
    assert dataset.items[0].query == 'Prompt1'


def test_to_dataset_from_dataset():
    _ = DummyRunner()
    dataset = Dataset(name='test', items=[DatasetItem(query='qq')])
    out = input_to_dataset(dataset)
    assert out == dataset


def test_to_dataset_from_datasetitem_list():
    item_list = [DatasetItem(id='1', query='foo', expected_output='bar')]
    out = input_to_dataset(item_list)
    assert out.items == item_list
