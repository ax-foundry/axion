from axion._handlers.utils import camel_to_snake, snake_to_camel


def test_camel_to_snake():
    assert camel_to_snake('HaiThere') == 'hai_there'
    assert camel_to_snake('ThisIsATest') == 'this_is_a_test'
    assert camel_to_snake('camelCase') == 'camel_case'
    assert camel_to_snake('simple') == 'simple'


def test_snake_to_camel_default_sep():
    assert snake_to_camel('hai_there') == 'Hai There'
    assert snake_to_camel('this_is_a_test') == 'This Is A Test'
    assert snake_to_camel('one') == 'One'
    assert snake_to_camel('alreadycamel') == 'Alreadycamel'
    assert snake_to_camel('') == ''


def test_snake_to_camel_custom_sep():
    assert snake_to_camel('hai_there', sep='') == 'HaiThere'
    assert snake_to_camel('test_example_case', sep='-') == 'Test-Example-Case'
    assert snake_to_camel('simple', sep='_') == 'Simple'
