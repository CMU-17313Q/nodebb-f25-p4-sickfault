from src.translator import translate_content


def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"

def test_llm_normal_response():
    is_english, translated_content = translate_content("Hello, this is already English.")
    assert is_english is True
    assert translated_content == "Hello, this is already English."

def test_llm_gibberish_response():
    is_english, translated_content = translate_content("$$$###???")
    assert is_english is True
    assert translated_content == "$$$###???"


from src.translator import client
from mock import patch

def _resp(text):
    o = type("O", (), {})()
    o.message = type("M", (), {})()
    o.message.content = text
    return o

@patch.object(client, "chat")
def test_unexpected_language_text(mocker):
    mocker.side_effect = [
        _resp("I don't understand your request"),
        _resp("I don't understand your request"),
    ]
    post = "Hier ist dein erstes Beispiel."
    result = translate_content(post)
    assert result == (False, "I don't understand your request")

@patch.object(client, "chat")
def test_language_is_none(mocker):
    mocker.side_effect = [
        _resp(None),
        _resp("whatever"),
    ]
    post = "Bonjour"
    result = translate_content(post)
    assert result == (True, post)

@patch.object(client, "chat")
def test_empty_translation(mocker):
    mocker.side_effect = [
        _resp("english"),
        _resp(""),
    ]
    post = "Hi"
    result = translate_content(post)
    assert result == (True, post)

@patch.object(client, "chat")
def test_service_down(mocker):
    mocker.side_effect = [
        Exception("down"),
        Exception("down"),
    ]
    post = "Hola"
    result = translate_content(post)
    assert result == (True, post)
