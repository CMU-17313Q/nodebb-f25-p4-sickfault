MODEL_NAME = "gemma3:12b"
import os
# Use Ollama library to interact with model:
from ollama import chat, ChatResponse, Client

# Get OLLAMA_HOST, if specified, or default to localhost:11434.
OLLAMA_URL = os.getenv("OLLAMA_HOST", "localhost:11434")

# Initialize the OpenAI client
client = Client(host=OLLAMA_URL)

TRANSLATION_CONTEXT = """\
You are a translation model. Your only task is to translate any non-English input into natural, fluent English.
Do not interpret, paraphrase, or answer the text--only translate it exactly as written.
Respond only with the English translation, and nothing else.

Example:
INPUT: Bonjour, je m'appelle Sarah
OUTPUT: Hello, my name is Sarah

INPUT: Il fait très chaud aujourd'hui, n'est-ce pas ?
OUTPUT: It is very hot today, isn't it?
"""

CLASSIFICATION_CONTEXT = """\
You are a language identifier.
Your task is to detect the primary language of the given text input and respond only with the English name of that language.

Rules:
- Do not translate or explain.
- Do not guess based on previous examples--decide only from the current input.
- If the text appears to be random characters, symbols, or nonsense, respond with "English".

Output format: a single word--the English name of the detected language.

Examples:
INPUT: Bonjour, je m'appelle Sarah
OUTPUT: French

INPUT: Können Sie mir bitte helfen?
OUTPUT: German

INPUT: %#$%#%#%#%#%@#!#!@#
OUTPUT: English"""

def get_translation(post: str) -> str:
    context = TRANSLATION_CONTEXT
    resp = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": context,
            },
            {
                "role": "user",
                "content": post,
            },
        ],
    )
    return resp.message.content.strip()

def get_language(post: str) -> str:
    context = CLASSIFICATION_CONTEXT
    resp = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system", 
                 "content": context,
            },
            {
                "role": "user", 
                "content": post,
            },
        ],
        options={
        'temperature': 0,
        'top_k': 1,
        }
    )
    return resp.message.content.strip()

def translate_content(post: str) -> tuple[bool, str]:
    try:
        lang = get_language(post)
    except Exception:
        return (True, post)

    if not isinstance(lang, str) or not lang.strip():
        return (True, post)

    is_english = lang.strip().lower() == "english"

    try:
        translated = get_translation(post)
    except Exception:
        return (is_english, post)

    if not isinstance(translated, str) or not translated.strip():
        translated = post

    return (is_english, translated)

# def translate_content_fallback(content: str) -> tuple[bool, str]:
#     if content == "这是一条中文消息":
#         return False, "This is a Chinese message"
#     if content == "Ceci est un message en français":
#         return False, "This is a French message"
#     if content == "Esta es un mensaje en español":
#         return False, "This is a Spanish message"
#     if content == "Esta é uma mensagem em português":
#         return False, "This is a Portuguese message"
#     if content  == "これは日本語のメッセージです":
#         return False, "This is a Japanese message"
#     if content == "이것은 한국어 메시지입니다":
#         return False, "This is a Korean message"
#     if content == "Dies ist eine Nachricht auf Deutsch":
#         return False, "This is a German message"
#     if content == "Questo è un messaggio in italiano":
#         return False, "This is an Italian message"
#     if content == "Это сообщение на русском":
#         return False, "This is a Russian message"
#     if content == "هذه رسالة باللغة العربية":
#         return False, "This is an Arabic message"
#     if content == "यह हिंदी में संदेश है":
#         return False, "This is a Hindi message"
#     if content == "นี่คือข้อความภาษาไทย":
#         return False, "This is a Thai message"
#     if content == "Bu bir Türkçe mesajdır":
#         return False, "This is a Turkish message"
#     if content == "Đây là một tin nhắn bằng tiếng Việt":
#         return False, "This is a Vietnamese message"
#     if content == "Esto es un mensaje en catalán":
#         return False, "This is a Catalan message"
#     if content == "This is an English message":
#         return True, "This is an English message"
#     return True, content
