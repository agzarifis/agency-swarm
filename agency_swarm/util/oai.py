import openai
import instructor


def get_openai_client():
    from clients import build_client_from_user_session
    client = build_client_from_user_session()
    client = instructor.patch(client=client)
    return client


def set_openai_client(new_client):
    global client
    with client_lock:
        client = new_client


def set_openai_key(key):
    if not key:
        raise ValueError("Invalid API key. The API key cannot be empty.")
    openai.api_key = key
