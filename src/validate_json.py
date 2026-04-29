import json

def validate_json(text: str) ->tuple[bool, str]:
    try:
        json.loads(text)
        return True, "Valid Json"
    except json.JSONDecodeError as e:
        return False, f"Invalid: {e.msg}"
    

if __name__ == "__main__":
    test: str = '{"salut": "ca va", "bonjour": "ca va"}'
    print(validate_json(test))