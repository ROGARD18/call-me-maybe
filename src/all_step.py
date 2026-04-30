class JSONState():
    def __init__(self):
        self.JSON_START = '[Ċĉ{Ċĉĉ'
        self.JSON_END = 'Ċĉ}Ċ]'
        self.LINE_END = ',Ċĉĉ'
        self.KEY_PROMPT = '"prompt":Ġ"'
        self.KEY_NAME = '"name":Ġ"'
        self.KEY_PARA = '"parameters":Ġ"'