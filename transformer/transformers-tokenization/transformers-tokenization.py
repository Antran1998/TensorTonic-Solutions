class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 4
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token
        
    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        text = " ".join(texts)
        words = text.split(" ")
        words.sort()
        for word in words:
            if self.word_to_id.get(word) is None:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1
        pass

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        text = text.lower().strip()
        if len(text) == 0:
            return []
        text = text.split(' ')
        result = []
        for word in text:
            id = self.word_to_id.get(word) if self.word_to_id.get(word) else self.word_to_id[self.unk_token]  
            result.append(id) 
        return result
        pass

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        result = []
        for id in ids:
            if id < 0 or id > self.vocab_size:
                result.append(self.unk_token)
            else:
                result.append(self.id_to_word[id])
        return " ".join(result)
        pass