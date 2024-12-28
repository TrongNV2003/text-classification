import string


class TextPreprocess:
    def __init__(self, text: str):
        self.stopwords_file = "stopwords/vietnamese-stopwords.txt"
        self.text = text

    def process_text(self) -> str:
        """
        This function will process the text by removing stopwords, punctuation, and whitespace
        """

        # cleaned_text = self._remove_stopwords(self.text)
        cleaned_text = self._lowercase(self.text)
        cleaned_text = self._remove_punctuation(cleaned_text)
        cleaned_text = self._remove_whitespace(cleaned_text)
        return cleaned_text

    def _remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from the text

        Parameters:
            text: str

        Returns:
            text after removing stopwords
        """

        stopwords = self._load_stopwords(self.stopwords_file)
        words = text.split()
        filtered_words = [word for word in words if word not in stopwords]
        return " ".join(filtered_words)

    def _load_stopwords(self, file_path: str) -> list:
        with open(file_path, "r", encoding="utf-8") as f:
            stopwords = f.read().splitlines()
        return stopwords

    # def _remove_punctuation(self, text):
    #     cleaned_text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    #     return cleaned_text.lower()

    def _remove_punctuation(self, text: str) -> str:
        """
        This function will remove punctuation from the text

        Parameters:
            text: str

        Returns:
            text without punctuation
        """

        punc = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        full_text = ""
        for word in text:
            if word not in punc:
                full_text += word
        return full_text

    def _remove_whitespace(self, text: str) -> str:
        """
        This function will remove whitespace from the text

        Parameters:
            text: str

        Returns:
            text without whitespace
        """

        return " ".join(text.split())

    def _lowercase(self, text: str) -> str:
        """
        This function will convert the text to lowercase

        Parameters:
            text: str

        Returns:
            lower text
        """

        return text.lower()
