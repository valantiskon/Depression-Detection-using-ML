from pathlib import Path

class EmoticonDetector:
    emoticons = [] # Save a list of emoticons that is saved in the emoticons.txt file

    # Added spaces at the end of each emoji in the txt file so that it does not detect emojis created from expressions, like Boy:because that creates :b emoji
    # Also added a variation of each emoji with | at the end to cover the

    def __init__(self, emoticon_file="dataset\\emoticons.txt"):
        content = Path(emoticon_file).read_text()
        for line in content.split("\n"):
            self.emoticons.append(line)

    # Count the emoticons that a tweet-STRING contains
    def count_emoticons(self, tweet_string):
        count_of_emot = 0
        for emot in self.emoticons:
            count_of_emot += tweet_string.count(emot)

        return count_of_emot