import pandas as pd
import ast
import gensim
import gensim.downloader
import logging
import os.path

from variables import ALL_DOCS_FILE, CONFIGS
from hidden_vars import PREPROCESSED_DATA_DIR

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


class LoadSentences:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        try:
            for chunk in pd.read_csv(self.path, usecols=["text"], chunksize=2**10):
                documents = chunk["text"].to_list()
                for document in documents:
                    for sentence in ast.literal_eval(document):
                        yield sentence

        except FileNotFoundError:
            print(f"The path provided for your dataset does not exist: {self.path}")
            exit()
        except ValueError:
            print("The CSV file has no 'text' column.")
            exit()


sentences_dict = {
    config: LoadSentences(os.path.join(PREPROCESSED_DATA_DIR, config, ALL_DOCS_FILE))
    for config in CONFIGS
}

if __name__ == "__main__":
    pre_trained_model = gensim.downloader.load("word2vec-google-news-300")
    pre_trained_model.save(f"models/pre_trained/word2vec-google-news-300.model")

    for config in CONFIGS:
        model = gensim.models.Word2Vec(
            sentences_dict[config], min_count=1, vector_size=200, workers=4
        )
        model.save(f"models/from_scratch/trained_on_{config}.model")
