from tqdm.contrib.telegram import tqdm as tqdm_tel
from tqdm import tqdm
import numpy as np

token = "327506757:AAFZFmREFpupJmk5MOjGJMsO5S8vlsvAcLY"
chat_id = "-381821633"



for i in tqdm_tel(range(int(9e8)), token = token, chat_id=chat_id, desc="Progress bar 2 l'espace"):
    pass

