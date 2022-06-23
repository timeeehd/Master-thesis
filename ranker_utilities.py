import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, TrainingArguments, Trainer
import numpy as np
from torch.nn import KLDivLoss, Softmax, LogSoftmax

import random
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
import os
from tqdm import tqdm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
STOP_WORDS = set(get_stop_words('english'))

import nltk
from nltk.stem import WordNetLemmatizer
# Find synonyms, nltk.download('wordnet')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
nltk.download('omw-1.4')

from nltk import pos_tag

nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn


def split_words(text):
    return re.split(r'[ ](?=[\w])', text)


def split_to_sentences(text):
    """
    Args:
        text (str): raw text to split to sentences on end of sentences marks.
    Returns:
        List of sentences from text.
    """
    # Multiple options Negative lookbehind to prevent splitting on Mr. or Mrs.
    split_pattern = r'(?<!Mr)(?<!Mrs)[.!?;"]+'
    # Split on end of sentence, but keep the punctuation marks.
    sentences = list(map(str.strip, re.sub(
        split_pattern, r'\g<0>[cut]', text.strip()).split('[cut]')))
    # If the last sentence is ''
    if len(sentences) > 1 and len(sentences[-1]) < 3:
        sentences.pop()
    return sentences


def _readabilty(text, texts_sentences):
    """
    Uses length of sentences and length of words.
    Higher is for more advanced readers.
    If text is sparse i.e. mostly new lines, and doesn't end with an eos -> add a negative cost.
    Args:
        text (str): original text to return score for.
        texts_sentences (list): text split to sentences.
    """
    txt_words = split_words(text)
    num_letters = sum(len(word) for word in txt_words)
    num_words = len(txt_words)
    num_sent = len(texts_sentences)

    # check if a "sparse" sentence
    if num_sent == 1:
        new_line_threshold = 0 if num_words == 0 else num_words // 4
        if texts_sentences[0].count('\n') > new_line_threshold or not re.search(r'(?<![A-Z])[.!?;"]+', texts_sentences[0]):
            num_sent = 0

    letters_per_word = -10 if num_words == 0 else num_letters/num_words
    words_per_sentence = -10 if num_sent == 0 else num_words/num_sent
    # 0.5 to weight words_per_sentence higher
    return 0.5*letters_per_word + words_per_sentence


def _sentiment_polarity(filtered_words):
    """
    Returns a positive sentiment polarity in range 0 = negative/objective to 100 = positive of the entire text.
    Uses SentiWordnet to compute the positiveness polarity of the words and average that value.
    Based on https://nlpforhackers.io/sentiment-analysis-intro/

    Args:
        filtered_words (set): set of non-stop, non-punctuation words.
    """
    # If  empty
    if len(filtered_words) < 2:
        return 0

    POS_TAG_TO_WN = {'J': wn.ADJ, 'N': wn.NOUN, 'R': wn.ADV, 'V': wn.VERB}
    lemmatizer = WordNetLemmatizer()

    text_sentiment = []
    tagged_words = pos_tag(filtered_words)

    for word, tag in tagged_words:
        wn_tag = POS_TAG_TO_WN.get(tag[0], None)
        if not wn_tag:
            continue

        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue

        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue

        # Take the most common of the synthsets
        synset = synsets[0]
        pos_sentiment = swn.senti_synset(synset.name()).pos_score(
        ) - swn.senti_synset(synset.name()).neg_score()
        text_sentiment.append(pos_sentiment)
    return 0 if not text_sentiment else max(int(np.mean(text_sentiment)*100), 0)

def _simplicity(filtered_words_set):
    """
    Fraction of most frequent words from generated text.
    Args:
        filtered_words_set (set): set of non-stop, non-punctuation words.
    """
    SEVEN_PREC_MOST_FREQ_FAIRY_WORDS = {'six', 'month', 'drew', 'want', 'hands', 'staring', 'guests', 'goose', 'fitted', 'rope', 'grace', 'delightful', 'meg', 'peace', 'lovely', 'iron', 'dark', 'cloak', 'pictures', 'eaten', 'sake', 'hurt', 'soldiers', 'dragon', 'late', 'unusual', 'centre', 'shore', 'gloomy', 'burning', 'time', 'foreign', 'bride', 'show', 'disappeared', 'light', 'spirits', 'arose', 'larger', 'sunshine', 'paul', 'cries', 'nearest', 'refuse', 'cut', 'fun', 'naughty', 'ears', 'remember', 'filled', 'playing', 'ask', 'loud', 'suggested', 'husband', 'placed', 'proud', 'places', 'difficult', 'somebody', 'eat', 'fault', 'school', 'honor', 'maybe', 'faith', 'win', 'full', 'missed', 'big', 'pieces', 'asking', 'case', 'unto', 'wolf', 'jennings', 'color', 'riding', 'elder', 'torn', 'stay', 'monkeys', 'comes', 'largest', 'crew', 'receive', 'feast', 'vast', 'foot', 'office', 'cake', 'throughout', 'indians', 'trust', 'cheerful', 'building', 'star', 'apple', 'younger', 'henry', 'matters', 'surface', 'york', 'summer', 'nothing', 'mentioned', 'come', 'hope', 'certain', 'seized', 'folded', 'jack', 'absolutely', 'surrounded', 'noise', 'hardly', 'entrance', 'hold', 'pass', 'subject', 'three', 'coming', 'rolling', 'mouse', 'sense', 'alarmed', 'influence', 'day', 'beneath', 'ceased', 'minchin', 'wood', 'laughter', 'difficulty', 'merry', 'marionette', 'state', 'dangerous', 'fought', 'lesson', 'ones', 'upper', 'house', 'mighty', 'ever', 'fountain', 'path', 'loss', 'help', 'present', 'tore', 'beat', 'princes', 'company', 'knees', 'wilt', 'charles', 'finished', 'blame', 'doors', 'replied', 'wait', 'cruel', 'glass', 'chest', 'extraordinary', 'mad', 'explain', 'eldest', 'sank', 'shed', 'driven', 'fall', 'prove', 'rising', 'angry', 'pulling', 'field', 'grandmother', 'mysterious', 'tea', 'sought', 'natural', 'raised', 'learned', 'must', 'hook', 'cared', 'gained', 'monster', 'beth', 'nearly', 'gazing', 'heap', 'rabbit', 'monsieur', 'stars', 'said', 'soft', 'jolly', 'stiff', 'exceedingly', 'us', 'named', 'smoke', 'colia', 'hurried', 'discover', 'close', 'removed', 'explained', 'crept', 'cutting', 'sight', 'mere', 'leaf', 'world', 'witch', 'putting', 'awful', 'grandfather', 'daughters', 'mountain', 'amid', 'understood', 'whispered', 'anger', 'hole', 'hunting', 'pretended', 'evening', 'shouted', 'sword', 'animal', 'swam', 'instant', 'madame', 'chariot', 'fairy', 'eager', 'master', 'raise', 'hercules', 'splendid', 'good', 'pocket', 'betty', 'bowed', 'thanks', 'regular', 'sorry', 'showed', 'dish', 'actually', 'place', 'sperm', 'water', 'aglaya', 'needs', 'pink', 'tumbled', 'ancient', 'rosy', 'sadly', 'else', 'lighted', 'sort', 'wall', 'dreamed', 'warm', 'bid', 'elinor', 'ice', 'week', 'supper', 'wounded', 'sell', 'refused', 'cook', 'otherwise', 'shoulder', 'glanced', 'dropped', 'stand', 'pretend', 'friends', 'skin', 'keep', 'call', 'beard', 'drawn', 'drawing', 'lying', 'move', 'married', 'send', 'edward', 'possibly', 'horrid', 'think', 'courage', 'great', 'work', 'gently', 'formed', 'society', 'rather', 'amy', 'branches', 'hopes', 'precious', 'depths', 'names', 'anxious', 'saw', 'cow', 'blue', 'forgot', 'talking', 'road', 'lose', 'rajah', 'attack', 'follow', 'third', 'leaves', 'waves', 'hang', 'jo', 'fancied', 'treasure', 'head', 'whatever', 'devil', 'fully', 'buried', 'wet', 'gun', 'looked', 'brought', 'next', 'farther', 'decided', 'band', 'cheeks', 'steal', 'sent', 'tossed', 'hear', 'maiden', 'servant', 'ride', 'tiny', 'feared', 'wave', 'travelled', 'one', 'green', 'teach', 'fro', 'tell', 'cool', 'cunning', 'catch', 'merely', 'badger', 'stepped', 'shining', 'owner', 'observe', 'chanced', 'bottom', 'hero', 'heavy', 'land', 'grateful', 'strange', 'circumstances', 'dwarf', 'quick', 'miserable', 'note', 'ship', 'fallen', 'word', 'john', 'turned', 'inclined', 'rat', 'easy', 'legs', 'bring', 'empty', 'touching', 'drinking', 'bade', 'army', 'sure', 'fierce', 'presented', 'key', 'tender', 'nobody', 'directly', 'build', 'edge', 'hidden', 'experience', 'begun', 'attempt', 'miles', 'violent', 'running', 'firmly', 'kicked', 'lived', 'party', 'chamber', 'savage', 'usually', 'drove', 'concluded', 'arranged', 'baby', 'pick', 'fail', 'england', 'reading', 'wants', 'generally', 'red', 'working', 'creatures', 'flew', 'keen', 'person', 'toad', 'related', 'twelve', 'aunt', 'confidence', 'moved', 'took', 'horrible', 'leaning', 'confess', 'spring', 'standing', 'vessel', 'locked', 'scarecrow', 'drank', 'reckon', 'happen', 'received', 'group', 'lebedeff', 'sun', 'sons', 'welcome', 'youngest', 'solid', 'impossible', 'princess', 'sky', 'learning', 'tongue', 'silly', 'began', 'fell', 'following', 'upstairs', 'moon', 'twice', 'given', 'everywhere', 'together', 'bell', 'heat', 'rough', 'gentle', 'carefully', 'longer', 'months', 'none', 'delicate', 'rest', 'darted', 'say', 'alone', 'former', 'everyone', 'rested', 'paid', 'devoted', 'understand', 'nodded', 'winter', 'animals', 'o', 'camp', 'consider', 'demanded', 'odd', 'pull', 'read', 'sold', 'bite', 'innocent', 'waters', 'heard', 'rode', 'stretched', 'gay', 'peculiar', 'position', 'done', 'shoes', 'express', 'streets', 'believe', 'later', 'fact', 'sea', 'seemed', 'drops', 'heaven', 'sail', 'hunt', 'grant', 'hans', 'grave', 'private', 'pipe', 'clock', 'put', 'taught', 'travel', 'beginning', 'slightly', 'view', 'slowly', 'blew', 'kept', 'plenty', 'bought', 'mind', 'belong', 'slow', 'eggs', 'special', 'happily', 'yellow', 'two', 'due', 'clear', 'handsome', 'deeply', 'short', 'cast', 'belonged', 'bore', 'enter', 'spread', 'shut', 'fill', 'closely', 'excited', 'capable', 'arm', 'obtained', 'seven', 'earth', 'truly', 'silence', 'matter', 'box', 'prisoner', 'gates', 'approaching', 'situation', 'mamma', 'gathered', 'mile', 'lot', 'way', 'clouds', 'song', 'without', 'distant', 'shoulders', 'kiss', 'cottage', 'armed', 'within', 'trying', 'smile', 'got', 'gate', 'eagle', 'neither', 'stands', 'somewhat', 'walking', 'asked', 'stubb', 'agreed', 'forgive', 'charge', 'bad', 'sprang', 'quite', 'washed', 'tear', 'retired', 'river', 'except', 'scarcely', 'landed', 'object', 'attracted', 'cock', 'rid', 'row', 'weight', 'watch', 'many', 'solitary', 'table', 'deeper', 'whether', 'swear', 'nearer', 'touched', 'colin', 'pleased', 'dreadful', 'false', 'ozma', 'dig', 'dorothy', 'future', 'cost', 'possible', 'anybody', 'offered', 'thy', 'sitting', 'line', 'darkness', 'kissed', 'summoned', 'laying', 'led', 'spell', 'considered', 'uncle', 'accustomed', 'marriage', 'whenever', 'pay', 'died', 'treated', 'hate', 'join', 'amuse', 'cried', 'finest', 'walk', 'things', 'won', 'hastened', 'leaped', 'emerald', 'swimming', 'affection', 'pounds', 'guard', 'swung', 'desert', 'bright', 'cure', 'laughing', 'piece', 'faster', 'hair', 'older', 'indeed', 'remain', 'reason', 'various', 'stuff', 'waving', 'comfortable', 'rise', 'settled', 'fool', 'friend', 'earnestly', 'island', 'comfort', 'beloved', 'kind', 'stayed', 'tin', 'happiness', 'ground', 'liked', 'sometimes', 'ivan', 'personal', 'longed', 'thing', 'meeting', 'simple', 'always', 'hunter', 'roof', 'obey', 'hot', 'delighted', 'moment', 'struck', 'cause', 'chance', 'whole', 'discovered', 'roll', 'seat', 'kitchen', 'bottle', 'taking', 'dickon', 'important', 'young', 'forget', 'bearing', 'safe', 'soon', 'deep', 'turning', 'holy', 'pinocchio', 'repeated', 'lady', 'papa', 'ways', 'crown', 'right', 'thou', 'body', 'dog', 'somewhere', 'steps', 'clearly', 'wedding', 'grey', 'cloth', 'mine', 'character', 'kindly', 'fish', 'silk', 'amongst', 'apparently', 'thinks', 'horse', 'latter', 'dim', 'wear', 'found', 'hurry', 'distance', 'perceived', 'apples', 'appeared', 'creature', 'form', 'tired', 'acquaintance', 'pale', 'remained', 'corner', 'moving', 'five', 'bless', 'queen', 'reward', 'defarge', 'kingdom', 'hit', 'governor', 'rubbed', 'stream', 'danced', 'host', 'bull', 'pressed', 'picture', 'escaped', 'crow', 'hall', 'wishing', 'persuaded', 'turkey', 'mortal', 'difference', 'laughed', 'thank', 'hand', 'front', 'dost', 'study', 'fly', 'nest', 'inquired', 'hours', 'burned', 'fetch', 'city', 'just', 'mischief', 'gradually', 'hated', 'claim', 'telling', 'yonder', 'faint', 'papers', 'meet', 'quickly', 'smith', 'plain', 'thrust', 'ran', 'entire', 'unhappy', 'lay', 'boat', 'roared', 'forty', 'wise', 'excellent', 'listened', 'vanished', 'climbed', 'manner', 'unless', 'parents', 'gift', 'rocks', 'force', 'low', 'garden', 'door', 'much', 'gray', 'greatest', 'wretched', 'year', 'angel', 'returned', 'stuck', 'waited', 'little', 'held', 'interested', 'fiery', 'purpose', 'gazed', 'woke', 'especially', 'recovered', 'useful', 'terribly', 'sad', 'wherever', 'might', 'carriage', 'maid', 'around', 'left', 'partly', 'sensible', 'ashamed', 'ordered', 'hollow', 'last', 'value', 'may', 'notice', 'troubles', 'mountains', 'wondered', 'weather', 'night', 'remarkable', 'slip', 'grow', 'floor', 'man', 'divided', 'wizard', 'highly', 'marched', 'rose', 'climb', 'fox', 'lucy', 'becoming', 'rushing', 'however', 'acted', 'number', 'meaning', 'orders', 'means', 'step', 'mention', 'tie', 'flowers', 'forced', 'promise', 'hoped', 'let', 'lie', 'poor', 'god', 'bundle', 'sacred', 'meantime', 'either', 'evil', 'white', 'game', 'return', 'invited', 'examined', 'swallowed', 'brave', 'shone', 'doctor', 'pointed', 'leg', 'deal', 'excuse', 'greater', 'yard', 'knock', 'accepted', 'share', 'wearing', 'proved', 'leading', 'still', 'books', 'seek', 'dying', 'higher', 'lion', 'flock', 'cross', 'favorite', 'laugh', 'see', 'houses', 'painful', 'stronger', 'adventure', 'parts', 'sat', 'bread', 'appear', 'sudden', 'proper', 'reach', 'worked', 'turns', 'memory', 'among', 'appearance', 'men', 'give', 'listen', 'heels', 'keeping', 'tsar', 'better', 'lead', 'point', 'best', 'guess', 'live', 'lift', 'leaving', 'write', 'shape', 'lifted', 'intention', 'blowing', 'stories', 'woman', 'windows', 'needed', 'bird', 'thither', 'cover', 'tears', 'allowed', 'wound', 'singing', 'lower', 'suppose', 'rain', 'room', 'hour', 'trouble', 'threw', 'tells', 'tall', 'top', 'sweet', 'fighting', 'kill', 'already', 'jason', 'tree', 'united', 'behind', 'shown', 'almost', 'danger', 'native', 'delicious', 'basket', 'dermat', 'leave', 'utter', 'pleasant', 'everything', 'wrote', 'knight', 'becky', 'vain', 'believed', 'bill', 'peter', 'music', 'ka', 'address', 'prepared', 'stupid', 'perfectly', 'admit', 'played',
                              'beside', 'finally', 'wept', 'familiar', 'suffered', 'everybody', 'similar', 'though', 'fled', 'passing', 'law', 'imagine', 'succeeded', 'alive', 'usual', 'gives', 'battle', 'thoroughly', 'rooms', 'gone', 'followed', 'carrying', 'effort', 'rich', 'bent', 'hastily', 'prison', 'square', 'hid', 'along', 'eight', 'eagerly', 'fond', 'since', 'dared', 'trembling', 'bare', 'smooth', 'away', 'loaded', 'south', 'thought', 'boots', 'black', 'busy', 'stone', 'van', 'presence', 'sounded', 'bound', 'pleasure', 'idea', 'wooden', 'death', 'ordinary', 'willing', 'original', 'arms', 'added', 'spare', 'losing', 'told', 'money', 'fair', 'quarter', 'words', 'hannah', 'voice', 'knife', 'getting', 'peeped', 'free', 'sees', 'street', 'dug', 'monkey', 'frightened', 'grown', 'dogs', 'weary', 'fellow', 'pulled', 'finger', 'dwelt', 'resolved', 'woods', 'pure', 'according', 'tail', 'loving', 'drive', 'habit', 'magician', 'thinking', 'admitted', 'jumped', 'joy', 'play', 'kinds', 'curiosity', 'returning', 'towards', 'size', 'lucky', 'major', 'sailing', 'interest', 'name', 'days', 'declared', 'spoken', 'plan', 'palace', 'castle', 'trembled', 'court', 'chair', 'equal', 'ahab', 'known', 'announced', 'passage', 'old', 'run', 'hoping', 'rush', 'lofty', 'talk', 'powerful', 'truth', 'george', 'attend', 'fat', 'fate', 'clothes', 'news', 'thief', 'taken', 'new', 'queer', 'circle', 'space', 'strength', 'lives', 'hanging', 'something', 'need', 'feet', 'awoke', 'whoever', 'waiting', 'sir', 'rage', 'temple', 'air', 'bench', 'early', 'stick', 'fear', 'sorrow', 'feel', 'soul', 'stout', 'shoot', 'nurse', 'convinced', 'thanksgiving', 'daughter', 'king', 'service', 'thousand', 'opening', 'shaking', 'flower', 'opportunity', 'minutes', 'ago', 'real', 'probably', 'gets', 'near', 'simply', 'ended', 'smiling', 'someone', 'sister', 'sum', 'whilst', 'enough', 'crying', 'writing', 'heart', 'silent', 'addressed', 'marianne', 'occasion', 'pity', 'forehead', 'west', 'u', 'somehow', 'marble', 'mark', 'faithful', 'mole', 'dream', 'bow', 'slight', 'brass', 'find', 'anything', 'boys', 'fingers', 'weak', 'wanted', 'caused', 'across', 'fed', 'side', 'outside', 'persons', 'grief', 'bushes', 'excitement', 'softly', 'charming', 'paused', 'eyes', 'foolish', 'oak', 'even', 'interrupted', 'finding', 'forest', 'buy', 'johnny', 'ten', 'lips', 'back', 'smiled', 'nights', 'muttered', 'walls', 'fifteen', 'hast', 'shame', 'thick', 'enemy', 'gave', 'thrown', 'task', 'knowledge', 'cold', 'certainly', 'like', 'shadows', 'enjoy', 'wine', 'pig', 'far', 'darling', 'fellows', 'pursued', 'snake', 'wendy', 'turn', 'looking', 'worthy', 'slightest', 'toto', 'striking', 'slept', 'throne', 'forward', 'hearts', 'slipped', 'figure', 'shelter', 'direction', 'attention', 'question', 'handed', 'treat', 'beg', 'runs', 'candle', 'fourth', 'interesting', 'bag', 'huge', 'hut', 'worse', 'knocked', 'years', 'magnificent', 'different', 'spend', 'whose', 'sound', 'miss', 'lit', 'small', 'lad', 'burn', 'wish', 'general', 'necessary', 'fields', 'clever', 'beasts', 'serious', 'sleeping', 'fixed', 'please', 'use', 'murmured', 'saint', 'went', 'throw', 'commanded', 'four', 'fresh', 'ball', 'giving', 'flung', 'arrived', 'sighed', 'carried', 'french', 'arrival', 'afraid', 'burnt', 'hung', 'exactly', 'cloud', 'face', 'calm', 'perfect', 'mean', 'er', 'robbers', 'strangely', 'rushed', 'hard', 'another', 'merchant', 'girls', 'continually', 'lizabetha', 'sack', 'milk', 'lord', 'common', 'window', 'fur', 'round', 'honour', 'ass', 'fight', 'human', 'recognized', 'nine', 'makes', 'dressed', 'suddenly', 'waved', 'double', 'avoid', 'seems', 'evgenie', 'morning', 'string', 'impression', 'conversation', 'glancing', 'middle', 'eleven', 'brother', 'seen', 'neck', 'search', 'weeks', 'started', 'stood', 'sides', 'dare', 'painted', 'miller', 'art', 'knowing', 'wash', 'midnight', 'became', 'fire', 'agreeable', 'soldier', 'blow', 'family', 'minute', 'coat', 'stop', 'beast', 'joe', 'true', 'others', 'robin', 'able', 'walked', 'shook', 'part', 'hat', 'wishes', 'peasant', 'people', 'ugly', 'jungle', 'surprise', 'evidently', 'board', 'closed', 'sing', 'loved', 'sang', 'rolled', 'crowd', 'serve', 'moments', 'immense', 'happy', 'anyone', 'stole', 'therefore', 'sheep', 'often', 'bank', 'hundred', 'meat', 'possessed', 'chase', 'remembered', 'pack', 'horror', 'stranger', 'heavily', 'wicked', 'mother', 'voyage', 'glance', 'dozen', 'visit', 'opinion', 'constantly', 'deck', 'single', 'holding', 'whales', 'enchanted', 'care', 'gods', 'considerable', 'food', 'beating', 'silver', 'colonel', 'feed', 'loves', 'become', 'faces', 'events', 'worst', 'using', 'rare', 'pine', 'grew', 'dinner', 'mud', 'driving', 'oil', 'purple', 'suffer', 'letter', 'narrow', 'sounds', 'drop', 'wished', 'sick', 'blood', 'mice', 'tiger', 'companion', 'every', 'noble', 'escape', 'lake', 'make', 'approached', 'aside', 'remarked', 'secret', 'sooner', 'ready', 'felt', 'tower', 'longing', 'spoke', 'bear', 'fifty', 'growing', 'visited', 'long', 'christmas', 'several', 'past', 'stones', 'greatly', 'born', 'frog', 'obliged', 'ere', 'joined', 'business', 'fine', 'dust', 'folks', 'beautiful', 'worn', 'holmes', 'wake', 'touch', 'twenty', 'glad', 'grand', 'enjoyed', 'satisfied', 'finds', 'valuable', 'less', 'uttered', 'straight', 'respect', 'mowgli', 'points', 'wind', 'thus', 'life', 'can', 'golden', 'branch', 'ships', 'afternoon', 'wondering', 'know', 'steady', 'change', 'hare', 'fancy', 'rapidly', 'proceeded', 'nose', 'helped', 'killed', 'expected', 'train', 'delight', 'de', 'feathers', 'tom', 'flying', 'midst', 'showing', 'observed', 'changed', 'chosen', 'dear', 'enormous', 'setting', 'lines', 'likewise', 'lamp', 'seldom', 'condition', 'listening', 'saying', 'equally', 'speak', 'cup', 'quietly', 'failed', 'came', 'surprised', 'captain', 'comrade', 'fairly', 'prevent', 'expression', 'continued', 'reached', 'gold', 'winged', 'thee', 'entirely', 'youth', 'waste', 'laurie', 'companions', 'picked', 'thin', 'pushed', 'finish', 'rock', 'stopped', 'seated', 'occurred', 'nastasia', 'drink', 'children', 'hind', 'broke', 'voices', 'covered', 'history', 'seeing', 'seem', 'paper', 'wife', 'half', 'breath', 'fairies', 'eating', 'flat', 'fastened', 'increased', 'exclaimed', 'possession', 'mary', 'hungry', 'assured', 'frequently', 'watched', 'lights', 'hearty', 'anywhere', 'sleep', 'bones', 'daily', 'son', 'trunk', 'promised', 'answer', 'marry', 'fit', 'first', 'leaned', 'straw', 'safely', 'luck', 'making', 'nice', 'handkerchief', 'take', 'epanchin', 'silently', 'breaking', 'thirty', 'open', 'huntsman', 'boy', 'swift', 'stared', 'nature', 'lorry', 'poured', 'tied', 'intended', 'delivered', 'wide', 'town', 'public', 'rang', 'mounted', 'horses', 'lies', 'concealed', 'made', 'disturbed', 'chief', 'speaking', 'north', 'spent', 'instantly', 'girl', 'goes', 'snow', 'main', 'expressed', 'tried', 'birds', 'completely', 'loudly', 'loose', 'bed', 'ate', 'called', 'blessed', 'begged', 'home', 'couple', 'shadow', 'bigger', 'queequeg', 'pitcher', 'bits', 'content', 'says', 'desire', 'sisters', 'mount', 'falling', 'dress', 'glimpse', 'march', 'anxiety', 'meal', 'command', 'kindness', 'dirty', 'war', 'beauty', 'sailed', 'worth', 'careful', 'accept', 'allow', 'complete', 'idle', 'travelling', 'ye', 'going', 'assure', 'suffering', 'spirit', 'woodman', 'particularly', 'harm', 'village', 'ear', 'pretty', 'image', 'wandering', 'honest', 'letters', 'stopping', 'cave', 'seeking', 'dead', 'set', 'tale', 'hunger', 'high', 'awake', 'sending', 'london', 'brothers', 'broad', 'dancing', 'farmer', 'smell', 'peaceful', 'breakfast', 'calls', 'also', 'gentleman', 'questions', 'inside', 'stolen', 'count', 'famous', 'regard', 'broken', 'speech', 'end', 'plunged', 'unable', 'gania', 'wonderful', 'shake', 'second', 'large', 'now', 'gladly', 'pair', 'clean', 'mistress', 'gather', 'watching', 'rogojin', 'pan', 'afterwards', 'deadly', 'supposed', 'signs', 'eye', 'funny', 'unfortunate', 'choose', 'troubled', 'bold', 'altogether', 'used', 'shall', 'trees', 'sorts', 'flight', 'twisted', 'crossed', 'guessed', 'forth', 'fast', 'hearing', 'owl', 'brown', 'pointing', 'well', 'fetched', 'throwing', 'love', 'screamed', 'haste', 'sara', 'least', 'opened', 'race', 'takes', 'dashwood', 'sharp', 'prince', 'taste', 'served', 'will', 'satisfaction', 'beyond', 'giant', 'wandered', 'shot', 'sign', 'household', 'capital', 'perhaps', 'meant', 'corn', 'duty', 'advice', 'informed', 'managed', 'regarded', 'spot', 'servants', 'book', 'ladies', 'dashed', 'mouth', 'yesterday', 'scene', 'floating', 'axe', 'offer', 'swiftly', 'feelings', 'power', 'hill', 'look', 'age', 'gentlemen', 'thoughts', 'lively', 'never', 'stairs', 'go', 'roses', 'fortune', 'knows', 'doubt', 'living', 'scattered', 'reply', 'draw', 'immediately', 'considering', 'engaged', 'sunday', 'brilliant', 'cat', 'learn', 'suspected', 'met', 'quiet', 'women', 'dry', 'majesty', 'forgotten', 'tailor', 'looks', 'saved', 'startled', 'pray', 'father', 'friendly', 'conduct', 'astonished', 'lump', 'passed', 'begin', 'manners', 'caught', 'child', 'upon', 'plainly', 'widow', 'provided', 'boats', 'starting', 'pot', 'manage', 'likely', 'besides', 'dance', 'solemn', 'creeping', 'curious', 'opposite', 'account', 'particular', 'written', 'burst', 'ring', 'required', 'till', 'start', 'ill', 'journey', 'settle', 'answered', 'unknown', 'yards', 'learnt', 'asleep', 'jump', 'wonder', 'strong', 'dollars', 'beheld', 'breast', 'story', 'terror', 'dull', 'expect', 'easily', 'presently', 'marked', 'swept', 'try', 'judge', 'bringing', 'carry', 'hills', 'lonely', 'american', 'happened', 'sit', 'country', 'length', 'wrong', 'wings', 'valley', 'aware', 'times', 'spite', 'lest', 'instead', 'oh', 'storm', 'feeling', 'royal', 'arrow', 'terrible', 'naturally', 'wore', 'intend', 'hide', 'whence', 'east', 'cap', 'yet', 'toward', 'floated', 'noticed', 'course', 'whale', 'pain', 'pride', 'canst', 'desired', 'surely', 'dragged', 'aid', 'magic', 'determined', 'lost', 'dried', 'entered', 'save', 'grass', 'oz', 'stir', 'break', 'really', 'blind', 'act', 'built', 'keeps', 'produced', 'although', 'tone', 'suit', 'order', 'emperor', 'lodge', 'bit', 'calling', 'bitter', 'cry', 'sympathy', 'extremely', 'kid', 'laid', 'heads', 'giants', 'wrapped', 'teeth', 'get', 'wild', 'talked', 'die', 'effect', 'church', 'strike', 'fruit', 'thanked', 'knew', 'respectable'}
    SEVEN_PREC_MOST_FREQ_FANT_WORDS = {'said', 'Sir', 'will', 'one', 'now', 'thou', 'King', 'upon', 'came', 'man', 'knight', 'great', 'come', 'well', 'day', 'thee', 'little', 'time', 'men', 'went', 'ye', 'hand', 'yet', 'may', 'good', 'know', 'see', 'way', 'go', 'us', 'made', 'might', 'two', 'thing', 'saw', 'never', 'say', 'thy', 'many', 'long', 'back', 'God', 'must', 'night', 'lord', 'much', 'tell', 'even', 'face', 'away', 'lady', 'place', 'old', 'love', 'though', 'head', 'horse', 'word', 'eye', 'make', 'unto', 'heart', "n't", 'thought', 'let', 'take', 'first', 'house', 'took', 'Tristram', 'looked', 'world', 'last', 'Queen', 'Arthur', 'fair', 'three', 'Launcelot', 'think', 'life', 'without', 'stood', 'heard', 'found', 'knew', 'side', 'sword', 'still', 'land', 'told', 'thus', 'set', 'toward', 'right', 'son', 'done', 'people', 'together', 'arm', 'woman', 'year', 'look', 'give', 'turned', 'left', 'friend', 'another', 'wood', 'name', 'far', 'young', 'seemed', 'fell', 'voice', 'castle', 'matter', 'put', 'water', 'nothing', 'till', 'sat', 'find', 'end', 'father', 'every', 'person', 'folk', 'seen', 'part', 'tree', 'asked', 'lay', 'rode', 'indeed', 'gave', 'cried', 'art', 'within', 'mine', 'none', 'brought', 'high', 'called', 'forth', 'hath', 'earth', 'door', 'hour', 'moment', 'need', 'gone', 'alway', 'hall', 'began', 'new', 'mind', 'light', 'death', 'leave', 'boy', 'sea', 'full', 'dead', 'Ralph', 'white', 'dear', 'sun', 'better', 'battle', 'gold', 'enough', 'presently', 'answered', 'feet', 'soon', 'round', 'body', 'red', 'hast', 'master', 'court', 'half', 'brother', 'best', 'others', 'quite', 'save', 'Oh', 'black', 'wall', 'therefore', 'home', 'fear', 'among', 'fire', 'country', 'somewhat', 'speak', 'fellow', 'wise', 'wife', 'beside', 'held', 'passed', 'room', 'going', 'spake', 'Jurgen', 'drew', 'Mr', 'rather', 'wilt', 'behind', 'morning', 'de', 'replied', 'live', 'Robin', 'girl', 'city', 'Prince', 'hear', 'war', 'coming', 'next', 'strange', 'work', 'something', 'hard', 'women', 'seem', 'near', 'joy', 'stone', 'four', 'CHAPTER', 'mother', 'shield', 'therewith', 'mountain', 'road', 'rose', 'poor', 'true', 'maiden', 'hold', 'either', 'Nay', 'looking', 'Yea', 'forest', 'tale', 'alone', 'bring', 'rest', 'meet', 'green', 'along', 'sent', 'Ah', 'loved', 'met', 'spear', 'sound', 'mean', 'John', 'truly', 'daughter', 'keep', 'ask', 'help', 'course', 'turn', 'surely', 'slain', 'forward', 'ran', 'led', 'hundred', 'fight', 'around', 'dark', 'spoke', 'ill', 'whose', 'smote', 'Heaven', 'less', 'blood', 'laughed', 'big', 'ground', 'town', 'certain', 'Manuel', 'ready', 'field', 'whether', 'bright', 'Mark', 'beyond', 'river', 'neither', 'bear', 'noble', 'hill', 'evil', 'table', 'Maid', 'known', 'shalt', 'strong', 'wind', 'youth', 'desire', 'open', 'quoth', 'sweet', 'evening', 'stand', 'child', 'got', 'taken', 'sure', 'palace', 'adventure', 'manner', 'gate', 'mighty', 'beautiful', 'longer', 'Grettir', 'book', 'hair', 'dream', 'garden', 'window', 'whole', 'lie', 'soul', 'air', 'given', 'die', 'sight', 'blue', 'call', 'fall', 'suddenly', 'talk', 'sleep', 'bed', 'rock', 'believe', 'wit', 'lost', 'nought', 'kind', 'power', 'morrow', 'nigh', 'glad', 'become', 'truth', 'wonder', 'small', 'anything', 'thine', 'Birdalone', 'close', 'laid', 'across', 'followed', 'cast', 'perhaps', 'grew', 'reason', 'chamber', 'damsel', 'hope', 'Duke', 'bade', 'arose', 'deep', 'Rodriguez', 'felt', 'Trot', 'story', 'pass', 'cause', 'Dale', 'shoulder', 'golden', 'ship', 'departed', 'answer', 'thereof', 'host', 'wish', 'afterward', 'yonder', 'past', 'clear', 'entered', 'song', 'became', 'order', 'journey', 'Gawaine', 'foot', 'Myles', 'return', 'deed', 'children', 'understand', 'plain', 'least', 'saying', 'happy', 'remember', 'silver', 'seek', 'short', 'Earl', 'show', 'Bill', 'low', 'deemed', 'already', 'returned', 'fine', 'beheld', 'island', 'valley', 'almost', 'drink', 'peace', 'pray', 'care', 'five', 'trouble', 'really', 'company', 'silence', 'spirit', 'ring', 'point', 'mouth', 'wherefore', 'beast', 'question', 'ride', 'making', 'doubt', 'eat', 'lo', 'Palomides', 'bird', 'emperor', 'merry', 'wild', 'able', 'concerning', 'blow', 'lip', 'edge', 'll', 'bow', 'Percival', 'silent', 'free', 'follow', 'sit', 'age', 'sorrow', 'hurt', 'slowly', 'behold', 'sore', 'speech', 'money', 'therein', 'grass', 'slay', 'clad', 'run', 'tower', 'ago', 'ear', 'ten', 'struck', 'smiled', 'neck', 'street', 'pleasure', 'kissed', 'opened', 'stranger', 'armed', 'letter', 'Dame', 'tall', 'captain', 'fast', 'second', 'whereas', 'beneath', 'shadow', 'often', 'reached', 'sky', 'piece', 'boat', 'space', 'abide', 'hung', 'want', "Cap'n", 'caught', 'warrior', 'La', 'lad', 'sister', 'six', 'creature', 'beauty', 'used', 'standing', 'use', 'sort', 'finger', 'fact', 'case', 'nature', 'weapon', 'strength', 'kept', 'living', 'anon', 'flower', 'passing', 'chance', 'heavy', 'fool', 'hither', 'honour', 'present', 'send', 'withal', 'amongst', 'whatever', 'seven', 'third', 'straight', 'shame', 'wound', 'twenty', 'force', 'pretty', 'certainly', 'step', 'magic', 'marvel', 'fought', 'quickly', 'feast', 'although', 'Morano', 'cut', 'wo', 'wide', 'thinking', 'large', 'spoken', 'dog', 'play', 'noise', 'top', 'appeared', 'broken', 'dare', 'walked', 'thank', 'sudden', 'cry', 'service', 'thousand', 'lived', 'north', 'died', 'counsel', 'star', 'thither', 'stroke', 'lead', 'gift', 'Count', 'ere', 'minute', 'feel', 'stay', 'greatly', 'grief', 'please', 'wherein', 'lying', 'except', 'moon', 'meat', 'pain', 'read', 'deal', 'Lancelot', 've', 'carried', 'Miss', 'bare', 'bid', 'baron', 'chief', 'princess', 'Bors', 'late', 'law', 'armour', 'raised', 'path', 'iron', 'easy', 'quiet', 'holy', 'sitting', 'moved', 'grey', 'squire', 'wine', 'deem', 'sake', 'welcome', 'meadow', 'tear', 'betwixt', 'cold', 'spring', 'Alas', 'depart', 'slew', 'ladies', 'happened', 'everything', 'damosel', 'mile', 'Lake', 'summer', 'unless', 'worship', 'band', 'stream', 'greater', 'continued', 'enter', 'suppose', 'kindred', 'sometime', 'rich', 'sang', 'ancient', 'tiding', 'taking', 'horn', 'thereafter', 'amidst', 'possible', 'showed', 'fallen', 'soft', 'seeing', 'England', 'wot', 'husband', 'escape', 'knee', 'scarce', 'knowledge', 'talking', 'Geraint', 'slept', 'wisdom', 'pleased', 'Colonel', 're', 'grow', 'floor', 'Musgrave', 'promise', 'goodly', 'moreover', 'dost', 'lovely', 'figure', 'servant', 'huge', 'business', 'wore', 'helm', 'wrong', 'riding', 'fashion', 'Galahad', 'born', 'afraid', 'thrust', 'priest', 'distance', 'south', 'lest', 'covered', 'wounded', 'bore', 'guest', 'yellow', 'aught', 'Richard', 'month', 'upward', 'pleasant', 'early', 'thin', 'abode', 'prisoner', 'running', 'number', 'gentle', 'watch', 'terrible', 'filled', 'sense', 'form', 'Kay', 'weary', 'alive', 'gathered', 'faith', 'aside', 'sought', 'companion', 'lifted', 'harm', 'change', 'dawn', 'winter', 'grown', 'shook', 'custom', 'Sire', 'fain', 'chair', 'received', 'speaking', 'marry', 'content', 'Patricia', 'mercy', 'human', 'sign', 'smile', 'darkness', 'cheek', 'front', 'utter', 'Hood', 'roof', 'quest', 'whence', 'draw', 'laugh', 'Merlin', 'le', 'serve', 'remain', 'prayed', 'sorry', 'leg', 'goes', 'Gareth', 'affair', 'food', 'wonderful', 'try', 'appearance', 'Nevertheless', 'stout', 'carle', 'giant', 'doth', 'safe', 'Sidenote', 'pity', 'hole', 'Mary', 'outside', 'Cliges', 'command', 'cloud', 'suffer', 'Rob', 'meanwhile', 'several', 'cross', 'lightly', 'carry', 'straightway', 'turning', 'lion', 'changed', 'Button', 'week', 'exclaimed', 'spread', 'devil', 'score', 'thence', 'walk', 'hot', 'bank', 'guard', 'ca', 'pale', 'remembered', 'memory', 'common', 'learned', 'saddle', 'loud', 'Dinadan', 'waste', 'smiling', 'desert', 'singing', 'midst', 'instead', 'appear', 'stopped', 'breath', 'bound', 'flame', 'perceived', 'dry', 'Hallblithe', 'gat', 'dragon', 'added', 'joust', 'broad', 'seat', 'forsooth', 'feeling', 'event', 'dwarf', 'yesterday', 'pay', 'later', 'secret', 'didst', 'line', 'length', 'watched', 'trust', 'sooth', 'Kai', 'slave', 'bridge', 'naught', 'forgotten', 'bearing', 'breast', 'brown', 'Thereupon', 'happen', 'bent', 'Ki', 'champion', 'cave', 'broke', 'win', 'west', 'begin', 'bit', 'tried', 'party', 'aid', 'lance', 'holding', 'wait', 'stayed', 'worse', 'forget', 'prove', 'served', 'fierce', 'move', 'asleep', 'beat', 'minded', 'break', 'tongue', 'laughter', 'kingdom', 'Perion', 'laughing', 'drawn', 'aloud', 'placed', 'fail', 'naked', 'lover', 'oft', 'telling', 'kindly', 'maybe', 'cousin', 'hidden', 'Rudolph', 'image', 'enemy', 'mounted', 'France', 'shining', 'danger', 'comfort', 'pink', 'thick', 'easily', 'anger', 'Wolf', 'remained', 'sailor', 'east', 'sing', 'beginning', 'lose', 'talked', 'account', 'sharp', 'instant', 'cup', 'drank', 'Dorothy', 'Sheriff', 'bad', 'arrow', 'dressed', 'cloth', 'foolish', 'clothe', 'purpose', 'fate', 'Majesty', 'foe', 'written', 'wept', 'fled', 'direction', 'caused', 'shore', 'rise', 'immediately', 'Charteris', 'gladly', 'Billy', 'shot', 'tent', 'occasion', 'sad', 'worthy', 'leaving', 'won', 'thrall', 'royal', 'oak', 'raiment', 'grace', 'nt', 'presence', 'colour', 'knowing', 'Roman', 'proper', 'mayst', 'altogether', 'Messire', 'Erec', 'middle', 'kill', 'search', 'picture', 'worth', 'strike', 'stop', 'spite', 'haste', 'monster', 'promised', 'reward', 'perceive', 'different', 'declared', 'game', 'thyself', 'named', 'softly', 'scarcely', 'goat', 'touch', 'dwelt', 'waiting', 'skin', 'following', 'church', 'brave', 'witch', 'Ireland', 'Walter', 'desired', 'Cornwall', 'cities', 'Gawain', 'fellowship', 'soldier', 'whiles', 'reach', 'act', 'gray', 'heed', 'prayer', 'afternoon', 'corner', 'thereby', 'wondered', 'message', 'state', 'wear', 'village', 'Louise', 'threw', 'morn', 'shone', 'belike', 'honor', 'built', 'kiss', 'angry', 'Jack', 'ended', 'throat', 'treasure', 'doubtless', 'paper', 'London', 'clearly', 'happiness', 'swiftly', 'Bishop', 'touched', 'observed', 'considered', 'page', 'likewise', 'nearer', 'exceeding', 'garment', 'single', 'proud', 'wrought', 'prepared', 'fresh', 'Isle', 'cloak', 'mad', 'flesh', 'feared', 'understood', 'robe', 'countenance', 'Philosopher', 'consider', 'dreadful', 'Ling', 'coat', 'Demon', 'verily', 'choose', 'weeping', 'delight', 'wouldst', 'ate', 'stared', 'Melicent', 'thereto', 'Inga', 'music', 'marble', 'killed', 'pointed', 'cheer', 'failed', 'stepped', 'glass', 'somehow', 'lack', 'kin', 'silk', 'wanted', 'bottom', 'played', 'twelve', 'Peter', 'learn', 'messenger', 'Saint', 'meant', 'flew', 'growing', 'shape', 'real', 'rule', 'getting', 'whither', 'fighting', 'married', 'mistress', 'grant', 'delay', 'due', 'ease', 'Margaret', 'afore', 'merely', 'condition', 'pavilion', 'listen', 'interest', 'fortune', 'Christopher', 'prison', 'Burg', 'errand', 'mist', 'quietly', 'waited', 'downward', 'Ector', 'wealth', 'brow', 'speed', 'entirely', 'Peredur', 'terror', 'rushed', 'hanging', 'snow', 'shut', 'subject', 'self', 'nearly', 'shaft', 'rising', 'angel', 'probably', 'reply', 'farewell', 'warm', 'empty', 'simple', 'position', 'dinner', 'luck', 'bushes', 'quick', 'candle', 'lot', 'greatest', 'plan', 'necessary', 'dwelling', 'brethren', 'idea', 'therewithal', 'paid', 'bowed', 'fat', 'closed', 'beloved', 'curious', 'befell', 'Wizard', 'intent', 'bread', 'ceased', 'hate', 'steel', 'note', 'bitter', 'fare', 'hardly', 'dust', 'bone', 'possession', 'leader', 'unknown', 'meal', 'remarked', 'blade', 'teeth', 'token', 'whereof', 'glory', 'according', 'ordinary', 'peril', 'natural', 'future', 'mere', 'Demetrios', 'faint', 'abroad', 'banner', 'discovered', 'vast', 'advice', 'poet', 'elder', 'smoke', 'avail', 'thirty', 'lodging', 'Lung', 'knife', 'gazed', 'army', 'cliff', 'woodland', 'wished', 'hid', 'offer', 'falling', 'measure', 'sick', 'Sigurd', 'fish', 'pocket', 'crown', 'swift', 'Isoud', 'seated', 'narrow', 'pulled', 'uncle', 'commanded', 'thereat', 'Rinkitink', 'vision', 'throne', 'troubled', 'clean', 'hide', 'rested', 'anigh', 'grave', 'shouted', 'gear', 'fiercely', 'decided', 'spot', 'upper', 'Lamorak', 'queer', 'gentleman', 'receive', 'awoke', 'Beaumains', 'attack', 'drawing', 'pearl', 'Percivale', 'false', 'worn', 'Vv', 'twice', 'wert', 'Balin', 'started', 'fared', 'crying', 'nobody', 'craft', 'jewel', 'emotion', 'stead', 'noon', 'borne', 'shout', 'yard', 'shoe', 'seeking', 'hearing', 'thicket', 'gain', 'bold', 'monstrous', 'attention', 'Dick', 'aware', 'wroth', 'armor', 'dwell', 'spent', 'hound', 'agreed', 'defend', 'enemies', 'twain', 'supper', 'Roger', 'encounter', 'tide', 'oath', 'leading', 'inquired', 'pride', 'awhile', 'tournament', 'bodie', 'gown', 'regard', 'throughout', 'handsome', 'board', 'tired', 'seeming', 'color', 'giving', 'Niafer', 'finding', 'scarlet', 'nose', 'everywhere', 'walking', 'Ursula', 'attendant', 'pure', 'wondering', 'storm', 'overcome', 'pace', 'marvellous', 'dance', 'passage', 'swear', 'Hell', 'pardon', 'crowd', 'simply', 'twilight', 'entire', 'cometh', 'write', 'brain', 'grim', 'likely', 'powerful', 'valiant', 'dropped', 'brake', 'watching', 'seized', 'honest', 'odd', 'sighed', 'utterly', 'market', 'rope', 'St', 'vain', 'justice', 'suffered', 'beam', 'slope', 'eager', 'carefully', 'dread', 'view', 'lower', 'Ho', 'noted', 'meeting', 'especially', 'animal', 'rank', 'feather', 'Boolooroo', 'effort', 'circumstance', 'charge', 'magician', 'bidding', 'Thiodolf', 'wing', 'fountain', 'escaped', 'chain', 'detail', 'fancy', 'whilst', 'meaning', 'fit', 'sorely', 'paused', 'glance', 'knighthood', 'perfectly', 'beard', 'lonely', 'listened', 'blew', 'awake', 'burned', 'Bulmer', 'hearken', 'famous', 'building', 'prophet', 'fairest', 'bell', 'sunlight', 'yield', 'hastily', 'hight', 'traitor', 'anyone', 'rain', 'action', 'shepherd', 'eight', 'courage', 'moving', 'row', 'sunset', 'cruel', 'frightened', 'Beale', 'hollow', 'ado', 'madam', 'knowest', 'temple', 'apart', 'thunder', 'key', 'fetch', 'demanded', 'sleeping', 'history', 'finally', 'weak', 'praise', 'wondrous', 'household', 'visit', 'marriage', 'slaying', 'family', 'crossed', 'canst', 'rough', 'leapt', 'blind', 'delivered', 'Pyramid', 'spare', 'hence', 'Eh', 'regarded', 'drop', 'flight', 'Gorge', 'backward', 'mass', 'arise', 'distant', 'degree', 'list', 'fully', 'wist', 'hang', 'ordered', 'tone', 'object', 'chieftain', 'tied', 'loss', 'proved', 'Lichfield', 'wicked', 'pit', 'period', 'sand', 'weep', 'fruit', 'purple', 'bag', 'Bull', 'slow', 'eyed', 'Ha', 'prowess', 'calling', 'Bride', 'higher', 'woe', 'wrath', 'fame', 'staff', 'playing', 'driven', 'destroy', 'pool', 'SLADDER', 'flung', 'excellent', 'sigh', 'Allonby', 'kindness', 'realm', 'drunk', 'stirred', 'various', 'fairy', 'ware', 'burning', 'burst', 'helmet', 'farther', 'speedily', 'race', 'stick', 'jest', 'loving', 'Grand', 'duty', 'breakfast', 'lit', 'staring', 'circle', 'beggar', 'umbrella', 'French', 'surprise', 'encountered', 'limb', 'Utterbol', 'lesser', 'existence', 'dreamed', 'quarrel', 'Simon', 'rage', 'lighted', 'sayest', 'ashamed', 'unhappy', 'trying', 'task', 'plainly', 'particular', 'main', 'scene', 'leaped', 'offered', 'faced', 'wave', 'gotten', 'sail', 'foemen', 'nephew', 'parted', 'discover', 'joyous', 'mayest', 'glittering', 'Otter', 'fairly', 'gay', 'friendly', 'season', 'precious', 'Clement', 'madame', 'Fox', 'smooth', 'onward', 'dared', 'eagle', 'Marhaus', 'Ozma', 'Upmead', 'verity', 'request', 'dim', 'nine', 'require', 'equal', 'Owain', 'grandfather', 'Dom', 'sin', 'gently', 'flee', 'thereon', 'sprang', 'settled', 'record', 'hunger', 'rescue', 'catch', 'courtesy', 'instantly', 'opinion', 'perfect', 'ball', 'vessel', 'tender', 'Ti', 'fifty', 'suit', 'anywhere', 'Jesu', 'merchant', 'Allan', 'tail', 'closely', 'heel', 'Stella', 'thanked', 'serpent', 'weather', 'assured', 'virtue', 'carrying', 'finished', 'romance', 'explained', 'willingly', 'inside', 'throng', 'mayhap', 'perchance', 'saved', 'Freydis', 'expression', 'mane', 'opening', 'attempt', 'stair', 'notice', 'everybody', 'noticed', 'arrived', 'tarry', 'important', 'array', 'doom', 'flat', 'gather', 'Mandarin', 'courteous', 'obey', 'quarter', 'mermaid', 'afar', 'general', 'painted', 'gaze', 'Gascoyne', 'Puysange', 'nodded', 'taste', 'eagerly', 'heap', 'hadst', 'Alexander', 'height', 'misery', 'freely', 'guess', 'hunting', 'whenever', 'axe', 'hermit', 'plenty', 'carved', 'ugly', 'saluted', 'Friar', 'fixed', 'despite', 'forty', 'price', 'stretched', 'longing', 'folly', 'expected', 'endure', 'ruin', 'slipped', 'effect', 'moonlight', 'replies', 'despair', 'needed', 'possessed', 'shown', 'meseemeth', 'box', 'travel', 'loose', 'Guenever', 'headed', 'throw', 'diver', 'willing', 'claim', 'dull', 'rede', 'wandering', 'splendid', 'pair', 'somewhere', 'horseback', 'knave', 'cease', 'English', 'mortal', 'treason', 'surprised', 'health', 'keeping', 'Lionel', 'branches', 'match', 'fly', 'autumn', 'string', 'unable', 'drive', 'believed', 'whispered', 'stir', 'result', 'blame', 'rein', 'lift', 'eaten', 'square', 'asking', 'liked', 'sheep', 'judge', 'Anne', 'perilous', 'Sage', 'chosen', 'horror', 'spur', 'Britain', 'writing', 'GUIDO', 'admitted', 'suggested', 'Otto', 'befallen', 'hungry', 'Hugh', 'share', 'hauberk', 'stuck', 'steed', 'Bilbil', 'methink', 'hearted', 'friendship', 'sooner', 'wooden', 'grieved', 'movement', 'skill', 'neighbour', 'ocean', 'greeting', 'Belle', 'trembling', 'blessed', 'captive', 'taught', 'dress', 'Edward', 'durst', 'familiar', 'drove', 'consequence', 'gleaming', 'leaned', 'Thorbiorn', 'Bleoberis', 'chapel', 'outer', 'eating', 'nonsense', 'reflected', 'respect', 'Gaheris', 'daily', 'Vale', 'sank', 'undoubtedly', 'exceedingly', 'unarmed', 'Chang', 'start', 'permit', 'passion', 'fairer', 'pause', 'haired', 'value', 'Diskos', 'Shard', 'befall', 'demand', 'Lamorack', 'choice', 'dancing', 'sounded', 'shine', 'beg', 'wedded', 'flying', 'cap', 'strode', 'bosom', 'crie', 'whereon', 'exactly', 'Agatha', 'gazing', 'public', 'verse', 'shoot', 'rolled', 'strangely', 'cook', 'bowmen', 'size', 'stolen', 'cat', 'buried', 'approached', 'Zog', 'Isoult', 'shade', 'alighted', 'Yvain', 'Nerle', 'safety', 'older', 'knelt', 'foul', 'emerald', 'milk', 'handed', 'accept', 'worst', 'heartily', 'guide', 'apple', 'Wolfing', 'advanced', 'espied', 'Rome', 'Ormskirk', 'fault', 'calm', 'climbed', 'double', 'lookt', 'joined', 'horrid', 'arranged', 'fond', 'usual', 'becoming', 'bench', 'nice', 'fifteen', 'slumber', 'cool', 'hedge', 'horrible', 'wholly', 'swept', 'expect', 'anxious', 'prize', 'cover', 'conquered', 'million', 'Toft', 'language', 'ford', 'Marquis', 'labour', 'David', 'keen', 'reading', 'nowise', 'greeted', 'busy', 'Mordred', 'heat', 'dying', 'character', 'coward', 'marvelled', 'waved', 'surface', 'sleeve', 'train', 'forced', 'lieth', 'GRACIOSA', 'gathering', 'ghost', 'tablet', 'sacred', 'beaten', 'forehead', 'allowed', 'Footnote', 'regret', 'sweetly', 'pressed', 'lean', 'intention', 'raise', 'mostly', 'tomb', 'thieve', 'Wale', 'fourth', 'Pwyll', 'merrily', 'pan', 'cavern', 'imagine', 'bringing', 'surrounded', 'destroyed', 'inner', 'sport', 'scattered', 'honourable', 'Aye', 'traveller', 'alien', 'sergeant', 'Atli', 'chest', 'Gouvernail', 'whereby', 'sufficient', 'forgive', 'hat', 'mount', 'hap', 'conversation', 'camp', 'anguish', 'tiny', 'dealing', 'vanished', 'mirror', 'Alderman', 'palm', 'Dusky', 'warning', 'dangerous', 'whatsoever', 'highway', 'crept', 'dozen', 'pound', 'Angle', 'chin', 'equally', 'grieve', 'sensible', 'saith', 'smite', 'advantage', 'determined', 'Frogman', 'constant', 'disappeared', 'allow', 'store', 'bar', 'leap', 'comrade', 'mirth', 'aforesaid', 'difficult', 'deer', 'Conant', 'Li', 'fill', 'refuse', 'hoped', 'George', 'clever', 'departure', 'shake', 'wandered', 'someone', 'highly', 'memories', 'impossible', 'younger', 'meantime', 'mid', 'forever', 'sending', 'happening', 'wrapped', 'press', 'sadly', 'appointed', 'engaged', 'steep', 'assuredly', 'besought', 'thrown', 'level', 'shirt', 'thorn', 'explain', 'charm', 'fro', 'prefer', 'mate', 'beating', 'ridge', 'ahead', 'proceeded', 'concern', 'bough', 'wearing', 'Raven', 'boar', 'mood', 'deliver', 'anybody', 'dusk', 'sworn', 'pig', 'wet', 'helped', 'swore', 'pipe', 'wake', 'accomplished', 'club', 'handle', 'victory', 'favour', 'office', 'hawk', 'weight', 'breaking', 'daylight', 'James', 'deny', 'wrote', 'notion', 'Monsieur', 'completely', 'climb', 'belt', 'success', 'altar', 'silently', 'fearful', 'admit', 'passe', 'thrice', 'thief', 'report'}
    SEVEN_PREC_MOST_FREQ_WORDS = SEVEN_PREC_MOST_FREQ_FANT_WORDS.union(SEVEN_PREC_MOST_FREQ_FAIRY_WORDS)
    return len(filtered_words_set.intersection(SEVEN_PREC_MOST_FREQ_WORDS))


def _diversity(filtered_words, filtered_words_set):
    """
    Fraction of unique words from the total number of words (exclusing stop words).
    Higher is more diversified.
    Args:
        filtered_words (list): set of non-stop tokenized words.
        filtered_words_set (set): unique filtered words.
    """
    MIN_WORDS_PER_STORY = 5
    # If empty sentence or only white space or \n or too repetitive.
    if len(filtered_words_set) < MIN_WORDS_PER_STORY:
        return 0

    return len(filtered_words_set) / len(filtered_words)


def _load_KLDIV_loss_function(device):
    """
    Load loss function and its utilities.
    """
    loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
    softmax = Softmax(dim=-1)
    logSoftmax = LogSoftmax(dim=-1)
    loss_fct.to(device)
    softmax.to(device)
    logSoftmax.to(device)
    return loss_fct, softmax, logSoftmax


def KLDIV_error_per_text(tokenizer, preset_model, finetuned_model, text):
    """
    Computes the difference in prediction scores of the given text between the two models.
    The preset_model scores are the input distribution and the finetuned_model scores are the target distribution.
    The forward pass of each model returns the "prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)" (from: https://huggingface.co/transformers/model_doc/gpt2.html).

    Args:
        tokenizer (Pytroch tokenizer): GPT2 Byte Tokenizer.
        preset_model (Pytorch model): preset GPT2 model of the same/ different size of the finetuned model. Assumes model max_length < num_of_words(text).
        finetuned_model (Pytorch model): fine-tuned GPT2 model. Assumes model max_length < num_of_words(text).
        text (str): generated text to check predictions scores for.
    Returns:
        float representing the difference in scores. Bigger is probably better since it usually means the text is closer to the finetuned distribution.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Text is divided to extracts e.g. start, middle, end.
    if isinstance(text, (list, np.ndarray)):
        text = ' '.join(text)

    # if too short return 0
    if len(text) < 10:
        return 0

    # Prepare text to forward pass.
    encodings_dict = tokenizer(text)
    encodings_dict['decoder_input_ids'] = encodings_dict['input_ids']
    text_ids = torch.tensor(
        encodings_dict['input_ids'], device=device, dtype=torch.long)

    # Load loss function, the losses are averaged over batch dimension.
    loss_fct, softmax, logSoftmax = _load_KLDIV_loss_function(device)
    # unsqueeze to add batch_size =1
    # zero index to logits, logits shape (batch_size, len(text_ids), config.vocab_size)
    logists_preset = preset_model(text_ids.to(device).unsqueeze(0), decoder_input_ids=text_ids.to(device).unsqueeze(0))[0]
    logits_finetuned = finetuned_model(text_ids.to(device).unsqueeze(0), decoder_input_ids=text_ids.to(device).unsqueeze(0))[0]

    # input should be in log-probabilities, and target in probabilities.
    return loss_fct(logSoftmax(logists_preset), softmax(logits_finetuned)).item()


def _coherency(texts_sentences, components):
    """
    Args:
        texts_sentences (list): one story text split to sentences.
        lsa (sklearn Vectorizer).
    Based on LSA (TF-IDF -> Truncated SVD), returns the similarity within the story setences in comparison to the first sentence.
    """
    vectorizer = TfidfVectorizer()
    # Compute tf-idf per extract.
    #     transformed_sentences = embedder(texts_sentences)
    transformed_sentences = vectorizer.fit_transform(texts_sentences)

    lsa_model = TruncatedSVD(n_components=components, algorithm='arpack', n_iter=5, random_state=42)

    lsa_top = lsa_model.fit_transform(transformed_sentences)

    # Compute cosine max similarity per extract, shape (#texts_sentences x #texts_sentences).
    similarity = cosine_similarity(lsa_top)
    # Compute similarity scores with first sentence of the rest of the sentences.
    return sum(similarity[0][1:])


def preprocess_generated_text(sample, tokenizer, has_space):
    decoded = tokenizer.decode(
        sample, skip_special_tokens=True, clean_up_tokenization_spaces=True,
)
    # Removing spaces.
    decoded = decoded.strip()
    # Adding a space at the beginning if needed.
    if not has_space:
        decoded = ' ' + decoded
    # Filtering � globally
    return re.sub(u'\uFFFD', '', decoded)


def score_text(text, tokenizer, preset_model, finetuned_model):
    # def score_text(text):
    """ Uses rule-based rankings. Higher is better, but different features have different scales.

    Args:
        text (str/ List[str]): one story to rank.
        tokenizer (Pytroch tokenizer): GPT2 Byte Tokenizer.
        preset_model (Pytorch model): preset GPT2 model of the same/ different size of the finetuned model.
        finetuned_model (Pytorch model): fine-tuned GPT2 model.

    Returns a scores np.array of corresponding to text.
    """
    assert isinstance(
        text, (str, list)), f"score_text accepts type(text) = str/list, but got {type(text)}"

    if isinstance(text, list):
        text = ' '.join(text)

    # Keep same order as in constants.FEATURES
    scores = [0 for _ in range(6)]
    texts_sentences = split_to_sentences(text)
    # print(texts_sentences)
    components = 25
    while components > 0:
        try:
            scores[0] = _coherency(texts_sentences, components)
            break
        except:
            #             print(f"components {components} didn't work, try next")
            components -= 1

    #     scores[0] = _coherency(texts_sentences)
    scores[1] = _readabilty(text, texts_sentences)

    #     Set of text words without punctuation and stop words.
    filtered_words = list(filter(
        lambda word: word not in STOP_WORDS, split_words(text.lower().strip())))
    filtered_words_set = set(filtered_words)
    # Sentiment.
    scores[2] = _sentiment_polarity(filtered_words)

    # Set based measures.
    scores[3] = _simplicity(filtered_words_set)
    scores[4] = _diversity(filtered_words, filtered_words_set)

    #     # The bigger differene, the more tale-like, similar to the fine-tuned model, the text is.
    scores[5] = KLDIV_error_per_text(
        tokenizer, preset_model, finetuned_model, text)

    #     # print(" | ".join(f'{key}: {score:.2f}' for key,
    #     #                  score in zip(constants.FEATURES, scores)))

    return np.array(scores)


def sort_scores(stories_scores):
    """
    Args:
        stories_scores (np.array): 2D matrix of shpae (#stories x #ranking_features).
    Returns the indices of top stories accroding to scores, from highest to lowest (descending).
    """
    # Rescale each feature column across all stories, so that all featues contribute equally.
    # stories_scores_std = (stories_scores - np.mean(stories_scores,axis=0))/np.std(stories_scores,axis=0)
    stories_scores_normalized = stories_scores - \
        np.min(stories_scores, axis=0)
    min_max_denominator = np.max(
        stories_scores, axis=0) - np.min(stories_scores, axis=0)
    # Avoid devision by zero, out to initialize idices where denominator ==0.
    stories_scores_normalized = np.divide(
        stories_scores_normalized, min_max_denominator, out=np.zeros_like(stories_scores_normalized), where=min_max_denominator != 0)

    # Sort by mean story score, shape: (num_stories)
    return np.argsort(np.mean(stories_scores_normalized, axis=1))[::-1]