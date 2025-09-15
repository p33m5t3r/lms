import os
import re
import requests
import time
from pathlib import Path


CORPUS = {
    # prose, english
    1400:   'great_expectations',
    1023:   'bleak_house',
    98:     'a_tale_of_two_cities',
    766:    'david_copperfield',
    1342:   'pride_and_prejudice',
    161:    'sense_and_sensibility',
    2701:   'moby_dick',
    10:     'the_kjv_bible',
    4300:   'ulysses',
    4217:   'portrait_of_the_artist',
    768:    'wuthering_heights',
    2591:   'grimms_fairy_tales',

    # prose, translated
    2554:   'crime_and_punishment',
    28054:  'the_brothers_karamazov',
    2600:   'war_and_peace',
    1399:   'anna_karenina',
    1184:   'the_count_of_monte_cristo',
    135:    'le_miserables',
    2413:   'madame_bovary',

    # verse
    26:     'paradise_lost',
    1727:   'the_odyssey',
    6130:   'the_illiad',
    228:    'the_aeneid',       # indented
    1322:   'leaves_of_grass',  # indented
    8800:   'the_divine_comedy',

    # mixed
    100:    'shakespeare_complete',

    # philosophy
    8438:   'aristotle_ethics',
    3800:   'spinoza_ethics',
    4280:   'critique_of_pure_reason',
    55108:  'logic_of_hegel',
    815:    'democracy_in_america_v1',
    816:    'democracy_in_america_v2',
    3207:   'leviathan',
    4363:   'beyond_good_and_evil',
    4705:   'a_treatise_of_human_nature',
    34901:  'on_liberty',

    # encyclopedia/dictionary/reference
    29765:  'websters_unabridged_dictionary'
}


DATA_DIR = Path(__file__).parent.parent / 'data'

def dl_title_by_id(id):
    url = f'https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt'
    response = requests.get(url)
    with open(f'{DATA_DIR}/raw/{id}.txt', 'w') as f:
        f.write(response.text)

def clean_text_by_id(id):
    with open(f'{DATA_DIR}/raw/{id}.txt', 'r') as f:
        text = f.read()
    
    # Find start and end markers
    start_match = re.search(r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', text)
    end_match = re.search(r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', text)
    
    if start_match and end_match:
        cleaned = text[start_match.end():end_match.start()].strip()
    else:
        cleaned = text  # Just use the whole thing if we can't find markers
        print("warning: cleaning didnt rm gutenberg copyright headers/footers")
    
    with open(f'{DATA_DIR}/processed/{id}.txt', 'w') as f:
        f.write(cleaned)

def ensure_corpus_texts(ids):
    os.makedirs(f'{DATA_DIR}/raw', exist_ok=True)
    os.makedirs(f'{DATA_DIR}/processed', exist_ok=True)
    
    for id in ids:
        if not os.path.exists(f'{DATA_DIR}/raw/{id}.txt'):
            print(f'Downloading {id}...')
            dl_title_by_id(id)
            time.sleep(2)  # Respect Gutenberg TOS
        
        if not os.path.exists(f'{DATA_DIR}/processed/{id}.txt'):
            print(f'Cleaning {id}...')
            clean_text_by_id(id)


def main():
    ids = list(CORPUS.keys())
    ensure_corpus_texts(ids)

if __name__ == "__main__":
    main()




