# https://www.kaggle.com/code/satishgunjal/tutorial-text-classification-using-spacy/notebook

from datetime import datetime
from dotenv import load_dotenv
import os
from uuid import uuid4
import spacy
import csv
import json
import re
import click
import sqlite3

from gensim.models import TfidfModel
from gensim.corpora import Dictionary

load_dotenv()

SPACY_MODEL = os.getenv('SPACY_MODEL')
SPACY_NPROCESS_COUNT = int(os.getenv('SPACY_NPROCESS_COUNT'))
SPACY_BATCH_SIZE = int(os.getenv('SPACY_BATCH_SIZE'))

DATA_BASE_PATH = os.getenv('DATA_PATH')
DATA_BASE_ABSPATH = os.path.abspath(DATA_BASE_PATH)

DB_FILENAME = os.getenv('DB_FILENAME')
DB_FILENAME = os.path.join(DATA_BASE_ABSPATH, DB_FILENAME)

NOTES_MAX_ERROR = int(os.getenv('NOTES_MAX_ERROR'))
NOTES_RESERVED_COLS = ['encounter_id', 'notes', 'mrn', 'row_num']
TERMS_RESERVED_COLS = ['snomed_id', 'title', 'ne', 'label']


@click.group()
def cli():
    pass

def log(txt):
    print('[{}] - {}'.format(datetime.now().isoformat(), txt))

def preprocess_ner(text):
    # Pre-process recognised named entities

    text = text.lower()  # Set to lower case
    # Keep alphanumeric, space, quote characters only
    text = re.sub("[^a-z0-9\'\s]", "", text)
    text = text.strip()

    return text


@cli.command(short_help="Clear named entities from previously parsed notes")
def clear_parsed():

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    cur.execute('DELETE FROM notes_entities;')
    cur.execute('UPDATE notes SET parsed = null;')

    log('Named entities deleted. Notes can now be reparsed.')

    con.commit()
    cur.close()


@cli.command(short_help="Calculate named entity's TFIDF for all parsed notes")
def calc_tfidf():

    con = sqlite3.connect(DB_FILENAME)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    def get_docs_entities():
        cur2 = con.cursor()

        for row in cur2.execute("select notes_id, group_concat(ne,'|') as ents from notes_entities group by notes_id;"):
            yield {'id': row['notes_id'], 'ents': row['ents'].split('|')}

    doc_ids = []
    doc_ents = []
    for doc in get_docs_entities():
        doc_ents.append(doc['ents'])
        doc_ids.append(doc['id'])

    docs_dict = Dictionary(doc_ents)

    docs_corpus = [docs_dict.doc2bow(doc, allow_update=True)
                   for doc in doc_ents]
    model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict, smartirs='bfu')
    docs_tfidf = model_tfidf[docs_corpus]
    docs_dict_values = list(docs_dict.values())

    log('Total Unique NE terms found:{}'.format(len(docs_dict_values)))

    for rownum in range(len(doc_ids)):

        ent_tfidf = None
        for ent in doc_ents[rownum]:

            try:
                doc_dict_index = docs_dict_values.index(ent)

                for token in docs_tfidf[rownum]:
                    if token[0] == doc_dict_index:
                        id, ent_tfidf = token
                        break
            except:
                pass

            sql = "UPDATE notes_entities SET ne_tfidf= ? WHERE notes_id= ? and ne = ?"
            cur.execute(sql, [ent_tfidf, doc_ids[rownum], ent])

    con.commit()
    cur.close()


@cli.command(short_help='Parse loaded notes using spaCY model.')
@click.option("--batchsize",default=1000,required=False,help="Adjust batch size depending on available system memory.(Default:1000)")
def parse_notes(batchsize):

    con = sqlite3.connect(DB_FILENAME)
    con.row_factory = sqlite3.Row
    cur2 = con.cursor()

    nlp = spacy.load(SPACY_MODEL)

    n = []
    n_ids = []

    docs_processed = 0

    def process_batch(notes, notes_ids):
        cur = con.cursor()

        duplicate_count = 0
        new_count = 0

        docs = [d for d in nlp.pipe(
            notes, n_process=SPACY_NPROCESS_COUNT, batch_size=SPACY_BATCH_SIZE)]

        doc_count = len(docs)

        for rownum in range(doc_count):  # Capture recognized NE into result

            for ent in docs[rownum].ents:
                try:
                    text = preprocess_ner(ent.text)
                    label = ent.label_.lower()
                    start_char = ent.start_char
                    end_char = ent.end_char

                    sql = "INSERT INTO notes_entities (notes_id,ne,label,start_char,end_char) values(?,?,?,?,?);"
                    cur.execute(sql, [notes_ids[rownum], text,
                                label, start_char, end_char])

                    sql = "UPDATE notes SET parsed ='Y' WHERE id=?"
                    cur.execute(sql, [notes_ids[rownum]])

                    new_count += 1

                except sqlite3.IntegrityError:
                    duplicate_count += 1

            docs_processed = docs_processed + 1

        log('New NEs inserted: {}. Duplicate NEs: {}. Total docs: {}'.format(
            new_count, duplicate_count, docs_processed))

        con.commit()
        cur.close()

    for row in cur2.execute('SELECT * from notes where parsed is null or id not in (select notes_id from notes_entities);'):
        row_dict = dict(row)
        n.append(row_dict['notes'])
        n_ids.append(row_dict['id'])

        if len(n) >= batchsize:
            process_batch(n, n_ids)
            n = []
            n_ids = []

    if len(n) > 0 or len(n) != batchsize:
        process_batch(n, n_ids)


@cli.command()
@click.option('--input_file', prompt='Input CSV File', help='Input CSV File')
def load_notes_to_db(input_file):

    build_database.callback()

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    input_filename = os.path.basename(input_file)

    error_count = 0

    insert_count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip().lower()
        cols = header_line.split(',')
        metadata_cols = list(set(cols) - set(NOTES_RESERVED_COLS))

    with open(input_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f, fieldnames=cols, delimiter=',')
        row_count = 1

        for row in csv_reader:  # Iterate through each row in csv
            if row_count == 1:
                row_count += 1
                continue

            metadata = {}

            for mc in metadata_cols:
                metadata[mc] = row[mc]

            try:
                sql = "INSERT INTO notes (id,encounter_id,mrn,row_num,notes,source_file,metadata) values(?,?,?,?,?,?,?);"
                cur.execute(sql, [uuid4().hex, row['encounter_id'], row['mrn'], row_count,
                            row['notes'], input_filename, json.dumps(metadata)])

                insert_count += 1
            except sqlite3.IntegrityError:
                log('Skipping row {} due to Integrity Error. Duplicate entry found for combination of encounter_id:{} & mrn:{} & source_file:{} & row_num:{}'.format(
                    row_count, row['encounter_id'], row['mrn'], input_filename, row_count))
                error_count += 1

                if error_count > NOTES_MAX_ERROR:
                    log('Max error count reached. Process cancelled.')
                    con.rollback()
                    cur.close
                    return

            row_count += 1

    con.commit()
    cur.close()

    log('Notes inserted {}'.format(insert_count))


@cli.command()
@click.option('--input_file', prompt='Input CSV File', help='Input CSV File')
def load_terms_to_db(input_file):

    build_database.callback()

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()
    insert_count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip().lower()
        cols = header_line.split(',')

    with open(input_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f, fieldnames=cols, delimiter=',')
        row_count = 1

        for row in csv_reader:  # Iterate through each row in csv
            if row_count == 1:
                row_count += 1
                continue

            try:
                sql = "INSERT INTO terms (id,snomed_id,title,ne,label) values(?,?,?,?,?);"
                cur.execute(sql, [uuid4().hex, row['snomed_id'].strip(), row['title'].strip(
                ), preprocess_ner(row['ne']), row['label'].strip().lower()])

                insert_count += 1
            except sqlite3.IntegrityError:
                log('Skipping row {} due to Integrity Error. Duplicate entry found for combination of label:{} & ne:{}'.format(
                    row_count, row['label'], row['ne']))

            row_count += 1

    con.commit()
    cur.close()

    log('Terms inserted {}'.format(insert_count))


@cli.command(short_help="Build a new database if it doesn't exist")
def build_database():

    con = sqlite3.connect(DB_FILENAME)

    cur = con.cursor()

    create_sql = "CREATE TABLE IF NOT EXISTS notes (id text PRIMARY KEY, mrn text, encounter_id text, notes text, source_file text, metadata text, parsed text, row_num integer, UNIQUE(encounter_id,mrn,source_file,row_num)); "
    create_sql = create_sql + \
        "CREATE TABLE IF NOT EXISTS notes_entities (notes_id text, ne text, label text, start_char integer, end_char integer, ne_tfidf real, PRIMARY KEY (notes_id, ne, label, start_char, end_char) FOREIGN KEY (notes_id) REFERENCES notes(id) FOREIGN KEY (ne,label) REFERENCES terms(ne,label)); "
    create_sql = create_sql + \
        "CREATE TABLE IF NOT EXISTS terms (id text, snomed_id text, title text, label text, ne text, PRIMARY KEY (label, ne)); "
    create_sql = create_sql + \
        "CREATE TABLE IF NOT EXISTS rejected_notes (notes_id text, rejected_date text, reason text); "

    cur.executescript(create_sql)

    con.commit()
    cur.close()


@cli.command(short_help="Reset database. ALL DATA WILL BE REMOVED.")
def reset_database():

    con = sqlite3.connect(DB_FILENAME)

    cur = con.cursor()

    sql = '''DELETE FROM notes;
    DELETE FROM notes_entities;
    DELETE FROM terms;
    DELETE FROM rejected_notes;
    '''

    cur.executescript(sql)

    con.commit()
    cur.close()

    log('Database reset successful.')


if __name__ == '__main__':
    cli()
    pass
