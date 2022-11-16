import sqlite3 as lite
import xml.etree.ElementTree as et
import sys, os, errno, codecs
import nltk.data
from tensorflow import flags

def delete_db_file(filename):
	try:
		os.remove(filename)
		print("Old database at file %s deleted. Generating new database.", filename)
	except OSError as e: # this would be "except OSError, e:" before Python 2.6
		if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
			raise # re-raise exception if a different error occurred
		else:
			print("Database file does not exist. Proceeding with database creation.")

def create_tables(c):
	if c:
		# Create tables
		sql = '''CREATE TABLE IF NOT EXISTS article (
							id INTEGER NOT NULL PRIMARY KEY, 
							filename TEXT NOT NULL);'''
		c.execute(sql)

		sql = '''CREATE TABLE IF NOT EXISTS sentence (
					id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
					fk_article_id INT NOT NULL,
					fragment TEXT NOT NULL,
					offset INT NOT NULL,
					length INT NOT NULL,
					isplag BOOL NOT NULL,
					foreign key (fk_article_id) references article(id));'''
		c.execute(sql)

def insert_into_article_table(c, f):
	sql = '''insert into article(id, filename) values(?,?)'''
	c.execute(sql, (int(f.name[-9:-4]), f.name))
	return c.lastrowid

def insert_into_sentence_table(c, article_id, data, sentence, plags):
	sql = '''insert into sentence (fk_article_id, fragment, offset, length, isplag) values (?,?,?,?,?)'''
	values = [article_id, sentence.replace('\n', ' ').replace('\r', ' '), data.index(sentence), len(sentence)]
	isplag = 0
	for plag_section in plags:
		plag_interval = range(plag_section[0], plag_section[0] + plag_section[1])
		if values[3] in plag_interval:
			isplag = 1
			break
	values.append(isplag)
	c.execute(sql, values)

def populate_tables(c, tokenizer, datafolder):
    filelist = os.listdir(datafolder)
    filelist.sort()
    for file in filelist:	
        file = os.path.join(datafolder, file) # Get path that works for Windows and Linux
        if file.endswith('.txt'):
            xmlfile = file.replace('.txt', '.xml')
            with codecs.open(file, encoding='utf-8-sig') as f:
                article_id = insert_into_article_table(c, f)
                data = f.read()
                print(f.name)
                tree = et.parse(xmlfile)
                root = tree.getroot()
                plags = []
                for feature in root:
                    if 'name' in feature.attrib and feature.attrib['name'] == 'artificial-plagiarism':
                        offset = int(feature.attrib['this_offset'])
                        length = int(feature.attrib['this_length'])
                        plags.append((offset, length))
                sentences = tokenizer.tokenize(data)
                for sentence in sentences:
                    insert_into_sentence_table(c, article_id, data, sentence, plags)

def get_ignore_list():
	return [
		'American Tract Society',
		'Consumers\' League of New York, The',
		'Guaranty Trust Company of New York',
		'Teachers of the School Street Universalist Sunday School, Boston',
		'Three Initiates',
		'United States Patent Office',
		'United States.Army.Corps of Engineers.Manhattan District',
		'United States.Congress.House.Committee on Science and Astronautics.',
		'United States.Dept.of Defense',
		'United States.Executive Office of the President',
		'United States.Presidents.',
		'Work Projects Administration',

	]


if __name__ == '__main__':
	flags.DEFINE_string('db', 'plag.db', 'Path to the database file to be generated (default: plag.db).')
	flags.DEFINE_string('data', 'dataset', 'Folder containing the input PAN dataset (default: dataset).')

	db_filename = flags.FLAGS.db
	datafolder = flags.FLAGS.data

	print('Database will be generated at ', db_filename)
	delete_db_file(db_filename)
	db = None
	global ignore_list
	ignore_list = get_ignore_list()
	try:
		db = lite.connect(db_filename)
		c = db.cursor()
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		create_tables(c)
		populate_tables(c, tokenizer, datafolder)

		db.commit()
	except lite.Error as e:
		print("Error %s:" % e.args[0])
		sys.exit(1)
	finally:
		if db:
			db.close()
