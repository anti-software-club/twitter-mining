import sys, os
import re
from PorterStemmer import PorterStemmer

PROJECT_DATA_FILE = "v4_11_23.txt"

CATEGORIES = {0: "0.txt", 1: "1.txt", 2:"2.txt", 3:"3.txt", 4:"4.txt", 5:"5.txt", 6:"6.txt", 7:"7.txt", 8:"8.txt", 9:"9.txt"}

LINE_PATTERN = re.compile(r""" \+(\d)\.(\d)\d (.*) """, re.X) 

alphanum_re = re.compile('[^a-zA-Z0-9$]')

p_stemmer = PorterStemmer()


def parse_line(line, stopList):
    line_matches = re.findall(LINE_PATTERN, line);

    if not line_matches:
        return ('', -1)

    l = line_matches[0]
    ones = int(l[0]) 
    tenths = int(l[1])
    content = l[2]
                
    #standardize
    content = content.lower()
    content = [xx.strip() for xx in content.split()] 
    content = [alphanum_re.sub('', xx) for xx in content] #remove nonalphanumeric
    content = [word for word in content if 'http' not in word and word != '\n'] #remove website addresses and newline chars
    content = [xx for xx in content if xx != ''] #remove any empty words
                
    #Stem and filter stopwords
    content = filterStopWords(content, stopList)
    content = [p_stemmer.stem(xx) for xx in content]  
    content = ' '.join(content)

    #score = '%s.%s  %s\n' % l
    if ones == 1: # so that scores of 1.00 are counted 
        tenths = 9
             
    return (content, tenths)

def filterStopWords(words, stopList):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def segmentWords(s):
    return s.split()

def readFile(fileName):
    #p_stemmer = PorterStemmer()
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    stemmed_contents = []
    for word in contents: #TODO: turn on stemming 
        #stemmed_contents.append(p_stemmer.stem(word))
        stemmed_contents.append(word);
    result = segmentWords('\n'.join(stemmed_contents)) 
    return result



        
def main(argv):
    lines_per_file = 10
    smallfile = None
    stopList = set(readFile('english.stop.txt'))

    with open(PROJECT_DATA_FILE) as bigfile:
        for lineno, line in enumerate(bigfile):
            (line, category) = parse_line(line, stopList)
            if category == -1: continue
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = 'tweet_{}_{}.txt'.format(category, lineno + lines_per_file)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()   


if __name__ == '__main__':
    main(sys.argv)
