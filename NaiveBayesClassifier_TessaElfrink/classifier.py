import os
import re
import operator
import random
import math
from nltk.tokenize import word_tokenize
print("Hoi")
class File:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        
        path = self.path + name
        f = open(path)       
        # maak een lijst waarin elk woord uit de tekst een element is
        self.all_words = self.tokenize_text(f.read())
        
    def all_words_in_file(self):
        """Return a list of all words in the file."""
        return self.all_words
    
    def contains_word(self, word):
        """Check if the file contains a given word."""
        word = word.lower()
        return word in self.all_words
    
    def tokenize_text(self, text):
        """Return a list of all the words in the given text, stripped of punctuation marks and in lower case."""
        # word_tokenize(text) maakt een lijst waarin elk woord of leesteken uit de tekst een element is
        word_list = word_tokenize(text)
        new_word_list = []
        # filter op tekstelementen die bestaan uit letters, een apostrof of een streepje (zoals binnen een woord kan voorkomen)     
        r = re.compile("[a-z'-]+")
        for word in word_list:
            word = word.lower()
            if r.match(word):
                # in de nieuwe woordenlijst komen alleen woorden, geen leestekens
                new_word_list.append(word)
        return new_word_list

class Category:
    def __init__(self, name, path, file_list):
        self.name = name
        self.path = path
        self.file_list = []
        
        for file in file_list:
            # maak een lijst met alle bestanden in de categorie
            self.file_list.append(File(file, self.path))
            
        self.all_words = []
        for file in self.file_list:
            # maak een lijst met alle woorden uit alle bestanden in de categorie
            self.all_words.extend(file.all_words_in_file())
                
    def get_name(self):
        """Return the name of the category."""
        return self.name
    
    def get_files(self):
        """Return a list of all the files in the category."""
        return self.file_list
    
    def num_of_files(self):
        """Return the number of files in the category."""
        return len(self.file_list)
    
    def all_words_in_category(self):
        """Return a list of all the words in the files in the category."""
        return self.all_words
    
    def num_of_files_with_word(self, word):
        """Return the number of files in the category that contain a given word."""
        file_number_with_word = 0
        for file in self.file_list:
            if file.contains_word(word):
                file_number_with_word += 1
        
        return file_number_with_word

class NBClassifier:
    def __init__(self, train_set_path):
        print("Started")
        self.top_path = train_set_path
        
        # deel de trainbestanden op in een 'ham'-categorie en een 'spam'-categorie
        self.ham_list, self.spam_list = sort_files_into_categories(self.top_path)
        c1 = Category("ham", self.top_path, self.ham_list)
        c2 = Category("spam", self.top_path, self.spam_list)
        self.categories = [c1, c2]
        
        # maak lijsten van alle woorden en alle bestanden in de trainset     
        self.all_words = []
        self.train_set = []
        for c in self.categories:
            self.all_words.extend(c.all_words_in_category())
            self.train_set.extend(c.get_files())
        
        # initialiseer variabelen voor training en classificatie
        self.word_counts = {}
        self.smoothed_probabilities = {}
        
        # initialiseer variabele voor voortgangsrapportage
        self.message = "Started classifier..."
        print("Started classifier...")
        
          
    def train(self, vocabulary_size):
        """Train the classifier with the given train set on the specified number of words."""
        # voortgangsrapportage
        self.message = "Started training..."
        print(self.message)
        
        # maak een lijst van de woorden die het beste kunnen worden gebruikt voor classificatie
        selected_words = [word[0] for word in self.select_best_x_words(vocabulary_size)]
               
        for word in selected_words:
            self.word_counts[word] = {}
            self.smoothed_probabilities[word] = {}
            for cat in self.categories:              
                self.word_counts[word][cat.get_name()] = cat.num_of_files_with_word(word)
                self.smoothed_probabilities[word][cat.get_name()] = (self.word_counts[word][cat.get_name()] + 1) / (cat.num_of_files() + 2)

    def classify(self, file):
        """Classify a file into of the categories and return the name of the category"""
        # voortgangsrapportage
        self.message = "Started classifiying..."
        print(self.message)
        
        current_score = -10000000
        current_category = self.categories[0]
        for c in self.categories:
            new_score = 0.0
            for word, data in self.smoothed_probabilities.items():
                if file.contains_word(word):
                    smoothed_probabily_estimate = data[c.get_name()]
                    new_score +=  math.log(smoothed_probabily_estimate, 2)
            prior_probability = c.num_of_files() / len(self.train_set)
            new_score +=  math.log(prior_probability, 2)
            if new_score > current_score:
                current_score = new_score
                current_category = c
        
        return current_category.get_name()
    
    def select_best_x_words(self, x):
        """Return a list that contains the specified number of words with the highest chi square scores."""
        word_set = set(self.all_words)
        word_scores = {}
        for word in word_set:        
            word_scores[word] = self.chi_square(word)
            
        sorted_word_scores = sorted(word_scores.items(), key=operator.itemgetter(1))
        best = sorted_word_scores[-x:]
        
#         f = open(r'C:/Users/TessaElfrink/Documents/PremasterPythonNLTK/300words_highest_chisquares.txt', 'w')
#         for word in best:
#             f.write('%s, %f\n' % (word[0], word[1]))
            
        return best
    
    def chi_square(self, word):
        "Return the chi square score for the given word based on the existing categories."
        observed_values = []
        expected_values = []
        
        sum_of_files = len(self.train_set)
        
        sum_of_files_with_word = 0
        sum_of_files_without_word = 0
        for c in self.categories:
            num_files = c.num_of_files()
            with_word = c.num_of_files_with_word(word)
            without_word = num_files - with_word
            observed_values.append(with_word)
            observed_values.append(without_word)
             
            sum_of_files_with_word += with_word
            sum_of_files_without_word += without_word
        
        for c in self.categories:
            num_files = c.num_of_files()
            expected_value1 = sum_of_files_with_word * num_files / sum_of_files
            expected_value2 = sum_of_files_without_word * num_files / sum_of_files
            expected_values.append(expected_value1)
            expected_values.append(expected_value2)
        
        test_statistic = 0
        for observed, expected in zip(observed_values, expected_values):
            if expected > 0:
                test_statistic += (float(observed)  - float(expected))**2 / float(expected)
         
        return test_statistic
    
def sort_files_into_categories(path):
    """Sort the files at the given location into a 'spam' category when the file name starts with 'sp' and into a 'ham' category otherwise."""
    files = os.listdir(path)
    ham_list = []
    spam_list = []
    r = re.compile("^sp")
    for file in files:
        if r.match(file):
            spam_list.append(file)
        else:
            ham_list.append(file)
   
    return [ham_list, spam_list]


def compute_accuracy(test_set_path, vocabulary_size = 100):
    """Compute and return the accuracy of the classifier using the files in the given test set and a given vocabulary size."""
    print("Started")
    correct = 0.0
    incorrect = 0.0
    
    ham_list, spam_list = sort_files_into_categories(test_set_path)
    
    nbc = NBClassifier(r"C:/Users/TessaElfrink/Documents/PremasterPythonNLTK/corpus-mails/corpus/train/")
    nbc.train(vocabulary_size)
    
    files = os.listdir(test_set_path)
    random.shuffle(files)
    for file in files:
        result = nbc.classify(File(file, test_set_path))
        if (file in ham_list and result == "ham") or (file in spam_list and result == "spam"):
            correct += 1
        else:
            incorrect += 1
    
    # voortgangsrapportage
    message = "Computing accuracy..."
    print(message)
       
    accuracy_stat = (correct / (correct + incorrect)) * 100
    return accuracy_stat

    # voortgangsrapportage
    message = "Finished."
    print(message)


print("100: ", compute_accuracy(r"C:/Users/TessaElfrink/Documents/PremasterPythonNLTK/corpus-mails/corpus/test/", 100))
#print("200: ", compute_accuracy(r"/test/", 200))
#print("300: ", compute_accuracy(r"/test/", 300))
