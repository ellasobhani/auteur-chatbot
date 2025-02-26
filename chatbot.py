"""
Building a chatbot for movie recommendations using ML-based modules and logistic regression.
"""
import numpy as np
import argparse
import joblib
import re  
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import nltk 
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple

import util

class Chatbot:
    """Class that implements chatbot auteur"""

    def __init__(self):
        # The chatbot's default name is `moviebot`.
        self.name = 'auteur' 

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')
        
        # Load sentiment words 
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Train the classifier
        self.train_logreg_sentiment_classifier()

        # used in process
        self.state = "rate"
        self.rated_films = {}
        self.rated_count = 0
        self.current_movie = ""
        self.current_mov_sentiment = ""
        self.indices = []
        self.rec_num = 0

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self) -> str:
        """
        Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """

        return """
        this is a chatbot which will provide you with film recommendations based on your proclivities.
        give me a film and tell me how you felt about it. and maybe four more after that. 

        to exit: write ":quit" (or press Ctrl-C to force the exit)
        """

    def greeting(self) -> str:
        """Return a message that the chatbot uses to greet the user."""

        greeting_message = "What's up? Tell me a movie you like with the movie name in quotations."

        return greeting_message

    def goodbye(self) -> str:
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """

        goodbye_message = "until next time..."

        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function. 
        """
        return str(line)

    def process(self, line: str) -> str:
        """
        Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'
        
        Arguments: 
            - line (str): a user-supplied line of text
        
        Returns: a string containing the chatbot's response to the user input

        Hints: 
            - We recommend doing this function last (after you've completed the
            helper functions below)
            - Try sketching a control-flow diagram (on paper) before you begin 
            this 
            - We highly recommend making use of the class structure and class 
            variables when dealing with the REPL loop. 
            - Feel free to make as many helper funtions as you would like 
        """
        response = ""
        print("rated movie count: " + str(self.rated_count))
        
        # rate state
        # check to see if we have enough rates to make a recommendation, if so, skip to recommend state
        if self.rated_count == 5:
            self.state = "recommend"

        if self.state == "rate":
            extracted_titles = self.extract_titles(line)
            print("extracted_titles: ")
            print(extracted_titles)
                
            # if no titles found, ask to try another movie
            if len(extracted_titles) == 0:
                response = "I couldn't find any movie name in '{}'. Please try again.".format(line)
                       
            # if a title is found from extract_titles, store the title, sentiment, and index (or indices) of movie in class vars            
            if len(extracted_titles) == 1:
                self.current_movie = extracted_titles[0]
                self.current_mov_sentiment = self.predict_sentiment_rule_based(line)
                self.indices = self.find_movies_idx_by_title(self.current_movie)
                print("current movie: ")
                print(self.current_movie)
                print("indices of current movie: ")
                print(self.indices)
                print("sentiment of current movie: ")
                print(self.current_mov_sentiment)

                # if the movie title has more than one index (sequels, etc.)
                if len(self.indices) > 1:
                    movies = [self.titles[index] for index in self.indices] 
                    response = "It seems there are several matches to the movie title you named. Which of {} is the movie you were referring to? Please tell us the date in quotations.".format(movies)
                    self.state = "disambiguate"
                    return response

                if len(self.indices) == 1:    
                    # if sentiment is unclear
                    if self.current_mov_sentiment == 0:
                        response = "I wasn't able to tell whether you enjoyed {} or not. Could you clarify?".format(self.current_movie)
                            
                    # if sentiment is positive
                    if self.current_mov_sentiment == 1:
                        self.rate_movie(self.current_movie)
                        # if self.rated_count == 5:
                        #     self.state = "recommend"
                        response = "So you liked {}. I'm glad! Can you give me another movie?".format(self.current_movie)
                                
                    # if sentiment is negative
                    if self.current_mov_sentiment == -1:
                        self.rate_movie(self.current_movie)
                        # if self.rated_count == 5:
                        #     self.state = "recommend"
                        response = "So you didn't like {}. Sorry about it. Can you give me another movie?".format(self.current_movie)
                
                if len(self.indices) == 0: 
                    response = "I couldn't find {} movie in my database. Could you tell me another movie you like or dislike?".format(self.current_movie)
                    
                
            # if extract_titles gives back several titles
            if len(extracted_titles) > 1:
                response = "Please just give us one movie title at a time. Which of the following did you mean: {}?".format(extracted_titles)
            
            
        # disambiguate state
        if self.state == "disambiguate":
            clarification = self.extract_titles(line)
            if len(clarification) ==1:
                disambiguated_list = self.disambiguate_candidates(clarification[0], self.indices)
                print("disambiguated index list: ")
                print(disambiguated_list)
                if len(disambiguated_list) == 1:
                    self.current_movie = self.titles[disambiguated_list[0]][0]
                    print("current movie: ")
                    print(self.current_movie)
                    self.rate_movie(self.current_movie)
                    response = "Awesome, thanks. Let's keep going then."
                    self.state = "rate" 
                    self.indices = []
                elif len(disambiguated_list) > 1:
                    disambiguated_list_titles = [self.titles[index] for index in disambiguated_list] 
                    response = "Hmm it still seems that there are several matches to the movie and year you named. Which of {} is the movie you were referring to? Please share the movie title AND year it was made in quotations.".format(disambiguated_list_titles)
                elif len(disambiguated_list) == 0:
                    response = "I couldn't resolve the title. Could you please clarify the title again?"
                    self.state = "rate"
                    self.current_movie = ""
            else:
                response = "Please only enter one date."
            
        # recommend state        
        if self.state == "recommend":
            recommended = self.recommend_movies(self.rated_films, 3)
    
            if line.lower() == "no":
                response = self.goodbye()
                print(response)  
                exit() 

            if self.rec_num < len(recommended):
                response = "I finally have a recommendation ready for you... {}. Would you like to hear another?".format(recommended[self.rec_num])
                self.rec_num += 1
            else:
                response = self.goodbye()
                print(response)  
                exit() 
                
        return response
    
    def rate_movie(self, title: str):
        if self.current_mov_sentiment != 0:
            print("title being rated: " + title)
            index = self.find_movies_idx_by_title(title)[0]
            self.rated_films[index] = self.current_mov_sentiment
            self.rated_count += 1
            self.current_mov_sentiment = ""
            if self.rated_count == 5:
                print("ready to recommend")
                self.state = "recommend"
            print(self.rated_films)
        else:
            print("title rated neutrally and not recorded in dict")

    def extract_titles(self, user_input: str) -> List[str]:
        """
        Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Arguments:     
            - user_input (str) : a user-supplied line of text

        Returns: 
            - (list) movie titles that are potentially in the text
        """
        
        # regex pattern to get anything within double quotes
        regex = r'"([^"]+)"'
        
        # match user input on the regex
        tuples = re.findall(regex, user_input)
        
        # get a list of the matches
        films = [fst for fst in tuples]
        
        return films
    

    def find_movies_idx_by_title(self, title:str) -> List[int]:
        """ 
        Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Arguments:
            - title (str): the movie title 

        Returns: 
            - a list of indices of matching movies
            """
        
        # initialize empty array of movies
        ret = []
        
        # look for matching movies in self.titles and add them to ret
        for i in range(len(self.titles)):
            if title in self.titles[i][0]:
                ret.append(i)
        
        # return the resulting list of matched movies
        return ret


    def disambiguate_candidates(self, clarification: str, candidates: list) -> List[int]: 
        """
        Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this 
        should return multiple elements in the list which the clarification could 
        be referring to. 
    
        Arguments: 
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns: 
            - a list of indices corresponding to the movies identified by the clarification

        """
        clarified_list = []

        for movie_index in candidates:
            line = self.titles[movie_index][0]
            matches = re.findall(clarification, line)
            for match in matches:
                if match != None:
                    clarified_list.append(movie_index)
        return clarified_list
    

    ############################################################################
    # 3. Sentiment                                                             #
    ########################################################################### 

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """
        Predict the sentiment class given a user_input, using rule-based approach.

        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: 
            - (int) a numerical value: -1 (negative sentiment), 0 (neutral), +1 (postive sentiment)
        """
        pos_tok_count = 0
        neg_tok_count = 0
        
        user_input_tokens = re.sub(r'[^\w\s]', '', user_input)
        
        for tok in user_input_tokens.lower().split():
            sentiment = self.sentiment.get(tok)
            if tok in self.sentiment:
                if self.sentiment[tok] == "pos":
                    pos_tok_count += 1
                else:
                    neg_tok_count += 1
                    
        if pos_tok_count > neg_tok_count:
            return 1
        elif pos_tok_count == neg_tok_count:
            return 0
        else:
            return -1
 

    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset 

            -1 inputed into sklearn corresponds to "rotten" in the dataset 
            +1 inputed into sklearn correspond to "fresh" in the dataset 
        
        """ 
        #load training data  
        texts, y_str = util.load_rotten_tomatoes_dataset()
        
        # transform class labels to ints
        y = [-1 if elem=="Rotten" else 1 for elem in y_str]
        
        # lowercase all the texts
        texts = [text.lower() for text in texts]

        
        # fit a count vectorizer to learn the vocab
        self.count_vectorizer = CountVectorizer(min_df=20,stop_words='english',max_features=1000)  
        self.X = self.count_vectorizer.fit_transform(texts).toarray()
        
        # train a logistic regression classifier on X and y
        self.model = LogisticRegression()
        self.model.fit(self.X, y)
        pass 


    def predict_sentiment_statistical(self, user_input: str) -> int: 
        """ 
        Uses a trained bag-of-words Logistic Regression classifier to classify the sentiment

        Arguments: 
            - user_input (str) : a user-supplied line of text
        
        Returns: int 
            -1 if the trained classifier predicts -1 
            1 if the trained classifier predicts 1 
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)
        """                                           
        
        # preset sentiment to 0
        sentiment = 0
        
        # use fitted vectorizer to eval the user input into bag of words
        vectorized = self.count_vectorizer.transform([user_input]).toarray()
        
        # if at least 1 word in the input is in the vocab, then predict
        if np.any(vectorized):
            # predict!
            sentiment = self.model.predict(vectorized)[0]

        return sentiment


    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the 
        recommended movie titles. This function needs at least 5 ratings to make a recommendation. 

        Arguments: 
            - user_ratings (dict): 
                - keys are indices of movies 
                  (corresponding to rows in both data/movies.txt and data/ratings.txt) 
                - values are 1, 0, and -1 corresponding to positive, neutral, and 
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend
        """ 
        
        # make sure precondition is satisfied
        assert len(user_ratings.keys()) >= 5
        
        # put user ratings at correct indices in an array of all user ratings
        user_array = np.zeros(len(self.titles))
        for key in user_ratings.keys():
            user_array[key] = user_ratings[key]
        
        # call util.recommend to get indices for recommendations
        indices = util.recommend(user_array, self.ratings, num_return)
        
        # get titles at indices
        films = [self.titles[i][0] for i in indices]
        
        # return those titles as the recommendations
        return films


    def grammarCheck(self, user_input: str):
        """
        This function takes in a user input string.
        It identifies movies without quotation marks, ignoring incorrect capitalization, and rejects titles which are strict substrings of other titles.
        It returns a list of the movies identified in the user input string.
        """
        # initialize empty array of movies
        movies = []
        indices = []
       
        # regex to capture title
        regex = r'(.+)(?:\(\d+\))'
        
        # look for matching movies in self.titles and add them to ret
        for i in range(len(self.titles)):
                
            # match on title, discard year
            matches = re.findall(regex, self.titles[i][0].lower())
            
            # continue if no matches
            if len(matches) == 0:
                continue
            
            # strip the whitespace
            title = matches[0].strip()
            
            regex2 = r'\b' + re.escape(title) + r'\b'
                
            # detect title in user string
            t = re.findall(regex2, user_input.lower())
            if len(t) > 0:
                # add actual title to list if it's not a substring of already-added film
                if all([t[0] not in film for film in movies]):
                    movies.append(self.titles[i][0])
                    indices.append(i)
        # return the resulting list of matched movies
        return movies

    def jumbledCheck(self, title: str):
        """
        This function takes in a title. 
        It matches on self.titles, accounting for articles (e.g. matching An American in Paris with American in Paris, An).
        It returns a list of the matched indices.
        """
        # capture the title split into not-article and article
        regex = r'(.+), ([an|the|a]+)[ |^\)]?'
        
        # match on all titles
        remove_parens = [re.findall(r'^[^\(]*', item[0].lower()) for item in self.titles]
        regexed = [re.findall(regex, item[0]) for item in remove_parens]
        
        # transform into article + not-article form
        titles = [match[0][1] + " " + match[0][0] if len(match)>0 else "" for match in regexed]
        
        # initialize empty array of movies
        indices = []
        
        # look for matching movies in self.titles and add them to ret
        for i in range(len(titles)):
            if title.lower() in titles[i]:
                indices.append(i)
        
        # return the resulting list of matched movies
        return indices


if __name__ == '__main__':
    print('To run Auteur in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')



