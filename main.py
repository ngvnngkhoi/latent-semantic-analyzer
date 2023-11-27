import time
import os
from semantic_analyzer import LSA

#if you get an error in regards to stopwords resources not found uncomment the sections below
# import nltk
# nltk.download('stopwords')
curr_test = 'test1.html'
curr_path = os.path.dirname(os.path.realpath(__file__))
target_path = f'{curr_path}/tests/{curr_test}'

with open(file = target_path, mode = 'r', encoding = 'utf-8') as file:
    html_content = file.read()

start_time = time.time()
LSA(html_content, dbug = False)
end_time = time.time()

print(f'Total runtime: {end_time - start_time}')