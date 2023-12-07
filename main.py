import time
import os
from semantic_analyzer import LSA
import numpy as np
import matplotlib.pyplot as plt

#if you get an error in regards to stopwords resources not found uncomment the sections below
# import nltk
# nltk.download('stopwords')
curr_test = 'test1.html'
curr_path = os.path.dirname(os.path.realpath(__file__))
target_path = f'{curr_path}/tests/{curr_test}'

with open(file = target_path, mode = 'r', encoding = 'utf-8') as file:
    html_content = file.read()

iteration_range = range(100,10000, 5)
t = []

for i in iteration_range:
    print(f'iteration number: {i}')
    start_time = time.time()
    LSA(html_content, False, i)
    end_time = time.time()
    t.append(end_time)

plt.title("Analyzation Complexity Study")
plt.plot(iteration_range, t)
plt.grid()
plt.ylabel('Computational Time')
plt.xlabel('Input Size')
plt.show()

