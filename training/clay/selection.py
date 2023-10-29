import random
import shutil
import json
import os

# Define the range and the number of integers to select
lower_bound = 0
upper_bound = 35794
num_integers = 75

# Generate 280 different random integers between 0 and 35794
random_integers = random.sample(range(lower_bound, upper_bound + 1), num_integers)

train_path = "../../../downloads/train.json"
f = open(train_path)
data = json.load(f)
f.close()

for i in range(len(data)):
	if i in random_integers:
		shutil.copy(os.path.join("./combined2", str(data[i]['id'])+".png"), os.path.join("./selection", str(data[i]['id'])+".png"))
		
