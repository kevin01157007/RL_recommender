import numpy as np
user_rec_item = [np.random.choice(range(3202), size=20, replace=False).tolist() for user in range(3)]
print(user_rec_item)