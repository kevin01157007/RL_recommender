pip install -r requirements.txt

conda env create -f environment.yml

rating > 3 => users:6038 movies:3533

過濾掉互動次數少於9次 => users:5950 movies:3532

過濾掉互動次數少於3次電影 => users:5950 movies:3201

過濾掉出現在test.dat但沒出現在train.dat => users:5950 movies:3191
train.dat => users:5950 movies:3191