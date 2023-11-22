import os

def parse_performance(s):
    out = []
    for line in s.split('\n'):
        score = line.split('Pearson ')[-1]
        out.append(float(score))

    print(out)
    print(f'Average score: {round(sum(out)/len(out), 3)}')

def make_and_clean_up_dirs(dir):
    os.makedirs(dir, exist_ok=True)
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))