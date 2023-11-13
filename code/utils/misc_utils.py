def parse_performance(s):
    out = []
    for line in s.split('\n'):
        score = line.split('Pearson ')[-1]
        out.append(float(score))

    print(out)
    print(f'Average score: {round(sum(out)/len(out), 3)}')