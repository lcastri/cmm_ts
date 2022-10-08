# string to search in file
SCORE = 'SCORE: '
MODEL = 'Model: '
grid_model = dict()
with open(r'cluster_logs/best_log.txt', 'r') as fp:
    # read all lines in a list
    lines = fp.readlines()
    for i in range(len(lines)):
        # check if string present on a current line
        if lines[i].find(SCORE) != -1:
            if lines[i + 1].find(MODEL) != -1:
                modelname = lines[i + 1].replace(MODEL, '')
                score = float(lines[i].replace(SCORE, '')[0:7])
                grid_model[modelname] = score

print("MODEL WITH LOWEST LOSS MSE: ", min(grid_model, key = grid_model.get))
