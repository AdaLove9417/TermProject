import os
import shutil

for i in range(0, 3):
    a = open('fold_{0}_data.txt'.format(i))
    line_iter = iter(a.readlines())
    for j in line_iter:
        details = str.split(j, '\t')
        age_range = eval(details[3])
        if isinstance(age_range, tuple):
            from_dir = os.path.join(os.getcwd, 'aligned', details[0], 'landmark' + details[2] + '.' + details[1])
            to_dir = 'imdb'
            if not os.path.exists(to_dir):
                os.mkdir(to_dir)
                to_dir = os.path.join(to_dir, 'train')
                if not os.path.exists(to_dir):
                    os.mkdir((to_dir))
                    to_dir = os.path.join(to_dir, 'ages_{0}_to_{1}'.format(age_range[0], age_range[1]))
                    if not os.path.exists(to_dir):
                        os.mkdir((to_dir))
            shutil.copyfile(from_dir, to_dir)