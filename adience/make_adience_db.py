import os
import shutil
from PIL import Image
import cv2
for i in range(0, 3):
    a = open('fold_{0}_data.txt'.format(i))
    line_iter = iter(a.readlines())
    next(line_iter)
    for j in line_iter:
        details = str.split(j, '\t')
        if details[3] == '(38, 42)':
            age_range = eval('(38, 43)')
        elif details[3] == '(27, 32)':
            age_range = eval('(25, 32)')
        else:
            age_range = eval(details[3])
        if isinstance(age_range, tuple) and details[3] != '(8, 23)' and details[3] != '(38, 48)':
            from_dir = os.path.join(os.getcwd(), 'aligned', 'aligned', details[0], 'landmark_aligned_face' + '.' + details[2] + '.' + details[1])
            to_dir = 'imdb'
            if not os.path.exists(to_dir):
                os.mkdir(to_dir)
            to_dir = os.path.join(to_dir, 'train')
            if not os.path.exists(to_dir):
                os.mkdir((to_dir))
            to_dir = os.path.join(to_dir, 'ages_{0}_to_{1}'.format(age_range[0], age_range[1]))
            if not os.path.exists(to_dir):
                os.mkdir((to_dir))
            to_dir = os.path.join(to_dir, 'landmark_aligned_face' + '.' + details[2] + '.' + details[1])
        image = cv2.imread(from_dir)
        if image is not None:
            image = cv2.resize(image, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[16:240, 16:240]
            image = Image.fromarray(image)
            image.save(to_dir)

a = open('fold_{0}_data.txt'.format(4))
line_iter = iter(a.readlines())
next(line_iter)
for j in line_iter:
    details = str.split(j, '\t')
    if details[3] == '(38, 42)':
        age_range = eval('(38, 43)')
    elif details[3] == '(27, 32)':
        age_range = eval('(25, 32)')
    else:
        age_range = eval(details[3])
    if isinstance(age_range, tuple):
        from_dir = os.path.join(os.getcwd(), 'aligned', 'aligned', details[0],
                                'landmark_aligned_face' + details[2] + '.' + details[1])
        to_dir = 'imdb'
        if isinstance(age_range, tuple) and details[3] != '(8, 23)' and details[3] != '(38, 48)':
            from_dir = os.path.join(os.getcwd(), 'aligned', 'aligned', details[0], 'landmark_aligned_face' + '.' + details[2] + '.' + details[1])
            to_dir = 'imdb'
            if not os.path.exists(to_dir):
                os.mkdir(to_dir)
            to_dir = os.path.join(to_dir, 'test')
            if not os.path.exists(to_dir):
                os.mkdir((to_dir))
            to_dir = os.path.join(to_dir, 'ages_{0}_to_{1}'.format(age_range[0], age_range[1]))
            if not os.path.exists(to_dir):
                os.mkdir((to_dir))
            to_dir = os.path.join(to_dir, 'landmark_aligned_face' + '.' + details[2] + '.' + details[1])
        image = cv2.imread(from_dir)
        if image is not None:
            image = cv2.resize(image, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[16:240, 16:240]
            image = Image.fromarray(image)
            image.save(to_dir)