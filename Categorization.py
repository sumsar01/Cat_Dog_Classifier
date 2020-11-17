import os, shutil

train_data_original = "./Input_data/train"

print('total images in folder: ', len(os.listdir(train_data_original)))
# Create a Directory where we’ll store our dataset
base_dir = "./Data"
os.mkdir(base_dir)

# directories for the training, validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# directory with test cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# directory with test dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

#Making data split

# copies the first 8750 cat images to train_cats_dir
file_name = ['cat.{}.jpg'.format(i) for i in range(8750)]
for name in file_name:
    src = os.path.join(train_data_original, name)
    dst = os.path.join(train_cats_dir, name)
    shutil.copyfile(src, dst)

# copies the next 2500 cat images to validation_cats_dir
file_name = ['cat.{}.jpg'.format(i) for i in range(8750, 11250)]
for name in file_name:
    src = os.path.join(train_data_original, name)
    dst = os.path.join(validation_cats_dir, name)
    shutil.copyfile(src, dst)


# copies the next 1250 cat images to test_cats_dir
file_name = ['cat.{}.jpg'.format(i) for i in range(11250, 12500)]
for name in file_name:
    src = os.path.join(train_data_original, name)
    dst = os.path.join(test_cats_dir, name)
    shutil.copyfile(src, dst)

# copies the first 8750 dog images to train_cats_dir
file_name = ['dog.{}.jpg'.format(i) for i in range(8750)]
for name in file_name:
    src = os.path.join(train_data_original, name)
    dst = os.path.join(train_dogs_dir, name)
    shutil.copyfile(src, dst)

# copies the next 2500 dog images to validation_cats_dir
file_name = ['dog.{}.jpg'.format(i) for i in range(8750, 11250)]
for name in file_name:
    src = os.path.join(train_data_original, name)
    dst = os.path.join(validation_dogs_dir, name)
    shutil.copyfile(src, dst)


# copies the next 1250 dog images to test_cats_dir
file_name = ['dog.{}.jpg'.format(i) for i in range(11250, 12500)]
for name in file_name:
    src = os.path.join(train_data_original, name)
    dst = os.path.join(test_dogs_dir, name)
    shutil.copyfile(src, dst)



# Seeing the content count of the splits
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))





















