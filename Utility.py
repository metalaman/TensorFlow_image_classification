import os
from shutil import copy2, rmtree
from sklearn import cross_validation
import pandas as pd
import imghdr
from PIL import Image


def split_data():
    '''
    Description: By default, the download_images() function downloads the
    images in Image/ folder in current directory. split_data() creates a random
    stratified split on the list of the images in Images/ and save the train
    and test images in Train/ and Test/ folders in Data/ folder in current
    directory.
    Parameters: None
    Return Value: None
    '''
    image_list = [image for image in os.listdir('Images/')]
    label = [image.split('_')[1].split('.')[0] for image in image_list]
    split = cross_validation.StratifiedShuffleSplit(
        label, 1, test_size=0.3, random_state=0)

    for tr, te in split:
        train_images = [image_list[i] for i in tr]
        test_images = [image_list[i] for i in te]

    print ' -- SPLIT COMPLETE -- '

    try:
        print ' -- Verifying length of training and testing data -- '
        assert len(os.listdir('Data/Train/')) == len(train_images)
        assert len(os.listdir('Data/Val/')) == len(test_images)

    except AssertionError:
        print ' -- Copying Data to Train and Val -- '
        rmtree('Data/Train/')
        rmtree('Data/Val/')
        os.mkdir('Data/Train/')
        os.mkdir('Data/Val/')

        for image in image_list:
            if image in train_images:
                copy2('Images/' + image, 'Data/Train/' + image)
            elif image in test_images:
                copy2('Images/' + image, 'Data/Val/' + image)


def maybe_convert_format(folder_path):
    '''
    Description: Change the format of the image to its correct one.
    Parameters: folder_path
    Return Value: None
    '''
    image_list = os.listdir(folder_path)
    for image in image_list:
        img_format = imghdr.what(folder_path + '/' + image)
        if img_format != image.split('.')[1]:
            os.rename(folder_path + '/' + str(image), folder_path +
                      '/' + str(image.split('.')[0]) + '.' + str(img_format))


def noise_check(image_path):
    '''
    Description: Checks if a image is invalid i.e. if the height or width of
    the image is less than 100 or if the image is only some solid color.
    Parameters: image_path
    Return Value: True - If image is invalid, False otherwise.
    '''
    image = Image.open(image_path)
    w, h = image.size
    if h < 100 or w < 100:
        return True
    extrema = image.convert("L").getextrema()
    if extrema[0] == extrema[1]:
        return True
    return False


def noise_remove(folder_path):
    '''
    Description: Calls noise_check() for every image in folder_path and deletes
    the image if it is invalid. List of invalid images is stored in inval.txt
    Parameters: folder_path
    Return Value: None
    '''
    invalid_images = []
    for image in os.listdir(folder_path):
        if noise_check(folder_path + '/' + image):
            invalid_images.append(image)

    print 'Number of invalid images: ' + str(len(invalid_images))

    print 'ALL INVALID IMAGES REMOVED '
    print 'LIST OF INVALID IMAGES STORED IN inval.txt FILE'
    with open('./inval.txt', 'w') as f:
        for image in invalid_images:
            f.write(image)

    for image in invalid_images:
        os.remove(folder_path + '/' + image)


def download_images(url_file):
    '''
    Description: Using request GET method, downloads images to Image/ folder.
    Corrupt urls i.e. without any content or 404s are stored in corrupt_url
    list.
    Parameters: url_file containing urls of the images. url_file should be a
    dataframe with the urls stored in 'imageUrl' series.
    Return Value: None
    '''
    if not os.path.isdir('./Images/'):
        os.mkdir('./Images')
    df = pd.read_csv(url_file, sep='\t')
    print ' -- Downloading Data -- '
    corrupt_url = []
    dir_images = [image.split('.')[0] for image in os.listdir('Images/')]
    for i, url in enumerate(list(df.imageUrl)):
        image_name = str(i) + '_' + str(df.categoryId[i])
        if image_name not in dir_images:
            try:
                r = requests.get(url, timeout=15)
                if r.status_code is 200 and r.content != '' and 'image' in r.headers[
                        'content-type']:
                    if 'gif' in r.headers['content-type']:
                        image_format = '.gif'
                    elif 'png' in r.headers['content-type']:
                        image_format = '.png'
                    else:
                        image_format = '.jpg'
                    with open('Images/' + image_name + image_format, 'w') as f:
                        f.write(r.content)
                else:
                    corrupt_url.append(url)
            except BaseException:
                corrupt_url.append(url)

    assert df.shape[0] == len(os.listdir('Images/')) + len(corrupt_url)
    print ' -- Downloaded -- '
