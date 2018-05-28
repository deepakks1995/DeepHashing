from imgaug import augmenters as iaa
import cv2
import os


if __name__ == '__main__':
    first = iaa.Sequential([
        iaa.Crop(px=(0, 64)),  # crop images from each side by 0 to 16px (randomly chosen)
    ])
    second = iaa.Sequential([
        iaa.GaussianBlur((0, 0.6), name="GaussianBlur")
    ])

    fourth = iaa.Sequential([
        iaa.Dropout((0.001, 0.005), per_channel=0.5)
    ])
    fifth = iaa.Sequential([
        iaa.CoarseDropout((0.003, 0.015), size_percent=(0.008, 0.05))
    ])
    sixth = iaa.Sequential([
        iaa.Affine(scale=(1, 1.5))
    ])


    seventh = iaa.Sequential([
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.95, 2.5))
    ])

    eight = iaa.Sequential([
        iaa.Affine(rotate=(-6, 6))
    ])

    nine = iaa.Sequential([
        iaa.Affine(shear=(-6, 6))
    ])

    tenth = iaa.Sequential([
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
    ])
    data_path = 'train_data/FVC2002/Db1/'
    subjects = 1
    samples = 1
    for subjects in range(1, 101):
        i = 9
        for samples in range(1, 9):
            image = cv2.imread(data_path + str(subjects) + '_' + str(samples))
            cv2.imwrite('train_data/gen/' + str(subjects) + '_' + str(samples) + '.png', image)
            for itr in [first, second, fourth, fifth, sixth, seventh, eight, nine, tenth]:
                img = itr.augment_image(image)
                cv2.imwrite('train_data/gen/' + str(subjects) + '_' + str(i) + '.png', img)
                i += 1