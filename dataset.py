import os
import json
import scipy.misc
import numpy as np
import cv2
import preprocessing
from random import shuffle


class MPII_dataset:

    def __init__(self, images_dir, annots_json_filename, input_shape, output_shape, type='train'):
        self.annots_json_filename = annots_json_filename
        self.images_dir = images_dir
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.type = type

        self.annots = []

        self.joints_num = 16
        self.joint_pairs = (
            [0, 5],     # COMMENT ankles
            [1, 4],     # COMMENT knees
            [2, 3],     # COMMENT hips
            [10, 15],   # COMMENT wrists
            [11, 14],   # COMMENT elbows
            [12, 13]    # COMMENT shoulders
        )
        self.color_mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)

        assert (len(input_shape), 3, "Input must have 3 dimensions")
        assert (len(output_shape), 3, "Output must have 3 dimensions")
        assert (input_shape[-1], 3, "Input channels dimension must be three (RGB image)")
        assert (output_shape[-1], self.joints_num, "Output channels dimension must be same as joints number")

        self.create_dataset()

    def create_dataset(self):
        with open(self.annots_json_filename) as f:
            json_parsed = json.loads(f.read())

        for index, value in enumerate(json_parsed):
            if value['isValidation'] == 1.0 and self.type == 'valid' or value['isValidation']  == 0.0 and self.type == 'train':
                self.annots.append(value)

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.output_shape

    def get_dataset_size(self):
        return len(self.annots)

    # TODO shuffle
    def generate_batches(self, batch_size, stacks_num):
        input_batch = np.zeros(shape=(batch_size,) + self.input_shape)
        output_batch = np.zeros(shape=(batch_size,) + self.output_shape)

        while True:
            shuffle(self.annots)
            for index, annotation in enumerate(self.annots):
                input_image, output_labelmaps = self.process_image(
                    annotation=annotation,
                    flip_flag=True,
                    scale_flag=True,
                    rotation_flag=True
                )

                batch_index = index % batch_size

                input_batch[batch_index, :, :, :] = input_image
                output_batch[batch_index, :, :, :] = output_labelmaps

                if batch_index == batch_size-1:
                    output_batch_total = []

                    for _ in range(stacks_num):
                        output_batch_total.append(output_batch)

                    yield input_batch, output_batch_total

    # TODO send original non modified imaga and create one image as data augmentation
    # COMMENT now it is only data augmentation with random modifications
    def process_image(self, annotation, flip_flag, scale_flag, rotation_flag, sigma=1):
        image_filename = annotation['img_paths']
        image = scipy.misc.imread(os.path.join(self.images_dir, image_filename))

        obj_center = np.array(annotation['objpos'])
        obj_joints = np.array(annotation['joint_self'])
        scale = annotation['scale_provided']

        obj_joints_visibilities = obj_joints[:, 2]
        obj_joints = obj_joints[:, :2]

        # COMMENT To avoid joints cropping
        scale *= 1.25

        # TODO change values for probs
        if flip_flag and np.random.sample() > 0:
            image, obj_center, obj_joints = self.flip(
                original_image=image,
                obj_center=obj_center,
                obj_joints=obj_joints
            )
        if scale_flag and np.random.sample() > 0:
            scale *= np.random.uniform(0.75, 1.25)
        if rotation_flag and np.random.sample() > 0:
            angle = np.random.randint(-30, 30)
            image, obj_center, obj_joints = preprocessing.rotate(
                original_image=image,
                obj_center=obj_center,
                obj_joints=obj_joints,
                angle=angle
            )

        # preprocessing.plot_processed_image(image, obj_center, obj_joints, scale, angle)

        image, obj_center, obj_joints = preprocessing.crop(
            original_image=image,
            obj_center=obj_center,
            obj_joints=obj_joints,
            scale=scale
        )

        # preprocessing.plot_processed_image(image, obj_center, obj_joints, scale, angle)

        image, obj_center, obj_joints = preprocessing.resize(
            original_image=image,
            obj_center=obj_center,
            obj_joints=obj_joints,
            shape=self.input_shape[:-1]
        )

        # preprocessing.plot_processed_image(image, obj_center, obj_joints, scale, angle, draw_bbox=False)

        image = self.normalize(original_image=image)

        # preprocessing.plot_processed_image(image, obj_center, obj_joints, scale, angle, draw_bbox=False)

        labelmap_joints = preprocessing.scale_points(
            input_res=image.shape[:-1],
            output_res=self.output_shape[:-1],
            points=obj_joints
        )

        labelmaps = self.generate_labelmaps(
            obj_joints=labelmap_joints,
            obj_joints_visibilities=obj_joints_visibilities,
            sigma=sigma)

        # preprocessing.plot_labelmaps(image, obj_joints, labelmaps, labelmap_joints)
        return image, labelmaps

    def normalize(self, original_image):
        original_image = np.divide(original_image, 255.0)

        for i in range(original_image.shape[-1]):
            original_image[:, :, i] -= self.color_mean[i]
        return original_image

    def flip(self, original_image, obj_center, obj_joints):
        flipped_joints = np.copy(obj_joints)

        im_height, im_width, im_channels = original_image.shape

        flipped_image = cv2.flip(original_image, flipCode=1)  # COMMENT mirrors image x coordinates
        flipped_joints[:, 0] = im_width - obj_joints[:, 0]  # COMMENT mirrors joints x coordinates

        for joint_pair in self.joint_pairs:
            temp = np.copy(flipped_joints[joint_pair[0], :])
            flipped_joints[joint_pair[0], :] = flipped_joints[joint_pair[1], :]
            flipped_joints[joint_pair[1], :] = temp

        flipped_center = np.copy(obj_center)
        flipped_center[0] = im_width - obj_center[0]

        return flipped_image, flipped_center, flipped_joints

    def generate_labelmaps(self, obj_joints, obj_joints_visibilities, sigma):
        labelmaps = np.zeros(shape=(self.output_shape[0], self.output_shape[1], self.joints_num))

        for i in range(len(obj_joints)):
            if obj_joints_visibilities[i] > 0.0:
                labelmaps[:, :, i] = preprocessing.generate_labelmap(labelmaps[:, :, i], obj_joints[i], sigma)

        return labelmaps
