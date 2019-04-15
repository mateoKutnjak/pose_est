from dataset import MPII_dataset
from models import HourglassModel
import tensorflow as tf
import keras.backend as k
import argparse

#TODO epochs implementation

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--num_stacks', type=int, default=2, help='number of hourglass modules')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--images_path', type=str, default='images/', help='path to images data')
    parser.add_argument('--annots_path', type=str, default='annotations.json', help='path to annotations data')
    parser.add_argument('--input_shape', type=int, nargs=3, default=(256, 256, 3), help='input data shape')
    parser.add_argument('--output_shape', type=int, nargs=3, default=(64, 64, 16), help='output data shape')

    args = parser.parse_args()


    mpii_dataset = MPII_dataset(
        images_dir=args.images_path,
        annots_json_filename=args.annots_path,
        input_shape=args.input_shape,
        output_shape=args.output_shape,
        type='train'
    )

    # for input_batch, output_batch in mpii_dataset.create_batches(batch_size=BATCH_SIZE):
    #     import pdb
    #     pdb.set_trace()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    k.tensorflow_backend.set_session(tf.Session(config=config))

    hg_model = HourglassModel(
        dataset=mpii_dataset,
        batch_size=args.batch_size,
        stacks_num=args.num_stacks,
    )

    hg_model.build()
    hg_model.train()

if __name__ == '__main__':
    main()