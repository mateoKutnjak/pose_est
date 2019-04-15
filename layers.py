from keras.layers import *
from keras.models import Model
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error


def create_network(input_shape, batch_size, channels, classes, stacks=1):
    input = Input(shape=(input_shape[0], input_shape[1], 3,))

    curr_output = create_front_module(input, channels)
    heatmaps = []

    for i in range(stacks):
        curr_output, heatmap = create_single_hourglass_module(curr_output, channels, classes, str(i))
        heatmaps.append(heatmap)

    model = Model(inputs=input, outputs=heatmaps)
    # TODO On keras github 5e-4
    rms = RMSprop(lr=2.5e-4)
    model.compile(optimizer=rms, loss=mean_squared_error, metrics=["accuracy"])

    return model

def residual_module(input, channels_out, layer_name=''):
    skip = Conv2D(filters=channels_out, kernel_size=(1, 1), padding='same', activation='relu', name=layer_name + '_skip')(input)

    x = Conv2D(filters=channels_out // 2, kernel_size=(1, 1), padding='same', activation='relu', name=layer_name + '_conv_1x1_first')(input)
    x = BatchNormalization()(x)
    x = Conv2D(filters=channels_out // 2, kernel_size=(3, 3), padding='same', activation='relu', name=layer_name + '_conv_3x3_second')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=channels_out, kernel_size=(1, 1), padding='same', activation='relu', name=layer_name + '_conv_1x1_third')(x)
    x = BatchNormalization()(x)

    x = Add(name='')([skip, x])

    return x

def create_front_module(input, channels_out):
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='front_module_conv2d_7x7')(input)
    x = BatchNormalization()(x)
    x = residual_module(x, channels_out // 2, layer_name='front_residual_1')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = residual_module(x, channels_out // 2, layer_name='front_residual_2')
    x = residual_module(x, channels_out, layer_name='front_residual_3')

    return x

def create_single_hourglass_module(input, channels, classes, layer_num):
    f1 = residual_module(input, channels, layer_name=layer_num + '_decreasing_layer_1')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)
    f2 = residual_module(x, channels, layer_name=layer_num + '_decreasing_layer_2')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)
    f3 = residual_module(x, channels, layer_name=layer_num + '_decreasing_layer_3')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f3)
    f4 = residual_module(x, channels, layer_name=layer_num + '_decreasing_layer_4')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

    g1 = residual_module(f1, channels, layer_name=layer_num + '_side_layer_1')
    g2 = residual_module(f2, channels, layer_name=layer_num + '_side_layer_2')
    g3 = residual_module(f3, channels, layer_name=layer_num + '_side_layer_3')
    g4 = residual_module(f4, channels, layer_name=layer_num + '_side_layer_4')

    x = residual_module(x, channels, layer_name=layer_num + '_middle_layer_1')
    x = residual_module(x, channels, layer_name=layer_num + '_middle_layer_2')
    x = residual_module(x, channels, layer_name=layer_num + '_middle_layer_3')

    x = UpSampling2D(size=(2, 2))(x)
    x = Add(name='')([x, g4])
    x = residual_module(x, channels, layer_name=layer_num + '_increasing_layer_1')
    x = UpSampling2D(size=(2, 2))(x)
    x = Add(name='')([x, g3])
    x = residual_module(x, channels, layer_name=layer_num + '_increasing_layer_2')
    x = UpSampling2D(size=(2, 2))(x)
    x = Add(name='')([x, g2])
    x = residual_module(x, channels, layer_name=layer_num + '_increasing_layer_3')
    x = UpSampling2D(size=(2, 2))(x)
    x = Add(name='')([x, g1])
    x = residual_module(x, channels, layer_name=layer_num + '_increasing_layer_4')

    return intermediate_supervision(input, x, channels, classes, layer_num)

def intermediate_supervision(prev_output, curr_output, channels, classes, layer_num):
    x = Conv2D(channels, kernel_size=(1, 1), padding='same', activation='relu', name=layer_num + '_head_conv_1x1_first')(curr_output)
    x = BatchNormalization()(x)
    h1 = Conv2D(channels, kernel_size=(1, 1), padding='same', activation='linear', name=layer_num + '_head_conv_1x1_second')(x)

    heatmaps = Conv2D(classes, kernel_size=(1, 1), padding='same', activation='linear', name=layer_num + '_head_conv_1x1_third-heatmap')(h1)
    h2 = Conv2D(channels, kernel_size=(1, 1), padding='same', activation='linear', name=layer_num + '_head_conv_1x1_forth-heatmap')(heatmaps)

    out = Add()([h1, h2, prev_output])

    return out, heatmaps
