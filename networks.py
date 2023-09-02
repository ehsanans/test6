'''
Authors: Jeff Adrion, Andrew Kern, Jared Galloway
'''

from imports import *


def replace_linear_with_Lrelu(model):
    '''
    Modify passed model by replacing swish activation with relu
    '''
    a=[]
    a.append(model.layers[7])
    a.append(model.layers[9])
    for layer in tuple(a):
        layer_type = type(layer).__name__
        if hasattr(layer, 'activation') :
            print(layer_type, layer.activation.__name__)
            if layer_type == "Dense":
                # conv layer with swish activation
                layer.activation = tf.keras.activations.LeakyReLU(alpha=0.6)
            else:
                pass
    return model



def GRU_TUNED84(x,y,trans_flag,pretrained_network):
    '''
    Same as GRU_VANILLA but with dropout AFTER each dense layer.
    '''

    haps,pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    print(trans_flag)
    if not trans_flag:
        print("we train model from zero point")
        genotype_inputs = layers.Input(shape=(numSNPs,numSamps))
        model = layers.Bidirectional(layers.GRU(128,return_sequences=False))(genotype_inputs)
        model = layers.Dense(512)(model)
        model = layers.Dropout(0.45)(model)

        #----------------------------------------------------

        position_inputs = layers.Input(shape=(numPos,))
        m2 = layers.Dense(512)(position_inputs)

        #----------------------------------------------------


        model =  layers.concatenate([model,m2])
        model = layers.Dense(256)(model)
        model = layers.Dropout(0.2)(model)
        model = layers.Dense(64(model)
        model = layers.Dropout(0.2)(model)
        model = layers.Dense(8)(model)
        output = layers.Dense(1)(model)

        #----------------------------------------------------

        model = Model(inputs=[genotype_inputs,position_inputs], outputs=[output])
        


    if trans_flag:
        print("we train model with pre trained model ")
        jsonFILE = open(pretrained_network[0],"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model=model_from_json(loadedModel)
        model.load_weights(pretrained_network[1])

###################################################
        model=replace_linear_with_Lrelu(model) ####
###################################################




        alllayers=model.layers




#####################################################
        layer_fix_ind=[1,2,5]            ############
        for l in layer_fix_ind:          ############
            alllayers[l].trainable=False ############
#####################################################
  
    

    model.compile(optimizer='Adam', loss='mse')
    model.summary()
    print("your need data:")
    input_layer_length=model.layers[0].__dict__['_batch_input_shape'][1]
    print(input_layer_length)

    return model

