"""
Author: Isaac Monroy
Project Title: Image Captioning Algorithm
Description:
    This code trains a neural network model to caption images from the COCO dataset.
    It first preprocesses the captions and images, then constructs a model using CNN and Transformer architecture.
    The model consists of a CNN encoder, Transformer encoder, and Transformer decoder.
    It uses the InceptionV3 pre-trained model as the base for the image processing.
"""

import tensorflow as tf # Deep Learning Framework
import os # Operating system interfaces
import json # JSON file parsing
import pandas as pd # Data manipulation and analysis
import re # Regular expression operations
import numpy as np # Numerical operations
import time # Time-related functions
import matplotlib.pyplot as plt # Plotting and visualization
import collections # Collections module for specialized container datatypes
import random # Generate random numbers
import requests # HTTP library for Python
from PIL import Image # Python Imaging Library
from tqdm.auto import tqdm # Progress bar library
import pickle # Object serialization

# Load annotations
data = json.load(open('./coco2017/annotations/captions_val2017.json'))
data = data['annotations']
img_cap_pairs = [('%012d.jpg' % sample['image_id'], sample['caption']) for sample in data]

# Create dataframe of image names and captions, with standardized image names and paths
captions = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
captions['image'] = captions['image'].apply(lambda x: f'./coco2017/val2017/{x}')
captions = captions.sample(5000).reset_index(drop=True)

def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')).strip()
    return f'[start] {text} [end]'

# Preprocess captions
captions['caption'] = captions['caption'].apply(preprocess)

# Demonstrate random image-caption pair
random_row = captions.sample(1).iloc[0]
print(random_row.caption)
Image.open(random_row.image)

# Initialize global variables
max_length = 40
size_of_vocab = 5000
batch_size = 64
buffer_size = 1000
embedding_dim = 512
units = 512
epochs = 5

# Create text vectorization layer
tokenizer = tf.keras.layers.TextVectorization(max_tokens=size_of_vocab, standardize=None, output_sequence_length=max_length)
tokenizer.adapt(captions['caption'])
pickle.dump(tokenizer.get_vocabulary(), open('vocab_coco.file', 'wb'))

# Create lookup tables for text conversion
word2idx = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
idx2word = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)

# Create dictionary of image-caption pairs
img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(captions['image'], captions['caption']):
    img_to_cap_vector[img].append(cap)

# Shuffle keys and split into training and validation
img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)
slice_index = int(len(img_keys)*0.80)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

# Create training and validation lists
train_imgs, train_captions = zip(*[(img, cap) for img in img_name_train_keys for cap in img_to_cap_vector[img]])
val_imgs, val_captions = zip(*[(img, cap) for img in img_name_val_keys for cap in img_to_cap_vector[img]])

def load_data(img_path, caption):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    caption = tokenizer(caption)
    return img, caption

# Initialize training and validation datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_captions)).map(load_data, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((val_imgs, val_captions)).map(load_data, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size).batch(batch_size)

# Image augmentation
image_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.2), tf.keras.layers.RandomContrast(0.3),])

def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    output = inception_v3.output
    output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)
    return tf.keras.models.Model(inception_v3.input, output)

# Implementing embeddings
class Embeddings(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(
            vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(
            max_len, embed_dim, input_shape=(None, max_len))
    
    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]
        position_ids = tf.range(start=0, limit=length, delta=1)
        position_ids = tf.expand_dims(position_ids, axis=0)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        return token_embeddings + position_embeddings

# Implementing Transformer encoder layer
class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
    
    def call(self, x, training):
        x = self.layer_norm_1(x)
        x = self.dense(x)
        attn_output = self.attention(
            query=x,
            value=x,
            key=x,
            attention_mask=None,
            training=training
        )
        x = self.layer_norm_2(x + attn_output)
        return x

# Implementing Transformer decoder layer
class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, units, num_heads):
        super().__init__()
        self.embedding = Embeddings(
            tokenizer.vocabulary_size(), embed_dim, max_length)
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)
        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)   

    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)
        combined_mask = None
        padding_mask = None        
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        attn_output_1 = self.attention_1(
            query=embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask=combined_mask,
            training=training
        )
        out_1 = self.layernorm_1(embeddings + attn_output_1)
        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_output,
            key=encoder_output,
            attention_mask=padding_mask,
            training=training
        )
        out_2 = self.layernorm_2(out_1 + attn_output_2)
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        return tf.tile(mask, mult)

# Implementing full model
class ImageCaptioningModel(tf.keras.Model):

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
    
    def compute_loss_and_acc(self, img_embed, captions, training=True):
        encoder_output = self.encoder(img_embed, training=True)
        y_input = captions[:, :-1]
        y_true = captions[:, 1:]
        mask = (y_true != 0)
        y_pred = self.decoder(
            y_input, encoder_output, training=True, mask=mask
        )
        loss = self.calculate_loss(y_true, y_pred, mask)
        acc = self.calculate_accuracy(y_true, y_pred, mask)
        return loss, acc
    
    def train_step(self, batch):
        imgs, captions = batch
        if self.image_aug:
            imgs = self.image_aug(imgs)
        img_embed = self.cnn_model(imgs)
        with tf.GradientTape() as tape:
            loss, acc = self.compute_loss_and_acc(
                img_embed, captions
            )   
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
    
    def test_step(self, batch):
        imgs, captions = batch
        img_embed = self.cnn_model(imgs)
        loss, acc = self.compute_loss_and_acc(
            img_embed, captions, training=False
        )
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

encoder = TransformerEncoderLayer(embedding_dim, 1)
decoder = TransformerDecoderLayer(embedding_dim, units, 8)

# Instantiate model, and define its arguments
cnn_model = CNN_Encoder()
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
)

# Initialize loss function - Cross-Entropy loss
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Adam chosen for optimizer
# Compile the model with the optimizer 
# and loss function
caption_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=cross_entropy
)

# Train model
history = caption_model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

# Obtain the image and preprocess it
# for the model
def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

# Generate caption from the analyzed image
def generate_caption(img_path, add_noise=False):
    img = load_image_from_path(img_path)
    if add_noise:
        noise = tf.random.normal(img.shape)*0.1
        img = img + noise
        img = (img - tf.reduce_min(img))/(tf.reduce_max(img) - tf.reduce_min(img))
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img)
    img_encoded = caption_model.encoder(img_embed, training=False)
    y_inp = '[start]'
    for i in range(max_length-1):
        tokenized = tokenizer([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = caption_model.decoder(
            tokenized, img_encoded, training=False, mask=mask)     
        pred_idx = np.argmax(pred[0, i, :])
        pred_idx = tf.convert_to_tensor(pred_idx)
        pred_word = idx2word(pred_idx).numpy().decode('utf-8')
        if pred_word == '[end]':
            break      
        y_inp += ' ' + pred_word
    y_inp = y_inp.replace('[start] ', '')
    return y_inp

# Test with a random image from the validation set
image_path = val_imgs[random.randint(0, len(val_imgs)-1)]
print(f'Generated caption: {generate_caption(image_path)}')
Image.open(image_path)