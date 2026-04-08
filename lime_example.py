# %%
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import kagglehub
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
# 1. Download dataset and define directories
#path = kagglehub.dataset_download("dansbecker/hot-dog-not-hot-dog")
#print("Path to dataset files:", path)

train_dir = os.path.join(path, 'seefood', 'train')
test_dir  = os.path.join(path, 'seefood', 'test')

model_dir = os.path.join(os.getcwd(), 'saved_models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'hotdog_resnet50.keras')

# %%
# 2. Build data generators from train/test directory structure
train_datagen = ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)
test_datagen  = ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

print("Classes:", train_generator.class_indices)

# %%
# 3. Load existing model if available; otherwise train and save
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"Loaded saved model from: {model_path}")
else:
    base_model = resnet50.ResNet50(weights='imagenet', include_top=False)

    # Freeze all base layers
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator, epochs=5, validation_data=test_generator)
    model.save(model_path)
    print(f"Saved trained model to: {model_path}")

# %%
# 5. Load one test image for LIME explanation
hot_dog_dir = os.path.join(test_dir, 'hot_dog')
img_path = os.path.join(hot_dog_dir, os.listdir(hot_dog_dir)[0])
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_expanded = np.expand_dims(img_array, axis=0)
processed_img = resnet50.preprocess_input(img_array_expanded.copy())

# Wrap predict for LIME (outputs [p_class0, p_class1])
def predict_fn(images):
    preprocessed = resnet50.preprocess_input(images.copy())
    preds = model.predict(preprocessed)
    return np.hstack([1 - preds, preds])


# %%
# 3. Initialize LIME Explainer
explainer = lime_image.LimeImageExplainer()

# 4. Generate Explanation
# num_samples: number of random perturbations to generate (more = better but slower)
explanation = explainer.explain_instance(
    img_array.astype('double'), 
    predict_fn, 
    top_labels=2, 
    hide_color=0, 
    num_samples=500
)

# %%
# 5. Visualize the "Pros" (Positive attributes)
# We choose the top predicted label index
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=False
)

plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title(f"LIME Explanation for Class: {explanation.top_labels[0]}")
plt.show()

# %%
# choose another label index to visualize the "Cons" (Negative attributes)
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[1], 
    positive_only=False, 
    num_features=5, 
    hide_rest=False
)
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title(f"LIME Explanation for Class: {explanation.top_labels[1]}")
plt.show()   
# %%

# %%
# 6. Non-hot-dog example: load a test image and explain it with LIME
not_hot_dog_dir = os.path.join(test_dir, 'not_hot_dog')
non_hot_dog_img_path = os.path.join(not_hot_dog_dir, os.listdir(not_hot_dog_dir)[0])

non_hot_dog_img = image.load_img(non_hot_dog_img_path, target_size=(224, 224))
non_hot_dog_array = image.img_to_array(non_hot_dog_img)

non_hot_dog_explanation = explainer.explain_instance(
    non_hot_dog_array.astype('double'),
    predict_fn,
    top_labels=2,
    hide_color=0,
    num_samples=500
)

# %%
# 7. Visualize influential regions for the non-hot-dog image
temp, mask = non_hot_dog_explanation.get_image_and_mask(
    non_hot_dog_explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)

plt.figure(figsize=(6, 6))
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title(f"Non-Hot-Dog LIME (Class: {non_hot_dog_explanation.top_labels[0]})")
plt.axis('off')
plt.show()

# %%



