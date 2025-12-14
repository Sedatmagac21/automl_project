import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import streamlit as st
import os
from datetime import datetime

# CUDA kullanımını etkinleştir
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def prepare_data(image_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    try:
        class_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        if not class_dirs:
            st.error("Sınıf klasörleri bulunamadı!")
            return None, None
            
        st.success(f"Bulunan sınıf sayısı: {len(class_dirs)}")
        
        train_data = datagen.flow_from_directory(
            image_dir,
            target_size=(64, 64),
            batch_size=64,
            class_mode='categorical',
            subset='training'
        )
        
        val_data = datagen.flow_from_directory(
            image_dir,
            target_size=(64, 64),
            batch_size=64,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_data, val_data
        
    except Exception as e:
        st.error(f"Veri hazırlama hatası: {str(e)}")
        return None, None

def train_model_with_params(train_data, val_data, model_name, params):
    models = {
        'ResNet50': ResNet50,
        'DenseNet121': DenseNet121,
        'VGG16': VGG16
    }
    
    optimizers = {
        'Adam': tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        'Nadam': tf.keras.optimizers.Nadam(learning_rate=params['learning_rate']),
        'SGD': tf.keras.optimizers.SGD(learning_rate=params['learning_rate'], momentum=0.9)
    }
    
    base_model = models[model_name](
        weights='imagenet',
        include_top=False,
        input_shape=(64, 64, 3)
    )
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(len(train_data.class_indices), activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers[params['optimizer']],
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=params['epochs'],
        steps_per_epoch=min(50, len(train_data)),
        validation_steps=min(20, len(val_data))
    )
    
    return model, history.history['val_accuracy'][-1]

def grid_search(train_data, val_data, model_name):
    param_grid = {
        'learning_rate': [0.001, 0.0001, 0.00001],
        'optimizer': ['Adam', 'Nadam', 'SGD'],
        'epochs': [2, 3, 4]
    }
    
    best_model = None
    best_accuracy = 0
    best_params = None
    total_combinations = len(param_grid['learning_rate']) * len(param_grid['optimizer']) * len(param_grid['epochs'])
    current = 0
    
    progress_text = "Grid search ilerlemesi"
    progress_bar = st.progress(0, text=progress_text)
    
    for lr in param_grid['learning_rate']:
        for opt in param_grid['optimizer']:
            for epochs in param_grid['epochs']:
                params = {
                    'learning_rate': lr,
                    'optimizer': opt,
                    'epochs': epochs
                }
                
                st.write(f"Deneniyor: LR={lr}, Optimizer={opt}, Epochs={epochs}")
                model, accuracy = train_model_with_params(train_data, val_data, model_name, params)
                
                if accuracy > best_accuracy:
                    best_model = model
                    best_accuracy = accuracy
                    best_params = params
                
                current += 1
                progress_bar.progress(current / total_combinations, 
                                    text=f"{progress_text} - {int(current/total_combinations*100)}%")
    
    return best_model, best_accuracy, best_params

def predict_with_model(model, image_path, class_indices=None):
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=(64, 64)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0
        
        predictions = model.predict(img_array)
        predicted_class = tf.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        if class_indices:
            class_names = {v: k for k, v in class_indices.items()}
            predicted_label = class_names[predicted_class.numpy()]
        else:
            predicted_label = f"Sınıf_{predicted_class.numpy()}"
        
        return predicted_label, confidence
        
    except Exception as e:
        st.error(f"Tahmin hatası: {str(e)}")
        return None, 0.0

def process_image_data(image_dir):
    try:
        train_data, val_data = prepare_data(image_dir)
        if train_data is None:
            return None, 0, None, None
        
        models = ['ResNet50', 'DenseNet121', 'VGG16']
        best_overall_model = None
        best_overall_accuracy = 0
        best_overall_params = None
        best_model_name = None
        
        for model_name in models:
            st.write(f"\n## {model_name} için grid search yapılıyor...")
            model, accuracy, params = grid_search(train_data, val_data, model_name)
            
            st.write(f"\n{model_name} sonuçları:")
            st.write(f"Doğruluk: {accuracy:.4f}")
            st.write(f"En iyi parametreler: {params}")
            
            if accuracy > best_overall_accuracy:
                best_overall_model = model
                best_overall_accuracy = accuracy
                best_overall_params = params
                best_model_name = model_name
        
        if best_overall_model:
            model_path = f"models/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            best_overall_model.save(model_path)
            
            st.success(f"\nEn iyi model: {best_model_name}")
            st.write(f"En iyi doğruluk: {best_overall_accuracy:.4f}")
            st.write("En iyi parametreler:")
            st.write(f"- Learning Rate: {best_overall_params['learning_rate']}")
            st.write(f"- Optimizer: {best_overall_params['optimizer']}")
            st.write(f"- Epochs: {best_overall_params['epochs']}")
            
        return best_overall_model, best_overall_accuracy, model_path, train_data.class_indices
        
    except Exception as e:
        st.error(f"İşlem hatası: {str(e)}")
        return None, 0, None, None
