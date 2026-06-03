"""
Railway Track Defect Detection - Model Training Script
Save this as: train_model.py
Run with: python train_model.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

# Check TensorFlow and GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("=" * 50)

# ============= CONFIGURATION =============
# Adjust these paths to match your dataset location
DATASET_PATH = './dataset'  # Change this to your dataset folder
TRAIN_DIR = os.path.join(DATASET_PATH, 'Train')
VALIDATION_DIR = os.path.join(DATASET_PATH, 'Validation')
TEST_DIR = os.path.join(DATASET_PATH, 'Test')

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

# Output directory for results
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nDataset Path: {DATASET_PATH}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 50)

# ============= VERIFY DATASET STRUCTURE =============
def verify_dataset():
    """Check if dataset structure is correct"""
    print("\n📁 Verifying Dataset Structure...")
    
    required_dirs = [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            print("\nExpected structure:")
            print("dataset/")
            print("  ├── Train/")
            print("  │   ├── Defective/")
            print("  │   └── Non_defective/")
            print("  ├── Validation/")
            print("  │   ├── Defective/")
            print("  │   └── Non_defective/")
            print("  └── Test/")
            print("      ├── Defective/")
            print("      └── Non_defective/")
            return False
        
        # Check for class folders
        class_folders = os.listdir(dir_path)
        print(f"✅ {os.path.basename(dir_path)}: {class_folders}")
        
        # Count images in each class
        for class_folder in class_folders:
            class_path = os.path.join(dir_path, class_folder)
            if os.path.isdir(class_path):
                image_count = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"   - {class_folder}: {image_count} images")
    
    return True

# Verify dataset before proceeding
if not verify_dataset():
    print("\n❌ Dataset verification failed. Please check your dataset structure.")
    exit(1)

print("\n✅ Dataset structure verified!")
print("=" * 50)

# ============= DATA GENERATORS =============
print("\n📊 Creating Data Generators...")

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Only rescaling for validation and test
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = val_test_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print("\n✅ Data generators created successfully!")
    print(f"Class indices: {train_generator.class_indices}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    
except Exception as e:
    print(f"\n❌ Error creating data generators: {e}")
    print("Please check your dataset structure and try again.")
    exit(1)

print("=" * 50)

# ============= MODEL ARCHITECTURE =============
print("\n🧠 Building Model Architecture...")

def create_model():
    """Create model with MobileNetV2 backbone"""
    
    # Load pre-trained base model
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    return model, base_model

model, base_model = create_model()

print("\n✅ Model created successfully!")
print(f"Total parameters: {model.count_params():,}")
model.summary()

print("=" * 50)

# ============= COMPILE MODEL =============
print("\n⚙️ Compiling Model...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

print("✅ Model compiled!")
print("=" * 50)

# ============= CALLBACKS =============
print("\n📝 Setting up Callbacks...")

# Create timestamp for this training run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = os.path.join(OUTPUT_DIR, f'railway_model_{timestamp}.h5')
best_model_path = os.path.join(OUTPUT_DIR, 'railway_defect_model_best.h5')

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        os.path.join(OUTPUT_DIR, f'training_log_{timestamp}.csv')
    )
]

print("✅ Callbacks configured!")
print("=" * 50)

# ============= TRAINING PHASE 1 =============
print("\n🚀 Starting Training Phase 1: Frozen Base Model")
print("=" * 50)

history1 = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Phase 1 Training Complete!")
print("=" * 50)

# ============= FINE-TUNING PHASE 2 =============
print("\n🚀 Starting Training Phase 2: Fine-tuning")
print("=" * 50)

# Unfreeze base model
base_model.trainable = True

# Freeze early layers, fine-tune later layers
for layer in base_model.layers[:100]:
    layer.trainable = False

print(f"Trainable layers: {len([l for l in model.layers if l.trainable])}")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    initial_epoch=len(history1.history['loss']),
    verbose=1
)

print("\n✅ Phase 2 Training Complete!")
print("=" * 50)

# ============= COMBINE HISTORIES =============
history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

# ============= PLOT TRAINING HISTORY =============
print("\n📊 Generating Training Plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
ax1.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
ax1.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history['loss'], label='Train Loss', linewidth=2)
ax2.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
print(f"✅ Training plot saved: {os.path.join(OUTPUT_DIR, 'training_history.png')}")
plt.show()

# ============= EVALUATE ON TEST SET =============
print("\n🎯 Evaluating on Test Set...")
print("=" * 50)

test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator, verbose=1)

f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

print("\n📈 Test Results:")
print(f"  • Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  • Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"  • Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"  • F1-Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
print(f"  • Loss:      {test_loss:.4f}")

# ============= DETAILED PREDICTIONS =============
print("\n🔍 Generating Detailed Predictions...")

test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Classification report
print("\n📋 Classification Report:")
print("=" * 50)
print(classification_report(y_true, y_pred, 
                          target_names=['Non Defective', 'Defective'],
                          digits=4))

# ============= CONFUSION MATRIX =============
print("\n📊 Generating Confusion Matrix...")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non Defective', 'Defective'],
            yticklabels=['Non Defective', 'Defective'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix saved: {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")
plt.show()

# ============= SAVE MODEL =============
print("\n💾 Saving Model...")

# Save the final model
final_model_path = os.path.join(OUTPUT_DIR, 'railway_defect_detection_model.h5')
model.save(final_model_path)
print(f"✅ Final model saved: {final_model_path}")

# Also save in SavedModel format (recommended for deployment)
saved_model_path = os.path.join(OUTPUT_DIR, 'saved_model')
model.save(saved_model_path)
print(f"✅ SavedModel format saved: {saved_model_path}")

# Save model summary to text file
with open(os.path.join(OUTPUT_DIR, 'model_summary.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print(f"✅ Model summary saved: {os.path.join(OUTPUT_DIR, 'model_summary.txt')}")

# ============= SAVE TRAINING REPORT =============
print("\n📄 Generating Training Report...")

report_path = os.path.join(OUTPUT_DIR, 'training_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("RAILWAY TRACK DEFECT DETECTION - TRAINING REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("DATASET INFORMATION:\n")
    f.write(f"  Training samples:   {train_generator.samples}\n")
    f.write(f"  Validation samples: {validation_generator.samples}\n")
    f.write(f"  Test samples:       {test_generator.samples}\n")
    f.write(f"  Class indices:      {train_generator.class_indices}\n\n")
    
    f.write("MODEL CONFIGURATION:\n")
    f.write(f"  Architecture:   MobileNetV2 + Custom Layers\n")
    f.write(f"  Input size:     {IMG_SIZE}x{IMG_SIZE}x3\n")
    f.write(f"  Batch size:     {BATCH_SIZE}\n")
    f.write(f"  Epochs:         {len(history['loss'])}\n")
    f.write(f"  Learning rate:  {LEARNING_RATE}\n\n")
    
    f.write("TEST RESULTS:\n")
    f.write(f"  Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    f.write(f"  Precision:  {test_precision:.4f} ({test_precision*100:.2f}%)\n")
    f.write(f"  Recall:     {test_recall:.4f} ({test_recall*100:.2f}%)\n")
    f.write(f"  F1-Score:   {f1_score:.4f} ({f1_score*100:.2f}%)\n")
    f.write(f"  Loss:       {test_loss:.4f}\n\n")
    
    f.write("CONFUSION MATRIX:\n")
    f.write(f"{cm}\n\n")
    
    f.write("FILES GENERATED:\n")
    f.write(f"  - {final_model_path}\n")
    f.write(f"  - {saved_model_path}\n")
    f.write(f"  - {os.path.join(OUTPUT_DIR, 'training_history.png')}\n")
    f.write(f"  - {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}\n")
    f.write(f"  - {report_path}\n")

print(f"✅ Training report saved: {report_path}")

# ============= COMPLETION =============
print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📁 All outputs saved in: {OUTPUT_DIR}")
print(f"\n🎯 Best model: {best_model_path}")
print(f"🎯 Final model: {final_model_path}")
print("\n💡 Next steps:")
print("  1. Check the training_history.png for training curves")
print("  2. Review the confusion_matrix.png for model performance")
print("  3. Read the training_report.txt for detailed results")
print("  4. Use the saved model in your Streamlit app!")
print("\n" + "=" * 60)