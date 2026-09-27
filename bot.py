import os
import time
import uuid
import shutil
import threading
import json
from concurrent.futures import ThreadPoolExecutor
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import tensorflow as tf
from PIL import Image
import numpy as np

MODEL_PATH = "plant_disease_model.h5"
MODEL_BACKUP = "plant_disease_model_backup.h5"
FEEDBACK_DIR = "feedback_dataset" 
IMG_SIZE = (128, 128)
FINETUNE_EPOCHS = 1
FINETUNE_LR = 1e-6

PARKS = {
    "First President's Park": {"lat": 43.1921, "lon": 76.8952},
    "Gorky Park": {"lat": 43.2587, "lon": 76.9531},
    "Botanical Garden": {"lat": 43.2173, "lon": 76.9280},
    "Panfilov Park (28 Guardsmen)": {"lat": 43.2567, "lon": 76.9476},
}

PARK_SUMMARIES = {
    "First President's Park": [
        "Birch_Powdery_mildew — common in wet weather and shade. At least 3 infected plants were found.",
        "Birch_healthy — many healthy specimens in early spring.",
    ],
    "Gorky Park": [
        "Maple — most samples turned out to be healthy.",
        "Rare cases of powdery mildew."
    ],
    "Botanical Garden": [
        "Fraxinus — half of the samples are healthy.",
        "Cases of Fraxinus insect damage were found."
    ],
    "Panfilov Park (28 Guardsmen)": [
        "Coniferous trees — a large number of healthy samples were found.",
        "Isolated cases of rust were detected."
    ],
}

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
if not os.path.exists(MODEL_BACKUP):
    shutil.copy(MODEL_PATH, MODEL_BACKUP)

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Birch_healthy',
    'Birch_powdery_mildew',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Fraxinus_Insect_damage',
    'Fraxinus_healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Maple_healthy',
    'Maple_powdery_mildew',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper_bell_Bacterial_spot',
    'Pepper_bell_healthy',
    'Potato_Early_blight',
    'Potato_healthy',
    'Potato_late_blight',
    'Tomato_Bacterial_Spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotter_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_YellowLeaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

plants = ["Apple", "Birch", "Cherry", "Corn", "Fraxinus", "Grape", "Peach", "Maple", "Pepper_bell", "Potato", "Tomato"]
model_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

def predict_from_path(image_path, chosen_plant):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    x = np.expand_dims(img_array, axis=0)
    with model_lock:
        preds = model.predict(x)[0]
    indices = [i for i, cls in enumerate(class_names) if cls.startswith(chosen_plant)]
    if not indices:
        best_idx = int(np.argmax(preds))
        return class_names[best_idx], float(preds[best_idx])
    filtered = preds[indices]
    best_local = int(np.argmax(filtered))
    real_idx = indices[best_local]
    return class_names[real_idx], float(preds[real_idx])

def is_healthy_class(class_name):
    return "healthy" in class_name.lower()

def finetune_on_example(image_path, class_idx):
    try:
        img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
        x = np.array(img) / 255.0
        x = np.expand_dims(x, 0) 
        y = tf.keras.utils.to_categorical([class_idx], num_classes=len(class_names))
        with model_lock:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                metrics=['accuracy']
            )
            model.fit(x, y, epochs=FINETUNE_EPOCHS, verbose=0)
            model.save(MODEL_PATH)
            print(f"[finetune] done for {image_path} -> class {class_idx}")
    except Exception as e:
        print("Error during finetune:", e)

def save_feedback_example(src_path, class_idx):
    class_name = class_names[class_idx]
    target_dir = os.path.join(FEEDBACK_DIR, class_name)
    os.makedirs(target_dir, exist_ok=True)
    basename = f"{int(time.time())}_{uuid.uuid4().hex}.jpg"
    dest_path = os.path.join(target_dir, basename)
    shutil.move(src_path, dest_path)
    return dest_path

def format_park_summary(park_name: str) -> str:
    items = PARK_SUMMARIES.get(park_name, [])
    if not items:
        return f"📍 {park_name}\nNo summary available for this park yet."
    lines = [f"📍 {park_name}", "Common diseases:"]
    for s in items:
        lines.append(f"• {s}")
    return "\n".join(lines)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(plant, callback_data=f"plant:{plant}")] for plant in plants]
    keyboard.append([InlineKeyboardButton("🩺 General Health Check", callback_data="healthcheck_mode")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "👋 Hello! Select a plant or run a general health check:",
        reply_markup=reply_markup
    )

async def healthcheck_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["awaiting_healthcheck_photo"] = True
    await query.edit_message_text(
        "📸 Great! Send a photo of the leaf so I can determine if it's healthy or diseased."
    )

async def plant_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    plant = query.data.split(":")[1]
    context.user_data["plant"] = plant
    await query.edit_message_text(
        f"✅ You selected: {plant}\nNow send a photo of the leaf 🌿\n"
        f"You can also check the park summaries via: /parks"
    )

async def parks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows, row = [], []
    for name in PARKS.keys():
        row.append(InlineKeyboardButton(name, callback_data=f"park:{name}"))
        if len(row) == 1:  
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    await update.message.reply_text(
        "Select a park to view common plant disease summaries:",
        reply_markup=InlineKeyboardMarkup(rows)
    )

async def park_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    park_name = q.data.split(":", 1)[1]
    context.user_data["chosen_park"] = park_name
    await q.edit_message_text(format_park_summary(park_name))

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not update.message or not update.message.photo:
        return

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = f"temp_{user.id}_{int(time.time())}.jpg"
    await file.download_to_drive(file_path)

    if context.user_data.get("awaiting_healthcheck_photo"):
        context.user_data.pop("awaiting_healthcheck_photo")
        context.user_data["last_image_path"] = file_path

        predicted_class, confidence = predict_from_path(file_path, chosen_plant="")

        if "healthy" in predicted_class.lower():
            result = f"✅ Plant is healthy\n🔎 Prediction: {predicted_class}\n📊 Confidence: {confidence:.2%}"
        else:
            result = f"❌ Plant is diseased\n🔎 Prediction: {predicted_class}\n📊 Confidence: {confidence:.2%}"

        await update.message.reply_text(result)
        return

    plant = context.user_data.get("plant")
    if not plant:
        await update.message.reply_text("❗ Please select a plant first using /start")
        os.remove(file_path)
        return

    predicted_class, confidence = predict_from_path(file_path, plant)
    context.user_data["last_image_path"] = file_path
    context.user_data["last_prediction"] = predicted_class

    keyboard = [
        [InlineKeyboardButton("✅ Correct", callback_data="feedback:correct"),
         InlineKeyboardButton("❌ Incorrect", callback_data="feedback:wrong")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"🌱 Plant: {plant}\n"
        f"🔎 Disease: {predicted_class.replace(plant + '_','')}\n"
        f"📊 Confidence: {confidence:.2%}\n\n"
        "Is this correct?",
        reply_markup=reply_markup
    )

async def feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    fb = query.data.split(":")[1]

    if fb == "correct":
        last_path = context.user_data.pop("last_image_path", None)
        if last_path and os.path.exists(last_path):
            os.remove(last_path)
        await query.edit_message_text("👍 Great! Thanks for the confirmation.")
        return

    plant = context.user_data.get("plant")
    if not plant:
        await query.edit_message_text("❗ Please select a plant first using /start")
        return

    keyboard = []
    options = [(i, class_names[i].replace(plant + "_", "")) 
               for i in range(len(class_names)) if class_names[i].startswith(plant)]
    row = []
    for idx, label in options:
        row.append(InlineKeyboardButton(label, callback_data=f"correction:{idx}"))
        if len(row) == 2:
            keyboard.append(row); row = []
    if row:
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("📷 Upload correct photo", callback_data="correction:upload")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("Sorry to hear that 😕. Select the correct diagnosis or upload a correct photo:", reply_markup=reply_markup)

async def health_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    predicted_class = context.user_data.get("last_prediction")

    if not predicted_class:
        await query.edit_message_text("No data available for analysis.")
        return

    if is_healthy_class(predicted_class):
        result = "✅ Plant is healthy"
    else:
        result = "❌ Plant is diseased"

    await query.edit_message_text(result)

async def health_check_no_plant(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    last_path = context.user_data.get("last_image_path")
    if not last_path or not os.path.exists(last_path):
        await query.edit_message_text("❗ No photo found for analysis. Send a photo of the leaf first.")
        return

    predicted_class, confidence = predict_from_path(last_path, chosen_plant="")

    if "healthy" in predicted_class.lower():
        result = f"✅ Plant is healthy\n🔎 Prediction: {predicted_class}\n📊 Confidence: {confidence:.2%}"
    else:
        result = f"❌ Plant is diseased\n🔎 Prediction: {predicted_class}\n📊 Confidence: {confidence:.2%}"

    await query.edit_message_text(result)

async def correction_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data.split(":")[1]
    if data == "upload":
        context.user_data["awaiting_correct_image"] = True
        await query.edit_message_text("Okay. Send a photo with the correct leaf — I will save it as a fine-tuning example.")
        return
    try:
        chosen_idx = int(data)
    except:
        await query.edit_message_text("Selection error. Please try again.")
        return
    if context.user_data.get("correct_image_path"):
        src = context.user_data.pop("correct_image_path")
    else:
        src = context.user_data.pop("last_image_path", None)
    if not src or not os.path.exists(src):
        await query.edit_message_text("Failed to find image file to save. Please try again.")
        return
    dest = save_feedback_example(src, chosen_idx)
    executor.submit(finetune_on_example, dest, chosen_idx)
    context.user_data.pop("last_prediction", None)
    await query.edit_message_text("Thank you! Example saved, fine-tuning initiated.")

if __name__ == "__main__":
    token = "7876291729:AAHRB8wlt0gFH5yqcl0VI4t4h1BoJlmIzCk"
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("parks", parks_cmd))
    app.add_handler(CallbackQueryHandler(plant_choice, pattern="^plant:"))
    app.add_handler(CallbackQueryHandler(park_callback, pattern="^park:"))
    app.add_handler(CallbackQueryHandler(feedback, pattern="^feedback:"))
    app.add_handler(CallbackQueryHandler(correction_choice, pattern="^correction:"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(health_check, pattern="^healthcheck$"))
    app.add_handler(CallbackQueryHandler(health_check_no_plant, pattern="^healthcheck_no_plant$"))
    app.add_handler(CallbackQueryHandler(healthcheck_mode, pattern="^healthcheck_mode$"))
    print("✅ Bot started. Waiting for photos and commands (/parks)...")
    app.run_polling()
