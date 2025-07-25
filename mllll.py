import cv2
import os
import numpy as np
from deepface import DeepFace

# Directory to save the dataset
dataset_dir = "Dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Load face detector once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create dataset function
def create_dataset(name):
    person_dir = os.path.join(dataset_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("[INFO] Starting to capture face images. Press 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_image = frame[y:y + h, x:x + w]  # capture in color
            face_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(face_path, face_image)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Image Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Dataset created: {count} images saved in {person_dir}")

# Train Dataset
def train_dataset():
    print("[INFO] Starting training process...")
    embeddings = {}
    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_path):
            embeddings[person_name] = []
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                try:
                    embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    embeddings[person_name].append(embedding)
                except Exception as e:
                    print(f"Failed to process image {image_path}: {e}")
    print("[INFO] Training completed.")
    return embeddings

# Recognize the face
def recognize_face(embeddings):
    cap = cv2.VideoCapture(0)

    print("[INFO] Starting face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_image = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            try:
                # Age, gender, emotion
                analysis = DeepFace.analyze(face_image, actions=["age", "gender", "emotion"], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]

                age = analysis["age"]
                emotion = max(analysis["emotion"], key=analysis["emotion"].get)
                gender = analysis["gender"] if isinstance(analysis["gender"], str) else max(analysis["gender"], key=analysis["gender"].get)

                # Face embedding
                face_embedding = DeepFace.represent(face_image, model_name="Facenet", enforce_detection=False)[0]["embedding"]

                # Find best match
                match = "Unknown"
                max_similarity = -1

                for person, person_embeddings in embeddings.items():
                    for emb in person_embeddings:
                        similarity = np.dot(face_embedding, emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(emb))
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match = person

                if max_similarity > 0.7:
                    label = f"{match} ({max_similarity:.2f})"
                else:
                    label = "Unknown"

                display_text = f"{label}, Age: {int(age)}, Gender: {gender}, Emotion: {emotion}"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            except Exception as e:
                print("Failed to analyze face:", e)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Face recognition stopped.")

# Main
if __name__ == "__main__":
    print("\nChoose an option:")
    print("1. Create a new dataset")
    print("2. Train dataset and save embeddings")
    print("3. Recognize face using saved embeddings\n")

    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        name = input("Enter name for the dataset: ")
        create_dataset(name)
    elif choice == "2":
        embeddings = train_dataset()
        np.save("embeddings.npy", embeddings)
        print("[INFO] Embeddings saved to 'embeddings.npy'")
    elif choice == "3":
        if os.path.exists("embeddings.npy"):
            embeddings = np.load("embeddings.npy", allow_pickle=True).item()
            recognize_face(embeddings)
        else:
            print("[ERROR] 'embeddings.npy' not found. Please train the dataset first.")
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
