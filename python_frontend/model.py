import sys
import os
import cv2
import numpy as np


class ModelTrain:

	NUM_SAMPLES = 200
	CONFIDENCE = 90
	SAVE_FORMAT = "jpg"

	def __init__(self, person : str) -> None :
		self.person = person
		self.dir =        os.path.join("./model_data", person)
		self.img_pth =    os.path.join(self.dir, "images")
		self.model_pth =  os.path.join(self.dir, "face_model.xml")
		self.labels_pth = os.path.join(self.dir, "labels.npy")

		# It would be waste of time and resources if we wont load model once
		# But i did not come up for a better idea then making it class scope vars
		self.recognizer = None
		self.label_dict = None
		self.face_cascade = None


	def _is_trained(self) -> bool:
		return os.path.exists(self.model_pth) and os.path.exists(self.labels_pth)


	def _num_imgs_provided(self):
		if os.path.exists(self.img_pth):
			return sum(1 for file in os.listdir(self.img_pth) if file.lower().endswith(f".{ModelTrain.SAVE_FORMAT}"))
		return 0


	def _save_image(self, person_folder, img, img_ind) -> None :
		print("We need to collect images of you to train a model")
		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

		if not os.path.exists(person_folder):
			os.makedirs(person_folder)

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

		for (x, y, w, h) in faces:
			face_img = gray[y:y+h, x:x+w]
			file_path = os.path.join(person_folder, f"{img_ind}.{ModelTrain.SAVE_FORMAT}")
			cv2.imwrite(file_path, face_img)
			img_ind += 1
			print(f"Collected {img_ind}/{ModelTrain.NUM_SAMPLES}")

		print("Training data collection complete.")


	def _train_model(self, dataset_path) -> None :
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

		images, labels = [], []
		label_dict = {}  # Mapping label numbers to names
		label_id = 0

		label_dict[label_id] = self.person

		for image_name in os.listdir(dataset_path):
			img_path = os.path.join(dataset_path, image_name)

			img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			if img is None:
				continue  # Skip invalid images

			faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
			for (x, y, w, h) in faces:
				face_img = img[y:y+h, x:x+w]  # Extract face region
				face_img = cv2.resize(face_img, (100, 100))
				images.append(face_img)
				labels.append(label_id)

		if len(images) == 0:
			sys.exit("No valid training data found!")
			return

		images = np.array(images, dtype=np.uint8)
		labels = np.array(labels, dtype=np.int32)

		recognizer.train(images, labels)
		recognizer.save(self.model_pth)
		np.save(self.labels_pth, label_dict)
		print("Model training complete.")


	def predict(self, img):
		# Check if model was already trained
		if not self._is_trained():
			provided_imgs = self._num_imgs_provided()
			if provided_imgs < ModelTrain.NUM_SAMPLES :
				self._save_image(self.img_pth, img, provided_imgs)
				return
			self._train_model(self.img_pth)
			return

		if not self.recognizer or not self.label_dict:
			self.recognizer = cv2.face.LBPHFaceRecognizer_create()
			self.recognizer.read(self.model_pth)
			self.label_dict = np.load(self.labels_pth, allow_pickle=True).item()

		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

		for (x, y, w, h) in faces:
			face_img = gray[y:y+h, x:x+w]
			label, confidence = self.recognizer.predict(face_img)
			print(confidence)
			if confidence < ModelTrain.CONFIDENCE:
				name = self.label_dict[label]
				color = (0, 255, 0) if name == self.person else (0, 0, 255)
				cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
				cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			else:
				cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
