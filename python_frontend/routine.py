import cv2
import requests
import numpy as np

from model import ModelTrain

# Custom for every esp32 launch
ESP32_IP="http://192.168.0.111:80"

def run_routine() -> None:
	stream = requests.get(ESP32_IP, stream=True)
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	bytes_data = b''
	mt = ModelTrain("Daniil_Bulgakov")

	for chunk in stream.iter_content(chunk_size=1024):
		bytes_data += chunk
		a = bytes_data.find(b'\xff\xd8')
		b = bytes_data.find(b'\xff\xd9')

		if a != -1 and b != -1:
			jpg = bytes_data[a:b+2]
			bytes_data = bytes_data[b+2:]

			if len(jpg) > 0:
				img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

				if img is not None:
					mt.predict(img)
					cv2.imshow('ESP32-CAM Stream', img)

				if cv2.waitKey(1) == ord('q'):
					break

	cv2.destroyAllWindows()

if __name__ == "__main__":
    run_routine()