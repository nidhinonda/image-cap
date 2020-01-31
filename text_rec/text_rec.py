import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\tesseract\\tesseract.exe'

def get_text(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(image)


if __name__ == "__main__":
    print(get_text(cv2.imread('1.png')))