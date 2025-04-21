from mimetypes import init
import fitz   
import cv2  
import numpy as np
import pandas as pd
import os

# Function to Convert PDF to Image
def pdf_to_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):    
        pix = page.get_pixmap()
        img_path = os.path.join(output_folder, f"page_{i+1}.png")   
        pix.save(img_path)
        images.append((img_path, i + 1))  # Store page number along with image path
    return images
 
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte

def extract_tables_from_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read Image in Grayscale
 
    # Apply Sauvola thresholding
    window_size = 25  # You can tweak this based on image quality
    sauvola_thresh = threshold_sauvola(img, window_size=window_size)
    binary = img > sauvola_thresh

    binary = img_as_ubyte(binary)  # Convert boolean array to uint8 image
    binary = cv2.bitwise_not(binary)  # Invert so tables become white on black
 
    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_data = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:  # Filter based on expected table cell size
            table_data.append((x, y, w, h))

    return table_data



def save_tables_to_csv(tables, output_csv):
    if tables:
        df = pd.DataFrame(tables, columns=['X', 'Y', 'Width', 'Height'])

        # Rename the fifth column header to the machine name, with empty values
        df["260 MT-KOBELCO-CKL 2600i"] = ""

        df.to_csv(output_csv, index=False)
        print(f"Table saved to {output_csv}")
    else:
        print(f"No table found in {output_csv}")

# Function to Draw Detected Tables on the Image
def draw_tables_on_image(image_path, tables, output_image_path):
    img = cv2.imread(image_path)
    
    for (x, y, w, h) in tables:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_image_path, img)
    print(f"Visualized image saved to {output_image_path}")


def pdf_to_tables_pipeline(pdf_path, output_folder):
    # Step 1: Convert PDF to Images
    images = pdf_to_images(pdf_path, output_folder)
    
    # Step 2: Extract tables and save CSVs + visualized images
    for image_path, page_num in images:                     
        tables = extract_tables_from_image(image_path)   

        # Step 3: Save CSV
        output_csv = os.path.join(output_folder, f"tables_page_{page_num}.csv")
        save_tables_to_csv(tables, output_csv)

        # Step 4: Save image with bounding boxes
        output_img = os.path.join(output_folder, f"annotated_page_{page_num}.png")
        draw_tables_on_image(image_path, tables, output_img)


# Run Pipeline
pdf_path = r"260 MT-KOBELCO-CKL 2600i.pdf"  # PDF file
output_folder = "output_images3"

# Create output folder if not exists     
os.makedirs(output_folder, exist_ok=True)

# Execute Pipeline
pdf_to_tables_pipeline(pdf_path, output_folder) 