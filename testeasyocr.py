import easyocr

# Create the OCR reader
reader = easyocr.Reader(['ar', 'en'])

# Read the image
result = reader.readtext("easytest4.png")

# Print results
for bbox, text, confidence in result:
    print(f"Text: {text}")
    print(f"Confidence: {confidence:.2f}")
    print("-" * 40)