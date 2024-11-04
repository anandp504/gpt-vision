import streamlit as st
from reader.gpt_reader import GPTReader
from reader.qwen_reader import QwenOCRReader
from reader.llama_reader import LlamaReader
from pathlib import Path
from PIL import Image, ImageOps

@st.cache_resource
def load_model():
   return QwenOCRReader()

reader = load_model()

def resize_image(image, max_height=1000, max_width=1000):
   """Resize the image only if it exceeds the specified dimensions."""
   original_width, original_height = image.size
   
   # Check if resizing is needed
   if original_width > max_width or original_height > max_height:
      # Calculate the new size maintaining the aspect ratio
      aspect_ratio = original_width / original_height
      if original_width > original_height:
         new_width = max_width
         new_height = int(max_width / aspect_ratio)
      else:
         new_height = max_height
         new_width = int(max_height * aspect_ratio)
      
      # Resize the image using LANCZOS for high-quality downscaling
      return image.resize((new_width, new_height), Image.LANCZOS)
   else:
      return image


st.sidebar.markdown("**Jal Jeevan Mission**")
USER = "user"
ASSISTANT = "assistant"

img_file_buffer = st.file_uploader('Upload a PNG/JPEG image')

if img_file_buffer is not None:
   image = Image.open(img_file_buffer)
   # image = Image.open(img_file_buffer).convert("L")
   # image = ImageOps.colorize(image, black="black", white="white")
   display_image = resize_image(image)
   st.image(display_image)
   results = reader.reader(display_image)
   with st.chat_message(ASSISTANT):
      st.write(results)
         