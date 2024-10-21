from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq, TextStreamer, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch

class PaliGemmaReader():
  def __init__(self) -> None:
    model_id = "hiyouga/PaliGemma-3B-Chat-v0.1"
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map={"": "mps"})
    # self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map={"": "mps"})
    self.model = self.model.to("mps")
    self.processor = AutoProcessor.from_pretrained(model_id)
    self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

  def system_context(self):
    context = """
    You are an AI model tasked with extracting meter readings from images of bulk flow meters. The meter reading is displayed in a numeric format and indicates the total volume measured by the meter. The images provided will be clear and focused on the meter display. When processing each image, follow these steps:
    1. If the image is blurry or unclear, reply with the message \\"unclear\\"
    2. If the provided image is not of a water meter, return an error message \\"nometer\\"
    3. Extract the digits between vertical lines. The vertical lines are not to be considered as 1.
    4. If the last digit is of different colour, include the digit as a decimal point
    5. Extract the numeric value shown along with the leading zeros
    6. Ensure accuracy by double-checking the numbers for clarity
    Here are a few examples of the meter reading responses:
    1. 0035965
    2. 004583
    3. 16040
    """
    return context

  def reader(self, image):

    pixel_values = self.processor(images=[image], return_tensors="pt").to(self.model.device)["pixel_values"]

    messages = [
      {
        "role": "system", 
        "content": [
          {"type": "text"},
          {"type": "text", "text": self.system_context()}
        ]
      },
      {
        "role": "user", 
        "content": [
          {"type": "text", "text": """Extract the meter reading from the image"""}
        ]
      }
    ]

    input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
    image_prefix = torch.empty((1, getattr(self.processor, "image_seq_length")), dtype=input_ids.dtype).fill_(image_token_id)
    input_ids = torch.cat((image_prefix, input_ids), dim=-1).to(self.model.device)
    output = self.model.generate(input_ids, pixel_values=pixel_values, streamer=self.streamer, max_new_tokens=200)
    print(output)
    result = self.processor.decode(output[0], skip_special_tokens=True)
    print(result)
    
    return result