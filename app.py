from fastai.vision.all import *
import gradio as gr

# Categories for dinosaur types
categories = ("Ankylosaurus", "Brachiosaurus", "Compsognathus","Corythosaurus",
              "Dilophosaurus", "Dimorphodon", "Gallimimus", "Microceratus",
              "Pachycephalosaurus", "Parasaurolophus", "Spinosaurus", "Stegosaurus",
              "Triceratops", "Tyrannosaurus Rex", "Velociraptor")

# Load your trained model
learn = load_learner('model.pkl')

# Function to classify the image
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    top3 = sorted(zip(categories, map(float, probs)), key=lambda x: x[1], reverse=True)[:3]
    return {k: v for k, v in top3}

# Gradio interface
image = gr.Image(image_mode='RGB', height=192, width=192)
label = gr.Label()
examples = ['tyrannosaurus.jpg', 'triceratops.jpg', 'velociraptor.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)